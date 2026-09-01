from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import BackgroundConfig


@dataclass
class DirectBackgroundEvidence:
    """Fixed-camera background evidence shared by the complete sequence."""

    observed_background: Tensor  # [B,3,H,W]
    weight: Tensor  # [B,1,H,W]
    coverage: Tensor  # [B,1,H,W]
    uncertainty: Tensor  # [B,1,H,W]

    @property
    def observed_any(self) -> Tensor:
        return self.weight > 0


@dataclass
class RefractiveBackgroundEvidence:
    """Background samples recovered through the transparent-object operator."""

    background: Tensor  # [B,3,H,W]
    weight: Tensor  # [B,1,H,W]


@dataclass
class BackgroundOutput:
    """One reusable counterfactual background asset for a full clip."""

    background: Tensor  # [B,3,H,W]
    uncertainty: Tensor  # [B,1,H,W]
    coverage: Tensor  # [B,1,H,W]
    observed_background: Tensor  # [B,3,H,W]
    direct_coverage: Tensor  # [B,1,H,W]
    inverse_background: Tensor  # [B,3,H,W]
    inverse_coverage: Tensor  # [B,1,H,W]
    evidence_background: Tensor  # [B,3,H,W]
    direct_support: Tensor  # [B,1,H,W], bool
    inverse_support: Tensor  # [B,1,H,W], bool
    true_hole: Tensor  # [B,1,H,W], bool

    def video(self, frame_count: int) -> Tensor:
        """Broadcast the asset without allocating frame-specific copies."""

        return self.background[:, None].expand(-1, frame_count, -1, -1, -1)


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class _DilatedCompletionBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.body = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.GroupNorm(groups, channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, value: Tensor) -> Tensor:
        return self.activation(value + self.body(value))


class _BackgroundHoleCompletionNet(nn.Module):
    """Small deterministic context network with a 65-pixel receptive field."""

    def __init__(self, width: int, dilations: tuple[int, ...]) -> None:
        super().__init__()
        groups = _group_count(width)
        self.stem = nn.Sequential(
            nn.Conv2d(5, width, 3, padding=1),
            nn.GroupNorm(groups, width),
            nn.SiLU(inplace=True),
        )
        self.context = nn.Sequential(
            *[_DilatedCompletionBlock(width, dilation) for dilation in dilations]
        )
        self.head = nn.Sequential(
            nn.Conv2d(width, 3, 3, padding=1),
        )

    def forward(
        self,
        observed: Tensor,
        coverage: Tensor,
        true_hole: Tensor,
    ) -> Tensor:
        inputs = torch.cat((observed, coverage, true_hole.to(observed.dtype)), dim=1)
        residual = self.head(self.context(self.stem(inputs)))
        return (observed + residual).sigmoid()


class MaskedTemporalBackground(nn.Module):
    """Build one fixed-camera counterfactual background for the whole clip.

    Directly visible pixels are robustly aggregated at identical coordinates.
    They are preserved exactly by default. A second differentiable evidence
    source, recovered by inverse refractive splatting, may fill locations that
    the moving object never exposes. The learned completion network is used
    only where neither source provides evidence.
    """

    def __init__(
        self,
        config: BackgroundConfig | None = None,
        diffusion_completion: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = config or BackgroundConfig()
        if self.config.completion_width < 1:
            raise ValueError("completion_width must be positive")
        if not self.config.completion_dilations or any(
            dilation < 1 for dilation in self.config.completion_dilations
        ):
            raise ValueError("completion_dilations must contain positive integers")
        if self.config.exclusion_dilation < 1 or self.config.exclusion_dilation % 2 == 0:
            raise ValueError("exclusion_dilation must be a positive odd integer")
        if not 0.0 < self.config.object_threshold < 1.0:
            raise ValueError("object_threshold must lie strictly between zero and one")
        if self.config.inverse_min_transmittance <= 0:
            raise ValueError("inverse_min_transmittance must be positive")
        if self.config.inverse_weight_scale < 0:
            raise ValueError("inverse_weight_scale must be non-negative")
        if self.config.completion_variant not in ("base", "diffusion"):
            raise ValueError("completion_variant must be 'base' or 'diffusion'")
        self.completion = _BackgroundHoleCompletionNet(
            self.config.completion_width,
            self.config.completion_dilations,
        )
        self.diffusion_completion = diffusion_completion

    def _dilate_exclusion(self, probability: Tensor) -> Tensor:
        b, t, _, h, w = probability.shape
        flat = probability.reshape(b * t, 1, h, w)
        dilated = F.max_pool2d(
            flat,
            self.config.exclusion_dilation,
            stride=1,
            padding=self.config.exclusion_dilation // 2,
        )
        return dilated.reshape(b, t, 1, h, w)

    def observe(
        self,
        frames: Tensor,
        exclusion_probability: Tensor,
    ) -> DirectBackgroundEvidence:
        """Aggregate object-free observations at the same fixed-camera pixel."""

        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError("frames must have shape [B,T,3,H,W]")
        expected = (frames.shape[0], frames.shape[1], 1, *frames.shape[-2:])
        if exclusion_probability.shape != expected:
            raise ValueError("exclusion_probability must have shape [B,T,1,H,W]")
        if frames.shape[1] == 0:
            raise ValueError("frames must contain at least one frame")

        exclusion = self._dilate_exclusion(exclusion_probability).clamp(0, 1)
        # Reject pixels classified as object while retaining a soft weight on
        # visible pixels. This prevents tiny sigmoid tails inside the object
        # from being treated as high-priority direct background evidence.
        visible = (exclusion < self.config.object_threshold).to(frames.dtype)
        weight = (1.0 - exclusion).clamp(0, 1) * visible
        first_sum = weight.sum(dim=1)
        first_mean = (frames * weight).sum(dim=1) / first_sum.clamp_min(
            self.config.eps
        )
        residual = torch.linalg.vector_norm(
            frames - first_mean[:, None], dim=2, keepdim=True
        )
        delta = max(self.config.robust_color_delta, self.config.eps)
        robust_weight = weight / (1.0 + (residual / delta).square())
        robust_sum = robust_weight.sum(dim=1)
        observed = (frames * robust_weight).sum(dim=1) / robust_sum.clamp_min(
            self.config.eps
        )
        variance = (
            (frames - observed[:, None]).square().mean(dim=2, keepdim=True)
            * robust_weight
        ).sum(dim=1) / robust_sum.clamp_min(self.config.eps)

        observed_any = robust_sum > self.config.minimum_observation_weight
        coverage = (self.config.coverage_scale * robust_sum).clamp(0, 1)
        support_uncertainty = 1.0 - coverage
        variance_uncertainty = 1.0 - torch.exp(
            -self.config.variance_scale * variance
        )
        uncertainty = torch.where(
            observed_any,
            torch.maximum(support_uncertainty, variance_uncertainty),
            torch.ones_like(robust_sum),
        )
        return DirectBackgroundEvidence(
            observed_background=observed,
            weight=torch.where(observed_any, robust_sum, torch.zeros_like(robust_sum)),
            coverage=coverage,
            uncertainty=uncertainty.clamp(0, 1),
        )

    def _complete(
        self,
        evidence_background: Tensor,
        evidence_coverage: Tensor,
        true_hole: Tensor,
        completion_variant: Literal["base", "diffusion"],
    ) -> Tensor:
        if not self.config.use_temporal_completion:
            return torch.zeros_like(evidence_background)
        if completion_variant == "base":
            return self.completion(
                evidence_background,
                evidence_coverage,
                true_hole,
            )
        if self.diffusion_completion is None:
            raise RuntimeError(
                "PRISM-Diffusion requires a diffusion completion backend"
            )
        return self.diffusion_completion(
            evidence_background,
            evidence_coverage,
            true_hole,
        )

    def fuse(
        self,
        evidence: DirectBackgroundEvidence,
        refractive: RefractiveBackgroundEvidence | None = None,
        *,
        completion_variant: Literal["base", "diffusion"] | None = None,
    ) -> BackgroundOutput:
        """Fuse direct and inverse-refracted evidence into one asset."""

        variant = completion_variant or self.config.completion_variant
        if variant not in ("base", "diffusion"):
            raise ValueError("completion_variant must be 'base' or 'diffusion'")
        direct_any = evidence.weight > self.config.minimum_observation_weight
        if refractive is None:
            inverse_background = torch.zeros_like(evidence.observed_background)
            inverse_weight = torch.zeros_like(evidence.weight)
        else:
            if refractive.background.shape != evidence.observed_background.shape:
                raise ValueError("refractive background shape must match direct evidence")
            if refractive.weight.shape != evidence.weight.shape:
                raise ValueError("refractive weight shape must match direct evidence")
            inverse_background = refractive.background
            inverse_weight = refractive.weight

        inverse_any = inverse_weight > self.config.minimum_observation_weight
        inverse_coverage = inverse_weight.clamp(0, 1)
        if self.config.preserve_direct_observations:
            evidence_background = torch.where(
                direct_any,
                evidence.observed_background,
                torch.where(
                    inverse_any,
                    inverse_background,
                    torch.zeros_like(evidence.observed_background),
                ),
            )
        else:
            total_weight = evidence.weight + inverse_weight
            fused = (
                evidence.observed_background * evidence.weight
                + inverse_background * inverse_weight
            ) / total_weight.clamp_min(self.config.eps)
            evidence_background = torch.where(
                total_weight > 0,
                fused,
                torch.zeros_like(evidence.observed_background),
            )

        total_coverage = torch.maximum(evidence.coverage, inverse_coverage)
        true_hole = ~(direct_any | inverse_any)
        completed = self._complete(
            evidence_background,
            total_coverage,
            true_hole,
            variant,
        )
        # Generative completion is never allowed to redraw physical evidence.
        background = torch.where(true_hole, completed, evidence_background)
        inverse_uncertainty = 1.0 - inverse_coverage
        uncertainty = torch.where(
            direct_any,
            evidence.uncertainty,
            torch.where(inverse_any, inverse_uncertainty, torch.ones_like(inverse_weight)),
        ).clamp(0, 1)
        return BackgroundOutput(
            background=background,
            uncertainty=uncertainty,
            coverage=total_coverage,
            observed_background=evidence.observed_background,
            direct_coverage=evidence.coverage,
            inverse_background=inverse_background,
            inverse_coverage=inverse_coverage,
            evidence_background=evidence_background,
            direct_support=direct_any,
            inverse_support=inverse_any,
            true_hole=true_hole,
        )

    def forward(
        self,
        frames: Tensor,
        exclusion_probability: Tensor,
    ) -> BackgroundOutput:
        return self.fuse(self.observe(frames, exclusion_probability))
