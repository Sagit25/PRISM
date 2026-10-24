from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import BackgroundConfig
from .ffc_completion import GLaMaCompletionNet


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
    variance: Tensor | None = None  # [B,1,H,W], cross-sample RGB variance
    flow_canvas: Tensor | None = None  # [B,2,H,W], splatted u at background coords


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
        editable_mask: Tensor,
    ) -> Tensor:
        inputs = torch.cat((observed, coverage, editable_mask.to(observed.dtype)), dim=1)
        residual = self.head(self.context(self.stem(inputs)))
        return (observed + residual).sigmoid()


class MaskedTemporalBackground(nn.Module):
    """Build one fixed-camera counterfactual background for the whole clip.

    Directly visible pixels are robustly aggregated at identical coordinates.
    They are preserved exactly by default. A second differentiable evidence
    source, recovered by inverse refractive splatting, provides a fallible
    hint where the moving object never exposes the background. By default the
    learned completion network can correct that hint; only direct observations
    remain immutable.
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
        if self.config.completion_backbone not in ("ffc", "dilated"):
            raise ValueError("completion_backbone must be 'ffc' or 'dilated'")
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
        if self.config.inverse_min_reliable_weight < 0:
            raise ValueError("inverse_min_reliable_weight must be non-negative")
        if self.config.inverse_variance_scale < 0:
            raise ValueError("inverse_variance_scale must be non-negative")
        if self.config.inverse_radiance_tolerance < 0:
            raise ValueError("inverse_radiance_tolerance must be non-negative")
        if not 0 < self.config.flux_direct_observation_threshold <= 1:
            raise ValueError("flux_direct_observation_threshold must be in (0, 1]")
        if self.config.flux_min_consistent_observations < 1:
            raise ValueError("flux_min_consistent_observations must be positive")
        if not 0 <= self.config.diffusion_inverse_feedback_strength <= 1:
            raise ValueError("diffusion_inverse_feedback_strength must be in [0, 1]")
        if self.config.diffusion_inverse_feedback_max_delta <= 0:
            raise ValueError("diffusion_inverse_feedback_max_delta must be positive")
        if self.config.diffusion_inverse_feedback_agreement_scale < 0:
            raise ValueError("diffusion_inverse_feedback_agreement_scale must be non-negative")
        if (
            self.config.diffusion_inverse_overlap_kernel < 1
            or self.config.diffusion_inverse_overlap_kernel % 2 == 0
        ):
            raise ValueError("diffusion_inverse_overlap_kernel must be positive and odd")
        if not 0 < self.config.diffusion_inverse_overlap_min_support <= 1:
            raise ValueError("diffusion_inverse_overlap_min_support must be in (0, 1]")
        if not 0 <= self.config.diffusion_inverse_validated_strength <= 1:
            raise ValueError("diffusion_inverse_validated_strength must be in [0, 1]")
        if self.config.diffusion_inverse_validated_max_delta <= 0:
            raise ValueError("diffusion_inverse_validated_max_delta must be positive")
        if self.config.completion_variant not in ("base", "diffusion"):
            raise ValueError("completion_variant must be 'base' or 'diffusion'")
        if self.config.evidence_flux_only and not self.config.use_temporal_completion:
            raise ValueError("evidence_flux_only requires temporal completion")
        if self.config.completion_backbone == "ffc":
            self.completion = GLaMaCompletionNet(
                width=self.config.completion_width,
                down_blocks=self.config.completion_down_blocks,
                residual_blocks=self.config.completion_residual_blocks,
                global_ratio=self.config.completion_global_ratio,
                max_channels=self.config.completion_max_channels,
            )
        else:
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
        direct_threshold = (
            min(
                self.config.object_threshold,
                self.config.flux_direct_observation_threshold,
            )
            if self.config.evidence_flux_only
            else self.config.object_threshold
        )
        visible = (exclusion < direct_threshold).to(frames.dtype)
        weight = (1.0 - exclusion).clamp(0, 1) * visible
        # A mean-centered robust estimator cannot identify a clean frame when
        # one missed object pixel and one clean pixel have equal visibility:
        # both lie equally far from their mean, so the object color becomes
        # immutable "direct" background. Use a temporal median center and
        # reject ambiguous ties instead. At least a strict majority of the
        # candidate visible frames must agree within the color tolerance.
        visible_count = visible.sum(dim=1, keepdim=True)
        sorted_frames = frames.masked_fill(visible == 0, float("inf")).sort(dim=1).values
        center_index = ((visible_count.long() - 1).clamp_min(0) // 2).expand(
            -1, -1, frames.shape[2], -1, -1
        )
        center = sorted_frames.gather(dim=1, index=center_index).squeeze(1)
        center = torch.where(
            visible_count[:, 0] > 0,
            center,
            torch.zeros_like(center),
        )
        residual = torch.linalg.vector_norm(
            frames - center[:, None], dim=2, keepdim=True
        )
        delta = max(self.config.robust_color_delta, self.config.eps)
        inlier = visible * (residual <= delta).to(frames.dtype)
        inlier_count = inlier.sum(dim=1)
        consensus = (2 * inlier_count) > visible_count[:, 0]
        if (
            self.config.evidence_flux_only
            and self.config.flux_require_unanimous_direct_consensus
        ):
            # A slowly moving object can cover one pixel in three of four
            # frames while its semantic mask misses that pixel. Majority
            # voting would call the object color a certain background value.
            # A genuine static background should agree in every candidate
            # frame within the same tolerance; otherwise leave it editable.
            consensus = consensus & (inlier_count == visible_count[:, 0])
        if self.config.evidence_flux_only:
            consensus = consensus & (
                inlier_count >= self.config.flux_min_consistent_observations
            )
        robust_weight = (
            weight * inlier / (1.0 + (residual / delta).square())
        )
        robust_sum = robust_weight.sum(dim=1)
        observed = (frames * robust_weight).sum(dim=1) / robust_sum.clamp_min(
            self.config.eps
        )
        variance = (
            (frames - observed[:, None]).square().mean(dim=2, keepdim=True)
            * robust_weight
        ).sum(dim=1) / robust_sum.clamp_min(self.config.eps)

        observed_any = (robust_sum > self.config.minimum_observation_weight) & consensus
        observed = torch.where(observed_any, observed, torch.zeros_like(observed))
        coverage = torch.where(
            observed_any,
            (self.config.coverage_scale * robust_sum).clamp(0, 1),
            torch.zeros_like(robust_sum),
        )
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
        editable_mask: Tensor,
        completion_variant: Literal["base", "diffusion"],
        refractive: RefractiveBackgroundEvidence | None = None,
        object_reference: Tensor | None = None,
    ) -> Tensor:
        if not self.config.use_temporal_completion:
            return torch.zeros_like(evidence_background)
        if completion_variant == "base":
            return self.completion(
                evidence_background,
                evidence_coverage,
                editable_mask,
            )
        if self.diffusion_completion is None:
            raise RuntimeError(
                "PRISM-Diffusion requires a diffusion completion backend"
            )
        if getattr(self.diffusion_completion, "flow_latent_guidance", False):
            if refractive is None or refractive.flow_canvas is None:
                raise RuntimeError("flow-guided inpainting requires refractive flow evidence")
            extra = {}
            if getattr(self.diffusion_completion, "requires_object_reference", False):
                if object_reference is None:
                    raise ValueError("ObjectClear requires an observed object-reference frame")
                extra["object_reference"] = object_reference
            return self.diffusion_completion(
                evidence_background,
                evidence_coverage,
                editable_mask,
                flow_evidence=(
                    refractive.background,
                    refractive.weight,
                    refractive.flow_canvas,
                ),
                **extra,
            )
        return self.diffusion_completion(
            evidence_background, evidence_coverage, editable_mask
        )

    def fuse(
        self,
        evidence: DirectBackgroundEvidence,
        refractive: RefractiveBackgroundEvidence | None = None,
        *,
        completion_variant: Literal["base", "diffusion"] | None = None,
        object_reference: Tensor | None = None,
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
            if refractive.variance is not None:
                if refractive.variance.shape != inverse_weight.shape:
                    raise ValueError("refractive variance shape must match weight")
                inverse_weight = inverse_weight * torch.exp(
                    -self.config.inverse_variance_scale
                    * refractive.variance.clamp_min(0.0)
                )

        inverse_any = inverse_weight > max(
            self.config.minimum_observation_weight,
            self.config.inverse_min_reliable_weight,
        )
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
        if self.config.evidence_flux_only:
            # Keep fixed-point updates cheap. Inverse evidence is useful as a
            # hint for those updates, but it is estimated from predicted G,
            # tau, R and flow, not a direct background observation. At the
            # final pass let the configured pretrained prior fill non-direct
            # pixels, then admit only a
            # small agreement-gated inverse correction. Otherwise object-
            # colored inverse errors become immutable patches.
            if variant == "diffusion":
                refine_inverse = self.config.refine_inverse_evidence
                completion_mask = ~direct_any if refine_inverse else true_hole
                completion_input = (
                    evidence.observed_background
                    if refine_inverse else evidence_background
                )
                completion_coverage = (
                    evidence.coverage if refine_inverse else total_coverage
                )
                flux_completed = self._complete(
                    completion_input,
                    completion_coverage,
                    completion_mask,
                    "diffusion",
                    refractive,
                    object_reference,
                )
                if refine_inverse:
                    # The released inpainting core is a frozen prior. Let reliable
                    # flow-guided inverse
                    # samples make a bounded, differentiable correction. Where
                    # inverse and direct observations overlap, their *measured*
                    # local agreement calibrates trust without using GT or
                    # treating predicted inverse pixels as immutable evidence.
                    difference = inverse_background - flux_completed
                    prior_agreement = torch.exp(
                        -self.config.diffusion_inverse_feedback_agreement_scale
                        * difference.detach().abs().mean(dim=1, keepdim=True)
                    )
                    overlap = (direct_any & inverse_any).to(inverse_coverage.dtype)
                    kernel = self.config.diffusion_inverse_overlap_kernel
                    padding = kernel // 2
                    local_support = F.avg_pool2d(
                        overlap * inverse_coverage.detach(),
                        kernel,
                        stride=1,
                        padding=padding,
                    )
                    local_error = F.avg_pool2d(
                        overlap
                        * inverse_coverage.detach()
                        * (inverse_background - evidence.observed_background)
                        .detach()
                        .abs()
                        .mean(dim=1, keepdim=True),
                        kernel,
                        stride=1,
                        padding=padding,
                    ) / local_support.clamp_min(self.config.eps)
                    validated = local_support >= self.config.diffusion_inverse_overlap_min_support
                    overlap_agreement = torch.exp(
                        -self.config.diffusion_inverse_feedback_agreement_scale
                        * local_error
                    )
                    # Both overlap consistency and the prior discrepancy must
                    # pass. This prevents one good overlap pixel from licensing
                    # a distant object-colored inverse sample in its window.
                    validated_agreement = (overlap_agreement * prior_agreement).sqrt()
                    agreement = torch.where(
                        validated, validated_agreement, prior_agreement
                    )
                    strength = torch.where(
                        validated,
                        self.config.diffusion_inverse_validated_strength,
                        self.config.diffusion_inverse_feedback_strength,
                    )
                    feedback_limit = torch.where(
                        validated,
                        self.config.diffusion_inverse_validated_max_delta,
                        self.config.diffusion_inverse_feedback_max_delta,
                    )
                    feedback_weight = (
                        strength * inverse_coverage.clamp(0.0, 1.0)
                        * agreement
                        * inverse_any.to(inverse_coverage.dtype)
                    )
                    feedback = feedback_weight * feedback_limit * torch.tanh(
                        difference / feedback_limit
                    )
                    background = torch.where(
                        direct_any, evidence.observed_background, flux_completed + feedback
                    )
                else:
                    background = torch.where(
                        true_hole, flux_completed, evidence_background
                    )
            else:
                background = evidence_background
        else:
            refine_inverse = (
                self.config.refine_inverse_evidence
                and self.config.use_temporal_completion
            )
            # Inverse samples are estimated from predicted G, tau, R and flow.
            # Pass them as context, but make every non-direct pixel editable so a
            # mistaken sample cannot become an immutable object-colored patch.
            completion_mask = ~direct_any if refine_inverse else true_hole
            base_completed = self._complete(
                evidence_background,
                total_coverage,
                completion_mask,
                "base" if refine_inverse else variant,
            )
            if variant == "diffusion" and refine_inverse:
                # Keep the expensive prior restricted to genuine holes. The
                # trained base network corrects inverse-only pixels.
                diffusion_completed = self._complete(
                    evidence_background,
                    total_coverage,
                    true_hole,
                    "diffusion",
                    refractive,
                    object_reference,
                )
                completed = torch.where(true_hole, diffusion_completed, base_completed)
            else:
                completed = base_completed
            if refine_inverse:
                background = torch.where(
                    direct_any,
                    evidence.observed_background,
                    completed,
                )
            else:
                background = torch.where(true_hole, completed, evidence_background)
        inverse_uncertainty = 1.0 - inverse_coverage
        uncertainty = torch.where(
            direct_any,
            evidence.uncertainty,
            torch.where(inverse_any, inverse_uncertainty, torch.ones_like(inverse_weight)),
        ).clamp(0, 1)
        if (
            self.config.evidence_flux_only
            and variant == "diffusion"
            and self.config.refine_inverse_evidence
        ):
            # The final non-direct pixels are inpainted, not measured by the
            # inverse operator. Do not label them as high-confidence evidence.
            uncertainty = torch.where(
                direct_any, uncertainty, torch.ones_like(uncertainty)
            )
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
