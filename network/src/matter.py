from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import MatterConfig


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


@dataclass
class PhysicsMatterOutput:
    alpha: Tensor
    premultiplied_foreground: Tensor
    straight_foreground: Tensor
    color_transmission: Tensor
    transmittance: Tensor
    refractive_flow: Tensor
    residual: Tensor
    confidence: Tensor


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(_group_count(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(_group_count(channels), channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x + self.body(x))


class PhysicsAwareMatter(nn.Module):
    """Predict a reusable colored refractive foreground operator.

    The operator is ``(G, alpha, color_transmission, u, R, confidence)`` and
    renders as ``G + tau * sample(B_cf, x + u) + R``, where
    ``tau = (1 - alpha) * color_transmission``.  Outside the object support the
    identity operator is enforced: ``G=R=u=alpha=0`` and ``tau=1``.
    """

    IMAGE_CHANNELS = 16  # I, B, signed diff, abs diff, trimap(3), bg uncertainty

    def __init__(self, config: MatterConfig | None = None) -> None:
        super().__init__()
        self.config = config or MatterConfig()
        if self.config.flow_parameterization not in (
            "resolution_fraction",
            "fixed_pixels",
        ):
            raise ValueError(
                "flow_parameterization must be 'resolution_fraction' or 'fixed_pixels'"
            )
        if self.config.max_refractive_flow_fraction <= 0:
            raise ValueError("max_refractive_flow_fraction must be positive")
        if self.config.max_refractive_flow <= 0:
            raise ValueError("max_refractive_flow must be positive")
        width = self.config.width
        self.image_stem = nn.Sequential(
            nn.Conv2d(self.IMAGE_CHANNELS, width, 5, padding=2),
            nn.GroupNorm(_group_count(width), width),
            nn.SiLU(inplace=True),
        )
        self.feature_projection = nn.Sequential(
            nn.Conv2d(self.config.feature_channels, width, 1),
            nn.GroupNorm(_group_count(width), width),
            nn.SiLU(inplace=True),
        )
        self.encoder = nn.Sequential(
            _ResidualBlock(width),
            nn.Conv2d(width, width * 2, 4, stride=2, padding=1),
            nn.GroupNorm(_group_count(width * 2), width * 2),
            nn.SiLU(inplace=True),
            _ResidualBlock(width * 2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(width * 2, width, 4, stride=2, padding=1),
            nn.GroupNorm(_group_count(width), width),
            nn.SiLU(inplace=True),
            _ResidualBlock(width),
        )
        # alpha(1), G(3), refractive displacement(2), confidence(1),
        # per-channel color transmission(3), residual(3).
        self.head = nn.Conv2d(width * 2, 13, 3, padding=1)
        with torch.no_grad():
            self.head.bias[7:10].fill_(self.config.neutral_transmission_bias)

    def forward(
        self,
        frames: Tensor,
        counterfactual_background: Tensor,
        trimap_logits: Tensor,
        non_memory_features: Tensor,
        background_uncertainty: Tensor,
    ) -> PhysicsMatterOutput:
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError("frames must have shape [B,T,3,H,W]")
        if counterfactual_background.shape != frames.shape:
            raise ValueError("counterfactual_background must match frames")
        b, t, _, h, w = frames.shape
        if trimap_logits.shape[:3] != (b, t, 3):
            raise ValueError("trimap_logits must have shape [B,T,3,h,w]")
        if non_memory_features.shape[:3] != (b, t, self.config.feature_channels):
            raise ValueError(
                "non_memory_features channel count does not match MatterConfig.feature_channels"
            )
        if background_uncertainty.shape != (b, t, 1, h, w):
            raise ValueError("background_uncertainty must have shape [B,T,1,H,W]")

        flat_frames = frames.reshape(b * t, 3, h, w)
        flat_bg = counterfactual_background.reshape(b * t, 3, h, w)
        flat_uncertainty = background_uncertainty.reshape(b * t, 1, h, w)
        flat_trimap = trimap_logits.reshape(b * t, 3, *trimap_logits.shape[-2:])
        trimap_probability = F.softmax(
            F.interpolate(flat_trimap, size=(h, w), mode="bilinear", align_corners=False),
            dim=1,
        )
        flat_features = non_memory_features.reshape(
            b * t, self.config.feature_channels, *non_memory_features.shape[-2:]
        )
        flat_features = F.interpolate(
            flat_features, size=(h, w), mode="bilinear", align_corners=False
        )

        signed_difference = flat_frames - flat_bg
        image_input = torch.cat(
            (
                flat_frames,
                flat_bg,
                signed_difference,
                signed_difference.abs(),
                trimap_probability,
                flat_uncertainty,
            ),
            dim=1,
        )
        shallow = self.image_stem(image_input) + self.feature_projection(flat_features)
        deep = self.encoder(shallow)
        decoded = self.decoder(deep)
        if decoded.shape[-2:] != shallow.shape[-2:]:
            decoded = F.interpolate(decoded, size=(h, w), mode="bilinear", align_corners=False)
        raw = self.head(torch.cat((shallow, decoded), dim=1))

        soft_support = 1.0 - trimap_probability[:, 0:1]
        if not self.training and self.config.hard_support_at_inference:
            support = (soft_support >= 0.5).to(raw.dtype)
        else:
            support = soft_support
        alpha = torch.sigmoid(raw[:, 0:1]) * support
        # G is the learned variable. F_std is only derived where alpha is
        # identifiable, which avoids forcing an arbitrary color as alpha -> 0.
        premultiplied_foreground = torch.sigmoid(raw[:, 1:4]) * support
        valid_alpha = alpha >= self.config.straight_foreground_min_alpha
        straight_foreground = torch.where(
            valid_alpha,
            premultiplied_foreground
            / alpha.clamp_min(self.config.straight_foreground_eps),
            torch.zeros_like(premultiplied_foreground),
        ).clamp(0.0, 1.0)
        if self.config.flow_parameterization == "resolution_fraction":
            # Predict a resolution-independent normalized displacement, then
            # convert it to pixel units expected by the RCTrans contract.  The
            # default 0.25 gives approximately 64/128/256 px at 256/512/1024.
            flow_scale = raw.new_tensor(
                (
                    max(w - 1, 1) * self.config.max_refractive_flow_fraction,
                    max(h - 1, 1) * self.config.max_refractive_flow_fraction,
                )
            ).view(1, 2, 1, 1)
        else:
            flow_scale = raw.new_full((1, 2, 1, 1), self.config.max_refractive_flow)
        refractive_flow = torch.tanh(raw[:, 4:6]) * flow_scale * support
        confidence = torch.sigmoid(raw[:, 6:7]) * support
        # The RGB factor represents colored absorption independently from
        # scalar geometric opacity.  A value of one is neutral/colorless.
        learned_color_transmission = torch.sigmoid(raw[:, 7:10])
        if not self.config.use_rgb_transmission:
            learned_color_transmission = learned_color_transmission.mean(
                dim=1,
                keepdim=True,
            ).expand(-1, 3, -1, -1)
        color_transmission = 1.0 - support * (1.0 - learned_color_transmission)
        transmittance = (1.0 - alpha) * color_transmission
        residual = (
            torch.tanh(raw[:, 10:13]) * self.config.residual_scale * support
        )
        if not self.config.use_residual:
            residual = torch.zeros_like(residual)

        def video(x: Tensor) -> Tensor:
            return x.reshape(b, t, *x.shape[1:])

        return PhysicsMatterOutput(
            alpha=video(alpha),
            premultiplied_foreground=video(premultiplied_foreground),
            straight_foreground=video(straight_foreground),
            color_transmission=video(color_transmission),
            transmittance=video(transmittance),
            refractive_flow=video(refractive_flow),
            residual=video(residual),
            confidence=video(confidence),
        )
