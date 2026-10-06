from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import MatterConfig


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0 and channels // groups >= 2:
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
    # Optional for backwards-compatible test doubles and external callers.
    # Full PRISM predicts both tensors on every forward.
    refractive_kernel_weights: Tensor | None = None  # [B,T,K,H,W]
    refractive_kernel_flows: Tensor | None = None  # [B,T,K,2,H,W]
    # Optional object-surface geometry. These are deliberately appended to
    # the v17 operator contract so existing callers and checkpoints remain
    # valid when MatterConfig.predict_geometry is disabled.
    surface_normal: Tensor | None = None  # [B,T,3,H,W], signed world-space
    depth: Tensor | None = None  # [B,T,1,H,W], positive camera distance
    geometry_confidence: Tensor | None = None  # [B,T,1,H,W]


class _ConvNeXtBlock(nn.Module):
    """Large-receptive-field residual block without batch-size dependence."""

    def __init__(self, channels: int, expansion: int = 4) -> None:
        super().__init__()
        hidden = channels * expansion
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            7,
            padding=3,
            groups=channels,
        )
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.expand = nn.Conv2d(channels, hidden, 1)
        self.contract = nn.Conv2d(hidden, channels, 1)
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), 1e-6))

    def forward(self, value: Tensor) -> Tensor:
        residual = self.contract(F.gelu(self.expand(self.norm(self.depthwise(value)))))
        return value + self.scale * residual


class _TemporalContextBlock(nn.Module):
    """Parallel temporal mixing at the compact bottleneck resolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.temporal = nn.Conv3d(
            channels,
            channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=channels,
            bias=False,
        )
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.mix = nn.Conv3d(channels, channels, 1)
        hidden = max(channels // 4, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, value: Tensor) -> Tensor:
        volume = value.permute(0, 2, 1, 3, 4)
        mixed = self.mix(F.silu(self.norm(self.temporal(volume)), inplace=True))
        output = volume + mixed * self.gate(mixed)
        return output.permute(0, 2, 1, 3, 4)


class _DecoderStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        depth: int,
    ) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[_ConvNeXtBlock(out_channels) for _ in range(depth)]
        )

    def forward(self, value: Tensor, skip: Tensor) -> Tensor:
        value = F.interpolate(
            value,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.blocks(self.project(torch.cat((value, skip), dim=1)))


class PhysicsAwareMatter(nn.Module):
    """High-capacity spatio-temporal colored refractive operator predictor.

    The model fuses full-resolution photometric evidence with clean SAM2
    features at a compact bottleneck, mixes every frame in parallel with
    temporal Conv3d blocks, and decodes through two skip-connected scales.

    In addition to the supervised mean flow ``u``, the head predicts a local
    deformable kernel ``{w_k, u_k}``. Rendering becomes

    ``I = G + tau * sum_k w_k B_cf(x + u_k) + R``.

    The weighted expectation ``sum_k w_k u_k`` is returned as
    ``refractive_flow`` and remains directly compatible with the RCTrans
    single-correspondence ground truth and all existing metrics.
    """

    IMAGE_CHANNELS = 16  # I, B, signed diff, abs diff, trimap(3), uncertainty
    ARCHITECTURE = "prism-pam-multiscale-temporal-deformable-kernel-v1"

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
        if len(self.config.encoder_depths) != 3 or any(
            depth < 1 for depth in self.config.encoder_depths
        ):
            raise ValueError("encoder_depths must contain three positive depths")
        if self.config.temporal_blocks < 0:
            raise ValueError("temporal_blocks must be non-negative")
        kernel_size = self.config.refractive_kernel_size
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("refractive_kernel_size must be a positive odd integer")
        if self.config.refractive_kernel_radius_fraction < 0:
            raise ValueError("refractive_kernel_radius_fraction must be non-negative")
        if self.config.geometry_min_depth <= 0:
            raise ValueError("geometry_min_depth must be positive")

        width = self.config.width
        if width < 1 or self.config.max_channels < width:
            raise ValueError("matter width/max_channels are invalid")
        middle = min(width * 2, self.config.max_channels)
        bottleneck = min(width * 4, self.config.max_channels)
        depth0, depth1, depth2 = self.config.encoder_depths

        self.image_stem = nn.Sequential(
            nn.Conv2d(self.IMAGE_CHANNELS, width, 7, padding=3),
            nn.GroupNorm(_group_count(width), width),
            nn.SiLU(inplace=True),
        )
        self.encoder0 = nn.Sequential(
            *[_ConvNeXtBlock(width) for _ in range(depth0)]
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(width, middle, 4, stride=2, padding=1),
            nn.GroupNorm(_group_count(middle), middle),
            nn.SiLU(inplace=True),
        )
        self.encoder1 = nn.Sequential(
            *[_ConvNeXtBlock(middle) for _ in range(depth1)]
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(middle, bottleneck, 4, stride=2, padding=1),
            nn.GroupNorm(_group_count(bottleneck), bottleneck),
            nn.SiLU(inplace=True),
        )
        self.encoder2 = nn.Sequential(
            *[_ConvNeXtBlock(bottleneck) for _ in range(depth2)]
        )
        self.feature_projection = nn.Sequential(
            nn.Conv2d(self.config.feature_channels, bottleneck, 1),
            nn.GroupNorm(_group_count(bottleneck), bottleneck),
            nn.SiLU(inplace=True),
        )
        self.temporal_context = nn.ModuleList(
            [
                _TemporalContextBlock(bottleneck)
                for _ in range(self.config.temporal_blocks)
            ]
        )
        self.decoder1 = _DecoderStage(bottleneck, middle, middle, depth1)
        self.decoder0 = _DecoderStage(middle, width, width, depth0)

        self.kernel_points = kernel_size * kernel_size
        # alpha(1), G(3), mean displacement(2), confidence(1), C(3), R(3),
        # kernel logits(K), learned local offsets(2K).
        self.operator_channels = 13 + self.kernel_points * 3
        self.geometry_start = self.operator_channels
        # normal(3), metric depth(1), geometry confidence(1)
        output_channels = self.operator_channels + (
            5 if self.config.predict_geometry else 0
        )
        self.head = nn.Conv2d(width, output_channels, 3, padding=1)
        with torch.no_grad():
            self.head.bias[7:10].fill_(self.config.neutral_transmission_bias)
            kernel_start = 13
            self.head.weight[kernel_start : self.operator_channels].zero_()
            self.head.bias[kernel_start : self.operator_channels].zero_()
            self.head.bias[kernel_start : kernel_start + self.kernel_points].fill_(
                -self.config.refractive_kernel_center_bias
            )
            self.head.bias[
                kernel_start + self.kernel_points // 2
            ] = self.config.refractive_kernel_center_bias
            if self.config.predict_geometry:
                self.head.weight[self.geometry_start :].zero_()
                self.head.bias[self.geometry_start :].zero_()
                # A front-facing unit normal is a stable neutral prediction.
                self.head.bias[self.geometry_start + 2] = 1.0

    def _flow_scale(self, raw: Tensor, height: int, width: int) -> Tensor:
        if self.config.flow_parameterization == "resolution_fraction":
            return raw.new_tensor(
                (
                    max(width - 1, 1)
                    * self.config.max_refractive_flow_fraction,
                    max(height - 1, 1)
                    * self.config.max_refractive_flow_fraction,
                )
            ).view(1, 2, 1, 1)
        return raw.new_full(
            (1, 2, 1, 1),
            self.config.max_refractive_flow,
        )

    def _kernel(
        self,
        raw: Tensor,
        base_flow: Tensor,
        support: Tensor,
        height: int,
        width: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        count = self.kernel_points
        logits = raw[:, 13 : 13 + count]
        learned = raw[:, 13 + count : 13 + count * 3].reshape(
            -1,
            count,
            2,
            height,
            width,
        )
        side = self.config.refractive_kernel_size
        axis = torch.linspace(
            -1.0,
            1.0,
            side,
            device=raw.device,
            dtype=raw.dtype,
        )
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        fixed = torch.stack((xx, yy), dim=-1).reshape(1, count, 2, 1, 1)
        radius = raw.new_tensor(
            (
                max(width - 1, 1)
                * self.config.refractive_kernel_radius_fraction,
                max(height - 1, 1)
                * self.config.refractive_kernel_radius_fraction,
            )
        ).view(1, 1, 2, 1, 1)
        local_offset = (fixed + torch.tanh(learned)) * radius
        kernel_flows = base_flow[:, None] + local_offset * support[:, None]

        weights = logits.softmax(dim=1)
        center = torch.zeros_like(weights)
        center[:, count // 2] = 1.0
        weights = weights * support + center * (1.0 - support)
        expected_flow = (weights[:, :, None] * kernel_flows).sum(dim=1)
        return weights, kernel_flows, expected_flow

    def forward(
        self,
        frames: Tensor,
        counterfactual_background: Tensor,
        trimap_logits: Tensor,
        non_memory_features: Tensor,
        background_uncertainty: Tensor,
        mam2_alpha: Tensor | None = None,
    ) -> PhysicsMatterOutput:
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError("frames must have shape [B,T,3,H,W]")
        if counterfactual_background.shape != frames.shape:
            raise ValueError("counterfactual_background must match frames")
        b, t, _, h, w = frames.shape
        if trimap_logits.shape[:3] != (b, t, 3):
            raise ValueError("trimap_logits must have shape [B,T,3,h,w]")
        if mam2_alpha is not None and mam2_alpha.shape[:3] != (b, t, 1):
            raise ValueError("mam2_alpha must have shape [B,T,1,h,w]")
        if non_memory_features.shape[:3] != (
            b,
            t,
            self.config.feature_channels,
        ):
            raise ValueError(
                "non_memory_features channel count does not match "
                "MatterConfig.feature_channels"
            )
        if background_uncertainty.shape != (b, t, 1, h, w):
            raise ValueError(
                "background_uncertainty must have shape [B,T,1,H,W]"
            )

        flat_frames = frames.reshape(b * t, 3, h, w)
        flat_bg = counterfactual_background.reshape(b * t, 3, h, w)
        flat_uncertainty = background_uncertainty.reshape(b * t, 1, h, w)
        flat_trimap = trimap_logits.reshape(
            b * t,
            3,
            *trimap_logits.shape[-2:],
        )
        trimap_probability = F.interpolate(
            flat_trimap,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).softmax(dim=1)
        flat_mam2_alpha = (
            None
            if mam2_alpha is None
            else F.interpolate(
                mam2_alpha.reshape(
                    b * t,
                    1,
                    *mam2_alpha.shape[-2:],
                ),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).clamp(0.0, 1.0)
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
        skip0 = self.encoder0(self.image_stem(image_input))
        skip1 = self.encoder1(self.down1(skip0))
        deep = self.encoder2(self.down2(skip1))

        flat_features = non_memory_features.reshape(
            b * t,
            self.config.feature_channels,
            *non_memory_features.shape[-2:],
        )
        projected_features = self.feature_projection(flat_features)
        projected_features = F.interpolate(
            projected_features,
            size=deep.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        deep = deep + projected_features
        deep_video = deep.reshape(b, t, *deep.shape[1:])
        for temporal_block in self.temporal_context:
            deep_video = temporal_block(deep_video)
        deep = deep_video.reshape(b * t, *deep_video.shape[2:])

        decoded = self.decoder1(deep, skip1)
        decoded = self.decoder0(decoded, skip0)
        raw = self.head(decoded)

        soft_support = 1.0 - trimap_probability[:, 0:1]
        support = (
            (soft_support >= 0.5).to(raw.dtype)
            if not self.training and self.config.hard_support_at_inference
            else soft_support
        )
        if flat_mam2_alpha is None:
            alpha = torch.sigmoid(raw[:, 0:1]) * support
        elif self.config.refine_mam2_alpha:
            alpha_delta = (
                torch.tanh(raw[:, 0:1])
                * self.config.alpha_refinement_scale
                * trimap_probability[:, 1:2]
            )
            alpha = (flat_mam2_alpha + alpha_delta).clamp(0.0, 1.0)
        else:
            alpha = flat_mam2_alpha

        premultiplied_foreground = torch.sigmoid(raw[:, 1:4]) * support
        valid_alpha = alpha >= self.config.straight_foreground_min_alpha
        straight_foreground = torch.where(
            valid_alpha,
            premultiplied_foreground
            / alpha.clamp_min(self.config.straight_foreground_eps),
            torch.zeros_like(premultiplied_foreground),
        ).clamp(0.0, 1.0)

        base_flow = (
            torch.tanh(raw[:, 4:6])
            * self._flow_scale(raw, h, w)
            * support
        )
        kernel_weights, kernel_flows, refractive_flow = self._kernel(
            raw,
            base_flow,
            support,
            h,
            w,
        )
        confidence = torch.sigmoid(raw[:, 6:7]) * support
        learned_color_transmission = torch.sigmoid(raw[:, 7:10])
        if not self.config.use_rgb_transmission:
            learned_color_transmission = learned_color_transmission.mean(
                dim=1,
                keepdim=True,
            ).expand(-1, 3, -1, -1)
        color_transmission = 1.0 - support * (
            1.0 - learned_color_transmission
        )
        transmittance = (1.0 - alpha) * color_transmission
        residual = (
            torch.tanh(raw[:, 10:13])
            * self.config.residual_scale
            * support
        )
        if not self.config.use_residual:
            residual = torch.zeros_like(residual)

        surface_normal = None
        depth = None
        geometry_confidence = None
        if self.config.predict_geometry:
            geometry = raw[:, self.geometry_start : self.geometry_start + 5]
            surface_normal = F.normalize(
                geometry[:, :3], dim=1, eps=1e-6
            ) * support
            depth = (
                F.softplus(geometry[:, 3:4]) + self.config.geometry_min_depth
            ) * support
            geometry_confidence = torch.sigmoid(geometry[:, 4:5]) * support

        def video(value: Tensor) -> Tensor:
            return value.reshape(b, t, *value.shape[1:])

        return PhysicsMatterOutput(
            alpha=video(alpha),
            premultiplied_foreground=video(premultiplied_foreground),
            straight_foreground=video(straight_foreground),
            color_transmission=video(color_transmission),
            transmittance=video(transmittance),
            refractive_flow=video(refractive_flow),
            residual=video(residual),
            confidence=video(confidence),
            refractive_kernel_weights=video(kernel_weights),
            refractive_kernel_flows=kernel_flows.reshape(
                b,
                t,
                self.kernel_points,
                2,
                h,
                w,
            ),
            surface_normal=(
                None if surface_normal is None else video(surface_normal)
            ),
            depth=None if depth is None else video(depth),
            geometry_confidence=(
                None
                if geometry_confidence is None
                else video(geometry_confidence)
            ),
        )
