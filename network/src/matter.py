from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import FBA_PRE_FUSION_CHANNELS, MatterConfig
from .renderer import warp_background, warp_background_kernel


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
    # Present only when a multi-tap refractive kernel is explicitly enabled.
    refractive_kernel_weights: Tensor | None = None  # [B,T,K,H,W]
    refractive_kernel_flows: Tensor | None = None  # [B,T,K,2,H,W]
    # Optional object-surface geometry. These are deliberately appended to
    # the v17 operator contract so existing callers and checkpoints remain
    # valid when MatterConfig.predict_geometry is disabled.
    surface_normal: Tensor | None = None  # [B,T,3,H,W], signed world-space
    depth: Tensor | None = None  # [B,T,1,H,W], positive camera distance
    geometry_confidence: Tensor | None = None  # [B,T,1,H,W]
    coarse_refractive_flow: Tensor | None = None  # [B,T,2,H,W], before refinement


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


class _FlowResidualRefiner(nn.Module):
    """Correct coarse refraction from its current photometric mismatch."""

    INPUT_CHANNELS = 16  # I, warped B, I-I_hat, u, support, uncertainty, tau

    def __init__(self, width: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(self.INPUT_CHANNELS, width, 3, padding=1),
            nn.GroupNorm(_group_count(width), width),
            nn.SiLU(inplace=True),
        )
        self.context = nn.Sequential(_ConvNeXtBlock(width), _ConvNeXtBlock(width))
        self.head = nn.Conv2d(width, 2, 3, padding=1)
        # The optional module starts at precisely the coarse-flow solution.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, value: Tensor) -> Tensor:
        return self.head(self.context(self.stem(value)))


class _ForegroundResidualRefiner(nn.Module):
    """Recover fine foreground structure from the physical render mismatch."""

    INPUT_CHANNELS = 23  # I, warped B, F, error, tau, alpha, u, R, uncertainty, mask

    def __init__(self, width: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(self.INPUT_CHANNELS, width, 3, padding=1),
            nn.GroupNorm(_group_count(width), width),
            nn.SiLU(inplace=True),
        )
        self.context = nn.Sequential(_ConvNeXtBlock(width), _ConvNeXtBlock(width))
        self.head = nn.Conv2d(width, 3, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, value: Tensor) -> Tensor:
        return self.head(self.context(self.stem(value)))


class _FlowGuidedLatentBridge(nn.Module):
    """Align B features with refractive flow before correcting the operator.

    The FLUX image is only a fixed prior. A small trainable adapter mixes its
    flow-sampled features with the foreground decoder and physical render
    mismatch, avoiding any retraining or latent-space modification of FLUX.
    """

    def __init__(
        self, foreground_channels: int, width: int, *, prior_channels: int = 0
    ) -> None:
        super().__init__()
        self.background_encoder = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1),
            nn.GroupNorm(_group_count(width), width),
            nn.SiLU(inplace=True),
            _ConvNeXtBlock(width),
        )
        # Foreground features, refracted B features, render error, normalized
        # flow, RGB transmission, object support and B uncertainty.
        self.prior_channels = prior_channels
        self.fusion = nn.Sequential(
            nn.Conv2d(
                foreground_channels + width + 10 + prior_channels,
                width,
                3,
                padding=1,
            ),
            nn.GroupNorm(_group_count(width), width),
            nn.SiLU(inplace=True),
            _ConvNeXtBlock(width),
            _ConvNeXtBlock(width),
        )
        self.head = nn.Conv2d(width, 8, 3, padding=1)  # delta F, delta u, delta C
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        foreground_features: Tensor,
        background: Tensor,
        flow: Tensor,
        render_error: Tensor,
        transmission: Tensor,
        support: Tensor,
        uncertainty: Tensor,
        flow_scale: Tensor,
        foreground_prior: Tensor | None = None,
    ) -> Tensor:
        if self.prior_channels:
            if foreground_prior is None or foreground_prior.shape != (
                foreground_features.shape[0],
                self.prior_channels,
                *foreground_features.shape[-2:],
            ):
                raise ValueError("foreground_prior must match bridge batch/spatial shape")
        elif foreground_prior is not None:
            raise ValueError("foreground_prior was supplied to a bridge without prior channels")
        background_features = self.background_encoder(background)
        aligned_background = warp_background(background_features, flow)
        fused = torch.cat(
            (
                foreground_features,
                aligned_background,
                render_error,
                flow / flow_scale,
                transmission,
                support,
                uncertainty,
                *((foreground_prior,) if foreground_prior is not None else ()),
            ),
            dim=1,
        )
        return self.head(self.fusion(fused))


class PhysicsAwareMatter(nn.Module):
    """High-capacity spatio-temporal colored refractive operator predictor.

    The model fuses full-resolution photometric evidence with clean SAM2
    features at a compact bottleneck, mixes every frame in parallel with
    temporal Conv3d blocks, and decodes through two skip-connected scales.

    In addition to the supervised mean flow ``u``, the head predicts a local
    deformable kernel ``{w_k, u_k}``. Rendering becomes

    ``I = G + tau * sum_k w_k B_cf(x + u_k) + R``.

    At the default kernel size of one, the operator predicts the single
    correspondence directly and does not allocate or train kernel taps.
    For explicit multi-tap experiments, the weighted expectation
    ``sum_k w_k u_k`` is returned as ``refractive_flow``.
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
        if self.config.flow_refinement_width < 1:
            raise ValueError("flow_refinement_width must be positive")
        if self.config.flow_refinement_max_fraction <= 0:
            raise ValueError("flow_refinement_max_fraction must be positive")
        if self.config.foreground_refinement_width < 1:
            raise ValueError("foreground_refinement_width must be positive")
        if self.config.foreground_refinement_logit_scale <= 0:
            raise ValueError("foreground_refinement_logit_scale must be positive")
        if self.config.latent_bridge_width < 1:
            raise ValueError("latent_bridge_width must be positive")
        if self.config.latent_bridge_flow_fraction <= 0:
            raise ValueError("latent_bridge_flow_fraction must be positive")
        if self.config.latent_bridge_foreground_logit_scale <= 0:
            raise ValueError("latent_bridge_foreground_logit_scale must be positive")
        if self.config.latent_bridge_transmission_logit_scale <= 0:
            raise ValueError("latent_bridge_transmission_logit_scale must be positive")
        if self.config.alpha_refinement_scale <= 0:
            raise ValueError("alpha_refinement_scale must be positive")
        if not 0.0 <= self.config.boundary_alpha_threshold <= 1.0:
            raise ValueError("boundary_alpha_threshold must be in [0, 1]")
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
        if self.config.flow_guided_latent_bridge and kernel_size != 1:
            raise ValueError("flow_guided_latent_bridge currently requires a 1x1 kernel")
        if self.config.use_fba_foreground_prior and not self.config.flow_guided_latent_bridge:
            raise ValueError("FBA foreground prior requires flow_guided_latent_bridge")
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
        # alpha(1), straight F(3), mean displacement(2), confidence(1), C(3), R(3),
        # kernel logits(K), learned local offsets(2K).
        self.operator_channels = 13 + (
            self.kernel_points * 3 if self.kernel_points > 1 else 0
        )
        self.geometry_start = self.operator_channels
        # normal(3), metric depth(1), geometry confidence(1)
        output_channels = self.operator_channels + (
            5 if self.config.predict_geometry else 0
        )
        self.head = nn.Conv2d(width, output_channels, 3, padding=1)
        self.flow_refiner = (
            _FlowResidualRefiner(self.config.flow_refinement_width)
            if self.config.refine_refractive_flow
            else None
        )
        self.foreground_refiner = (
            _ForegroundResidualRefiner(self.config.foreground_refinement_width)
            if self.config.refine_straight_foreground
            else None
        )
        self.latent_bridge = (
            _FlowGuidedLatentBridge(
                width,
                self.config.latent_bridge_width,
                prior_channels=(
                    FBA_PRE_FUSION_CHANNELS
                    if self.config.use_fba_foreground_prior else 0
                ),
            )
            if self.config.flow_guided_latent_bridge
            else None
        )
        if self.config.use_rgb_transmission and self.config.canonical_rgb_transmission:
            # Old sigmoid-head checkpoints have different output semantics.
            self.ARCHITECTURE = self.ARCHITECTURE + "-canonical-rgb-transmission-v1"
        if self.config.hard_object_support:
            self.ARCHITECTURE = self.ARCHITECTURE + "-hard-object-support-v1"
        # The same head channels now represent F rather than G; loading an
        # older checkpoint silently would change its physical interpretation.
        self.ARCHITECTURE = self.ARCHITECTURE + "-straight-foreground-hard-flow-v1"
        # Alpha can now correct an object interior and preserve an optional
        # antialiased shell. Old PAM checkpoints have different semantics.
        self.ARCHITECTURE = self.ARCHITECTURE + "-alpha-interior-boundary-v2"
        if self.flow_refiner is not None:
            self.ARCHITECTURE = self.ARCHITECTURE + "-flow-residual-v1"
        if self.foreground_refiner is not None:
            self.ARCHITECTURE = self.ARCHITECTURE + "-foreground-residual-v1"
        if self.latent_bridge is not None:
            self.ARCHITECTURE = self.ARCHITECTURE + "-flow-guided-latent-bridge-v1"
        if self.config.use_fba_foreground_prior:
            self.ARCHITECTURE = self.ARCHITECTURE + "-pretrained-fba-features-v2"
        with torch.no_grad():
            if self.config.use_rgb_transmission and self.config.canonical_rgb_transmission:
                self.head.bias[7:10].zero_()
            else:
                self.head.bias[7:10].fill_(self.config.neutral_transmission_bias)
            if self.kernel_points > 1:
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
        object_mask_logits: Tensor | None = None,
        foreground_prior: Tensor | None = None,
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
        if object_mask_logits is not None and (
            object_mask_logits.ndim != 5
            or object_mask_logits.shape[:3] != (b, t, 1)
        ):
            raise ValueError("object_mask_logits must have shape [B,T,1,h,w]")
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
        if self.config.use_fba_foreground_prior:
            if foreground_prior is None or foreground_prior.shape != (
                b, t, FBA_PRE_FUSION_CHANNELS, h, w
            ):
                raise ValueError(
                    f"foreground_prior must have shape [B,T,{FBA_PRE_FUSION_CHANNELS},H,W]"
                )
        elif foreground_prior is not None:
            raise ValueError("foreground_prior supplied when FBA prior is disabled")

        inputs = (
            ("frames", frames),
            ("counterfactual_background", counterfactual_background),
            ("trimap_logits", trimap_logits),
            ("non_memory_features", non_memory_features),
            ("background_uncertainty", background_uncertainty),
        )
        if mam2_alpha is not None:
            inputs += (("mam2_alpha", mam2_alpha),)
        if object_mask_logits is not None:
            inputs += (("object_mask_logits", object_mask_logits),)
        if foreground_prior is not None:
            inputs += (("foreground_prior", foreground_prior),)
        finite_inputs = torch.stack([torch.isfinite(value).all() for _, value in inputs])
        if not bool(finite_inputs.all()):
            invalid_names = [
                name for (name, _), valid in zip(inputs, finite_inputs.tolist())
                if not valid
            ]
            raise FloatingPointError(
                "PAM input contains NaN or Inf: " + ", ".join(invalid_names)
            )

        def trace_finite(name: str, value: Tensor) -> Tensor:
            if os.environ.get("PRISM_NUMERIC_TRACE") == "1" and not bool(
                torch.isfinite(value).all()
            ):
                raise FloatingPointError(
                    f"PAM {name} contains NaN or Inf"
                )
            return value

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
        skip0 = trace_finite("encoder0", self.encoder0(
            trace_finite("image_stem", self.image_stem(image_input))
        ))
        skip1 = trace_finite("encoder1", self.encoder1(
            trace_finite("down1", self.down1(skip0))
        ))
        deep = trace_finite("encoder2", self.encoder2(
            trace_finite("down2", self.down2(skip1))
        ))

        flat_features = non_memory_features.reshape(
            b * t,
            self.config.feature_channels,
            *non_memory_features.shape[-2:],
        )
        projected_features = trace_finite(
            "feature_projection", self.feature_projection(flat_features)
        )
        projected_features = F.interpolate(
            projected_features,
            size=deep.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        deep = trace_finite("feature_fusion", deep + projected_features)
        deep_video = deep.reshape(b, t, *deep.shape[1:])
        for temporal_block in self.temporal_context:
            deep_video = trace_finite(
                "temporal_context", temporal_block(deep_video)
            )
        deep = deep_video.reshape(b * t, *deep_video.shape[2:])

        decoded = trace_finite("decoder1", self.decoder1(deep, skip1))
        decoded = trace_finite("decoder0", self.decoder0(decoded, skip0))
        raw = self.head(decoded)
        if not bool(torch.isfinite(raw).all()):
            raise FloatingPointError("PAM raw head output contains NaN or Inf")

        soft_support = 1.0 - trimap_probability[:, 0:1]
        # Use the predicted object mask for F and flow even if other physics
        # outputs are in the optional soft-trimap-support ablation.
        if object_mask_logits is None:
            object_probability = soft_support
        else:
            flat_object_logits = object_mask_logits.reshape(
                b * t, 1, *object_mask_logits.shape[-2:]
            )
            object_probability = F.interpolate(
                flat_object_logits,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).sigmoid()
        hard_support = (object_probability >= 0.5).to(raw.dtype)
        if (
            self.config.hard_object_support
            and self.config.antialiased_boundary_support
            and flat_mam2_alpha is not None
        ):
            one_pixel_neighborhood = F.max_pool2d(
                hard_support, kernel_size=3, stride=1, padding=1,
            )
            antialiased_edge = (
                one_pixel_neighborhood > 0
            ) & (flat_mam2_alpha >= self.config.boundary_alpha_threshold)
            hard_support = torch.maximum(
                hard_support, antialiased_edge.to(raw.dtype),
            )
        if self.config.hard_object_support:
            # The only extension beyond the predicted center-ray mask is a
            # MAM2-supported one-pixel antialiased shell; no GT enters here.
            support = (
                hard_support + (object_probability - object_probability.detach())
                if self.training else hard_support
            )
        else:
            support = (
                (soft_support >= 0.5).to(raw.dtype)
                if not self.training and self.config.hard_support_at_inference
                else soft_support
            )
        if flat_mam2_alpha is None:
            alpha = torch.sigmoid(raw[:, 0:1])
        elif self.config.refine_mam2_alpha:
            alpha_delta = (
                torch.tanh(raw[:, 0:1])
                * self.config.alpha_refinement_scale
            )
            alpha = (flat_mam2_alpha + alpha_delta).clamp(0.0, 1.0)
        else:
            alpha = flat_mam2_alpha
        if self.config.hard_object_support or flat_mam2_alpha is None:
            alpha = alpha * support

        # Predict the straight foreground color, then form its contribution
        # to the image with the *predicted* alpha.  In particular, G must
        # vanish when alpha does; dividing a separately predicted G by a
        # small alpha made F undefined and broke that physical constraint.
        # F has no meaning outside the predicted object. Use a detached hard
        # gate here: alpha already carries the differentiable semantic gate,
        # so applying its STE a second time would double the mask gradient.
        foreground_support = hard_support
        # GT F=G/alpha is nonnegative but can exceed one for a highlight on
        # a partly transmitting pixel, so a sigmoid would make it unreachable.
        raw_straight_foreground = F.softplus(raw[:, 1:4])
        straight_foreground = raw_straight_foreground * foreground_support
        # In forward evaluation this is exactly alpha*F. In backward training
        # the unmasked F is a surrogate only for the semantic-mask gradient:
        # an initially missed object pixel must still receive a render signal.
        premultiplied_foreground = alpha * raw_straight_foreground
        desired_g = alpha * straight_foreground
        premultiplied_foreground = premultiplied_foreground + (
            desired_g - premultiplied_foreground
        ).detach()

        # Use a binary predicted support even when the optional soft-support
        # ablation is selected, retaining an STE to the segmentation branch.
        flow_support = foreground_support + (
            object_probability - object_probability.detach()
        )

        base_flow = (
            torch.tanh(raw[:, 4:6])
            * self._flow_scale(raw, h, w)
            * flow_support
        )
        if self.kernel_points == 1:
            kernel_weights = None
            kernel_flows = None
            refractive_flow = base_flow
        else:
            kernel_weights, kernel_flows, refractive_flow = self._kernel(
                raw,
                base_flow,
                flow_support,
                h,
                w,
            )
        coarse_flow = refractive_flow
        confidence = torch.sigmoid(raw[:, 6:7]) * support
        if self.config.use_rgb_transmission and self.config.canonical_rgb_transmission:
            # GT T is tau / max_rgb(tau), not three independent attenuation
            # probabilities.  Relative log-ratios exactly enforce max(T)=1
            # and avoid the nearly-zero sigmoid gradient of the old +4 bias.
            color_logits = raw[:, 7:10]
            learned_color_transmission = torch.exp(
                color_logits - color_logits.amax(dim=1, keepdim=True)
            )
        else:
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

        if self.flow_refiner is not None:
            if kernel_weights is None:
                warped_background = warp_background(flat_bg, coarse_flow)
            else:
                warped_background = warp_background_kernel(
                    counterfactual_background,
                    kernel_flows.reshape(b, t, self.kernel_points, 2, h, w),
                    kernel_weights.reshape(b, t, self.kernel_points, h, w),
                ).reshape(b * t, 3, h, w)
            coarse_reconstruction = (
                premultiplied_foreground
                + transmittance * warped_background
                + residual
            )
            flow_scale = self._flow_scale(raw, h, w)
            correction_input = torch.cat(
                (
                    flat_frames,
                    warped_background,
                    flat_frames - coarse_reconstruction,
                    coarse_flow / flow_scale,
                    support,
                    flat_uncertainty,
                    transmittance,
                ),
                dim=1,
            )
            correction_scale = raw.new_tensor(
                (
                    max(w - 1, 1) * self.config.flow_refinement_max_fraction,
                    max(h - 1, 1) * self.config.flow_refinement_max_fraction,
                )
            ).view(1, 2, 1, 1)
            correction_raw = self.flow_refiner(correction_input)
            if not bool(torch.isfinite(correction_raw).all()):
                raise FloatingPointError("PAM flow refinement contains NaN or Inf")
            delta_flow = (
                torch.tanh(correction_raw)
                * correction_scale
                * flow_support
            )
            refractive_flow = coarse_flow + delta_flow
            if kernel_flows is not None:
                kernel_flows = kernel_flows + delta_flow[:, None]

        if self.latent_bridge is not None:
            aligned_rgb = warp_background(flat_bg, refractive_flow)
            current_reconstruction = (
                premultiplied_foreground
                + transmittance * aligned_rgb
                + residual
            )
            bridge_raw = self.latent_bridge(
                decoded,
                flat_bg,
                refractive_flow,
                flat_frames - current_reconstruction,
                transmittance,
                foreground_support,
                flat_uncertainty,
                self._flow_scale(raw, h, w),
                (
                    foreground_prior.reshape(b * t, FBA_PRE_FUSION_CHANNELS, h, w)
                    if foreground_prior is not None else None
                ),
            )
            if not bool(torch.isfinite(bridge_raw).all()):
                raise FloatingPointError("PAM latent bridge contains NaN or Inf")
            bridge_flow_scale = raw.new_tensor(
                (
                    max(w - 1, 1) * self.config.latent_bridge_flow_fraction,
                    max(h - 1, 1) * self.config.latent_bridge_flow_fraction,
                )
            ).view(1, 2, 1, 1)
            refractive_flow = refractive_flow + (
                torch.tanh(bridge_raw[:, 3:5])
                * bridge_flow_scale
                * flow_support
            )
            foreground_logits = (
                raw[:, 1:4]
                + torch.tanh(bridge_raw[:, :3])
                * self.config.latent_bridge_foreground_logit_scale
            )
            raw_straight_foreground = F.softplus(foreground_logits)
            straight_foreground = raw_straight_foreground * foreground_support
            premultiplied_foreground = alpha * raw_straight_foreground
            desired_g = alpha * straight_foreground
            premultiplied_foreground = premultiplied_foreground + (
                desired_g - premultiplied_foreground
            ).detach()
            transmission_logits = (
                raw[:, 7:10]
                + torch.tanh(bridge_raw[:, 5:8])
                * self.config.latent_bridge_transmission_logit_scale
            )
            if self.config.use_rgb_transmission and self.config.canonical_rgb_transmission:
                learned_color_transmission = torch.exp(
                    transmission_logits
                    - transmission_logits.amax(dim=1, keepdim=True)
                )
            else:
                learned_color_transmission = torch.sigmoid(transmission_logits)
            if not self.config.use_rgb_transmission:
                learned_color_transmission = learned_color_transmission.mean(
                    dim=1, keepdim=True
                ).expand(-1, 3, -1, -1)
            color_transmission = 1.0 - support * (
                1.0 - learned_color_transmission
            )
            transmittance = (1.0 - alpha) * color_transmission

        if self.foreground_refiner is not None:
            if kernel_weights is None:
                warped_background = warp_background(flat_bg, refractive_flow)
            else:
                warped_background = warp_background_kernel(
                    counterfactual_background,
                    kernel_flows.reshape(b, t, self.kernel_points, 2, h, w),
                    kernel_weights.reshape(b, t, self.kernel_points, h, w),
                ).reshape(b * t, 3, h, w)
            coarse_reconstruction = (
                premultiplied_foreground
                + transmittance * warped_background
                + residual
            )
            # Conditioning signals guide F without letting its direct GT loss
            # drive compensating changes in flow, T, alpha, R, or frozen FLUX.
            correction_input = torch.cat(
                (
                    flat_frames,
                    warped_background.detach(),
                    raw_straight_foreground,
                    (flat_frames - coarse_reconstruction).detach(),
                    transmittance.detach(),
                    alpha.detach(),
                    (refractive_flow / self._flow_scale(raw, h, w)).detach(),
                    residual.detach(),
                    flat_uncertainty,
                    foreground_support,
                ),
                dim=1,
            )
            correction_raw = self.foreground_refiner(correction_input)
            if not bool(torch.isfinite(correction_raw).all()):
                raise FloatingPointError("PAM foreground refinement contains NaN or Inf")
            correction = (
                torch.tanh(correction_raw)
                * self.config.foreground_refinement_logit_scale
            )
            base_foreground_logits = (
                foreground_logits if self.latent_bridge is not None else raw[:, 1:4]
            )
            raw_straight_foreground = F.softplus(
                base_foreground_logits + correction
            )
            straight_foreground = raw_straight_foreground * foreground_support
            premultiplied_foreground = alpha * raw_straight_foreground
            desired_g = alpha * straight_foreground
            premultiplied_foreground = premultiplied_foreground + (
                desired_g - premultiplied_foreground
            ).detach()

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
            coarse_refractive_flow=(
                video(coarse_flow)
                if self.flow_refiner is not None or self.latent_bridge is not None
                else None
            ),
            residual=video(residual),
            confidence=video(confidence),
            refractive_kernel_weights=(
                None if kernel_weights is None else video(kernel_weights)
            ),
            refractive_kernel_flows=(
                None
                if kernel_flows is None
                else kernel_flows.reshape(b, t, self.kernel_points, 2, h, w)
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
