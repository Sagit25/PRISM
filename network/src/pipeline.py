from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .background import BackgroundOutput, MaskedTemporalBackground
from .config import FBA_PRE_FUSION_CHANNELS, PipelineConfig
from .geometry_posthead import GeometryPostOutput, PRISMGeometryPostHead
from .inverse import inverse_refractive_splat
from .matter import PhysicsAwareMatter, PhysicsMatterOutput
from .renderer import recompose
from .types import MAM2Backbone, MAM2BackboneOutput


@dataclass
class RefinementSnapshot:
    """Aligned, detached assets at one fixed-point evaluation state."""

    name: str
    background: Tensor
    alpha: Tensor
    straight_foreground: Tensor
    transmittance: Tensor
    refractive_flow: Tensor
    residual: Tensor
    reconstruction: Tensor


@dataclass
class RefractiveMAM2Output:
    backbone: MAM2BackboneOutput
    background: BackgroundOutput
    matter: PhysicsMatterOutput
    reconstructed_frames: Tensor
    refracted_background: Tensor
    refinement_trace: tuple[RefinementSnapshot, ...] = ()
    geometry: GeometryPostOutput | None = None


class RefractiveMAM2(nn.Module):
    """Jointly infer one reusable background and a refractive operator."""

    def __init__(
        self,
        backbone: MAM2Backbone,
        config: PipelineConfig | None = None,
        background_model: MaskedTemporalBackground | None = None,
        matter: PhysicsAwareMatter | None = None,
        foreground_prior_model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(backbone, nn.Module):
            raise TypeError("backbone must be an nn.Module implementing MAM2Backbone")
        self.config = config or PipelineConfig()
        if self.config.joint_refinement_steps < 0:
            raise ValueError("joint_refinement_steps must be non-negative")
        self.backbone = backbone
        self.background_model = background_model or MaskedTemporalBackground(
            self.config.background
        )
        self.matter = matter or PhysicsAwareMatter(self.config.matter)
        geometry_prior = None
        if self.config.geometry_prior_model == "moge2":
            if not self.config.geometry_post_head:
                raise ValueError("MoGe-2 prior requires the geometry post head")
            from .moge_geometry_prior import MoGe2GeometryPrior

            geometry_prior = MoGe2GeometryPrior(
                self.config.geometry_prior_checkpoint or "",
                num_tokens=self.config.geometry_prior_tokens,
            )
        elif self.config.geometry_prior_model != "none":
            raise ValueError("unsupported geometry prior")
        self.geometry_head = (
            PRISMGeometryPostHead(
                min_depth=self.config.matter.geometry_min_depth,
                prior=geometry_prior,
            )
            if self.config.geometry_post_head else None
        )
        if self.config.matter.use_fba_foreground_prior != (
            foreground_prior_model is not None
        ):
            raise ValueError(
                "FBA foreground-prior config and foreground_prior_model must agree"
            )
        self.foreground_prior_model = foreground_prior_model

    def forward(
        self,
        frames: Tensor,
        prompts: Any | None = None,
        *,
        counterfactual_background_gt: Tensor | None = None,
        use_ground_truth_background: bool = False,
        skip_background_estimation: bool = False,
        capture_refinement_trace: bool = False,
    ) -> RefractiveMAM2Output:
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError("frames must have shape [B,T,3,H,W]")
        backbone_output = self.backbone(frames, prompts)
        return self.forward_from_backbone(
            frames,
            backbone_output,
            counterfactual_background_gt=counterfactual_background_gt,
            use_ground_truth_background=use_ground_truth_background,
            skip_background_estimation=skip_background_estimation,
            capture_refinement_trace=capture_refinement_trace,
        )

    @classmethod
    def _render_from_parts(
        cls,
        frames: Tensor,
        background: Tensor,
        matter: PhysicsMatterOutput,
        mask_probability: Tensor,
    ) -> tuple[Tensor, Tensor]:
        rendered, refracted = recompose(
            matter.alpha,
            matter.premultiplied_foreground,
            cls._video(background, frames.shape[1]),
            matter.refractive_flow,
            transmittance=matter.transmittance,
            residual=matter.residual,
            refractive_kernel_weights=matter.refractive_kernel_weights,
            refractive_kernel_flows=matter.refractive_kernel_flows,
        )
        if matter.object_support_logits is not None:
            support = matter.object_support_logits >= 0
        else:
            support = mask_probability >= 0.5
        support = support | (matter.alpha > 0)
        return torch.where(support, rendered, frames), refracted

    @classmethod
    def _snapshot(
        cls,
        name: str,
        frames: Tensor,
        background: Tensor,
        matter: PhysicsMatterOutput,
        mask_probability: Tensor,
    ) -> RefinementSnapshot:
        reconstructed, _ = cls._render_from_parts(
            frames, background, matter, mask_probability
        )
        return RefinementSnapshot(
            name=name,
            background=background.detach(),
            alpha=matter.alpha.detach(),
            straight_foreground=matter.straight_foreground.detach(),
            transmittance=matter.transmittance.detach(),
            refractive_flow=matter.refractive_flow.detach(),
            residual=matter.residual.detach(),
            reconstruction=reconstructed.detach(),
        )

    @staticmethod
    def _global_background_gt(value: Tensor, frames: Tensor) -> Tensor:
        """Accept the new global GT and the legacy repeated-video encoding."""

        b, t, _, h, w = frames.shape
        if value.shape == (b, 3, h, w):
            return value
        if value.shape == frames.shape:
            return value[:, 0]
        raise ValueError(
            "counterfactual_background_gt must be [B,3,H,W] or [B,T,3,H,W]"
        )

    @staticmethod
    def _video(value: Tensor, frame_count: int) -> Tensor:
        return value[:, None].expand(-1, frame_count, -1, -1, -1)

    def _direct_observation_exclusion(
        self, mask: Tensor, alpha: Tensor, optical_support: Tensor
    ) -> Tensor:
        """Exclude the localized object without trusting broad trimap/alpha maps."""

        if not self.config.background.evidence_flux_only:
            return optical_support
        b, t, _, h, w = mask.shape
        binary_mask = mask >= self.config.background.object_threshold
        near_mask = F.max_pool2d(
            binary_mask.reshape(b * t, 1, h, w).to(mask.dtype),
            kernel_size=3, stride=1, padding=1,
        ).reshape(b, t, 1, h, w) > 0
        return (
            binary_mask | ((alpha >= 0.5) & near_mask)
        ).to(mask.dtype)

    @staticmethod
    def _oracle_background_output(background: Tensor) -> BackgroundOutput:
        """Represent an exact Stage-2 background without running completion."""

        batch, _, height, width = background.shape
        zeros = background.new_zeros((batch, 1, height, width))
        ones = torch.ones_like(zeros)
        false = torch.zeros_like(zeros, dtype=torch.bool)
        true = torch.ones_like(zeros, dtype=torch.bool)
        return BackgroundOutput(
            background=background,
            uncertainty=zeros,
            coverage=ones,
            observed_background=background,
            direct_coverage=ones,
            inverse_background=torch.zeros_like(background),
            inverse_coverage=zeros,
            evidence_background=background,
            direct_support=true,
            inverse_support=false,
            true_hole=false,
        )

    def _matter_from_background(
        self,
        frames: Tensor,
        background: Tensor,
        uncertainty: Tensor,
        backbone_output: MAM2BackboneOutput,
        foreground_prior: Tensor | None = None,
    ) -> PhysicsMatterOutput:
        t = frames.shape[1]
        background_video = self._video(background, t)
        uncertainty_video = self._video(uncertainty, t)
        if self.config.detach_background_for_matter:
            background_video = background_video.detach()
            uncertainty_video = uncertainty_video.detach()
        # PAM predicts bounded physical quantities. Keep its convolutions and
        # nonlinearities in FP32 even when the backbone uses mixed precision;
        # a rare non-finite PAM head output was observed under BF16 autocast.
        with torch.autocast(device_type=frames.device.type, enabled=False):
            return self.matter(
                frames.float(),
                background_video.float(),
                backbone_output.trimap_logits.float(),
                backbone_output.non_memory_features.float(),
                uncertainty_video.float(),
                mam2_alpha=backbone_output.alpha_matte.float(),
                object_mask_logits=backbone_output.mask_logits.float(),
                foreground_prior=(
                    None if foreground_prior is None else foreground_prior.float()
                ),
            )

    def forward_from_backbone(
        self,
        frames: Tensor,
        backbone_output: MAM2BackboneOutput,
        *,
        counterfactual_background_gt: Tensor | None = None,
        use_ground_truth_background: bool = False,
        skip_background_estimation: bool = False,
        capture_refinement_trace: bool = False,
    ) -> RefractiveMAM2Output:
        """Run the differentiable shared-background/operator fixed-point loop."""

        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError("frames must have shape [B,T,3,H,W]")
        backbone_output.validate()
        b, t, _, h, w = frames.shape
        if backbone_output.mask_logits.shape[:2] != (b, t):
            raise ValueError("backbone output batch/time dimensions must match frames")

        flat_mask = backbone_output.mask_logits.reshape(
            b * t, 1, *backbone_output.mask_logits.shape[-2:]
        )
        mask_probability = (
            F.interpolate(flat_mask, size=(h, w), mode="bilinear", align_corners=False)
            .sigmoid()
            .reshape(b, t, 1, h, w)
        )
        flat_trimap = backbone_output.trimap_logits.reshape(
            b * t, 3, *backbone_output.trimap_logits.shape[-2:]
        )
        trimap_probability = (
            F.interpolate(
                flat_trimap, size=(h, w), mode="bilinear", align_corners=False
            )
            .softmax(dim=1)
            .reshape(b, t, 3, h, w)
        )
        trimap_support = 1.0 - trimap_probability[:, :, 0:1]
        alpha_probability = F.interpolate(
            backbone_output.alpha_matte.reshape(
                b * t, 1, *backbone_output.alpha_matte.shape[-2:]
            ),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).reshape(b, t, 1, h, w)
        object_support = torch.maximum(
            torch.maximum(mask_probability, trimap_support),
            alpha_probability,
        )
        base_object_support = object_support
        # Unknown trimap and broad alpha may be high on background pixels.
        direct_object_support = self._direct_observation_exclusion(
            mask_probability, alpha_probability, object_support
        )
        semantics_for_background = (
            direct_object_support.detach()
            if self.config.detach_semantics_for_background
            else direct_object_support
        )
        foreground_prior = None
        if self.foreground_prior_model is not None:
            # One pretrained-model pass per clip, not once per fixed-point
            # iteration. Use pre-fusion decoder features, never FBA's
            # conventional RGB F/B outputs, in PRISM's physical operator.
            foreground_prior = self.foreground_prior_model(frames, backbone_output)
            if foreground_prior.shape != (b, t, FBA_PRE_FUSION_CHANNELS, h, w):
                raise ValueError(
                    f"foreground prior must return [B,T,{FBA_PRE_FUSION_CHANNELS},H,W]"
                )

        if skip_background_estimation and not use_ground_truth_background:
            raise ValueError(
                "skip_background_estimation requires use_ground_truth_background"
            )

        direct_evidence = None
        refinement_trace: list[RefinementSnapshot] = []
        if skip_background_estimation:
            if counterfactual_background_gt is None:
                raise ValueError(
                    "counterfactual_background_gt is required when background "
                    "estimation is skipped"
                )
            global_gt = self._global_background_gt(counterfactual_background_gt, frames)
            estimated_background = self._oracle_background_output(global_gt)
        else:
            direct_evidence = self.background_model.observe(
                frames, semantics_for_background
            )
            # Fixed-point refinement uses deterministic evidence (or learned
            # completion in the base variant). The expensive diffusion prior
            # is applied only once after final inverse evidence is computed.
            estimated_background = self.background_model.fuse(
                direct_evidence,
                completion_variant="base",
            )

        if use_ground_truth_background:
            if counterfactual_background_gt is None:
                raise ValueError(
                    "counterfactual_background_gt is required for teacher forcing"
                )
            global_gt = self._global_background_gt(counterfactual_background_gt, frames)
            matter_output = self._matter_from_background(
                frames,
                global_gt,
                torch.zeros_like(estimated_background.uncertainty),
                backbone_output,
                foreground_prior,
            )
            background_for_render = global_gt
        else:
            assert direct_evidence is not None
            # Each iteration first predicts an operator, then inverts its
            # transparent interior observations into the same global canvas.
            # Base updates are differentiable. With a frozen FLUX prior, the
            # final inverse-feedback correction keeps a bounded gradient path
            # from render loss back through B to the refractive operator.
            matter_output: PhysicsMatterOutput | None = None
            refractive_evidence = None
            for iteration in range(self.config.joint_refinement_steps):
                matter_output = self._matter_from_background(
                    frames,
                    estimated_background.background,
                    estimated_background.uncertainty,
                    backbone_output,
                    foreground_prior,
                )
                if capture_refinement_trace:
                    refinement_trace.append(self._snapshot(
                        f"step_{iteration:02d}", frames,
                        estimated_background.background, matter_output,
                        mask_probability,
                    ))
                if matter_output.object_support_logits is not None:
                    # The frozen tracker can miss most of a transparent
                    # object. Do not then accept its RGB as certain direct B
                    # evidence or restrict inverse evidence to that miss.
                    object_support = torch.maximum(
                        base_object_support,
                        matter_output.object_support_logits.sigmoid(),
                    )
                    direct_object_support = torch.maximum(
                        direct_object_support,
                        (matter_output.object_support_logits >= 0).to(frames.dtype),
                    )
                    semantic_exclusion = (
                        direct_object_support.detach()
                        if self.config.detach_semantics_for_background
                        else direct_object_support
                    )
                    direct_evidence = self.background_model.observe(
                        frames, semantic_exclusion
                    )
                if self.config.background.use_inverse_evidence:
                    refractive_evidence = inverse_refractive_splat(
                        frames,
                        matter_output,
                        object_support,
                        self.config.background,
                    )
                estimated_background = self.background_model.fuse(
                    direct_evidence,
                    refractive_evidence,
                    completion_variant="base",
                )
            matte_model = getattr(self.backbone, "mam2_matter", None)
            if getattr(matte_model, "flow_feature_guidance", False):
                # Flow becomes a conditioning signal *inside official MEMatte*:
                # B(x+u), the current render mismatch and u modulate its
                # backbone feature before the pretrained decoder. The new
                # alpha then changes PAM and its inverse-splatted evidence.
                provisional = self._matter_from_background(
                    frames,
                    estimated_background.background,
                    estimated_background.uncertainty,
                    backbone_output,
                    foreground_prior,
                )
                provisional_render, _ = recompose(
                    provisional.alpha,
                    provisional.premultiplied_foreground,
                    self._video(estimated_background.background, t),
                    provisional.refractive_flow,
                    transmittance=provisional.transmittance,
                    residual=provisional.residual,
                    refractive_kernel_weights=provisional.refractive_kernel_weights,
                    refractive_kernel_flows=provisional.refractive_kernel_flows,
                )
                refined_alpha = matte_model.forward_with_flow(
                    frames,
                    backbone_output.trimap_logits,
                    estimated_background.background,
                    provisional.refractive_flow,
                    frames - provisional_render,
                    object_support,
                )
                trimap_from_alpha = getattr(matte_model, "trimap_from_alpha", None)
                refined_trimap = (
                    trimap_from_alpha(refined_alpha, backbone_output.mask_logits)
                    if callable(trimap_from_alpha)
                    else backbone_output.trimap_logits
                )
                backbone_output = replace(
                    backbone_output, alpha_matte=refined_alpha,
                    trimap_logits=refined_trimap,
                )
                matter_output = self._matter_from_background(
                    frames,
                    estimated_background.background,
                    estimated_background.uncertainty,
                    backbone_output,
                    foreground_prior,
                )
                if self.config.background.use_inverse_evidence:
                    refractive_evidence = inverse_refractive_splat(
                        frames, matter_output, object_support, self.config.background
                    )
                if self.config.background.completion_variant == "base":
                    estimated_background = self.background_model.fuse(
                        direct_evidence, refractive_evidence, completion_variant="base"
                    )
            if self.config.background.completion_variant == "diffusion":
                estimated_background = self.background_model.fuse(
                    direct_evidence,
                    refractive_evidence,
                    completion_variant="diffusion",
                    object_reference=frames[:, 0],
                )
            # The loop ends with a background update. Re-evaluate the operator
            # on that final asset so returned assets and the rendered frame do
            # not refer to adjacent fixed-point iterates.
            matter_output = self._matter_from_background(
                frames,
                estimated_background.background,
                estimated_background.uncertainty,
                backbone_output,
                foreground_prior,
            )
            background_for_render = estimated_background.background

        reconstructed, refracted_background = self._render_from_parts(
            frames, background_for_render, matter_output, mask_probability
        )
        # A visible pixel outside the predicted object is a direct
        # observation, not a completion target. Preserve the input exactly;
        # otherwise a global background estimate can rewrite an unrelated
        # part of the frame and obscure where the optical model is wrong.
        if capture_refinement_trace:
            final_name = (
                "final_flux"
                if not use_ground_truth_background
                and self.config.background.completion_variant == "diffusion"
                else "final"
            )
            refinement_trace.append(self._snapshot(
                final_name, frames, background_for_render, matter_output,
                mask_probability,
            ))
        result = RefractiveMAM2Output(
            backbone=backbone_output,
            background=estimated_background,
            matter=matter_output,
            reconstructed_frames=reconstructed,
            refracted_background=refracted_background,
            refinement_trace=tuple(refinement_trace),
        )
        if self.geometry_head is not None:
            result.geometry = self.geometry_head(frames, result)
        return result
