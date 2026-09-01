from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .background import BackgroundOutput, MaskedTemporalBackground
from .config import PipelineConfig
from .inverse import inverse_refractive_splat
from .matter import PhysicsAwareMatter, PhysicsMatterOutput
from .renderer import recompose
from .types import MAM2Backbone, MAM2BackboneOutput


@dataclass
class RefractiveMAM2Output:
    backbone: MAM2BackboneOutput
    background: BackgroundOutput
    matter: PhysicsMatterOutput
    reconstructed_frames: Tensor
    refracted_background: Tensor


class RefractiveMAM2(nn.Module):
    """Jointly infer one reusable background and a refractive operator."""

    def __init__(
        self,
        backbone: MAM2Backbone,
        config: PipelineConfig | None = None,
        background_model: MaskedTemporalBackground | None = None,
        matter: PhysicsAwareMatter | None = None,
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

    def forward(
        self,
        frames: Tensor,
        prompts: Any | None = None,
        *,
        counterfactual_background_gt: Tensor | None = None,
        use_ground_truth_background: bool = False,
    ) -> RefractiveMAM2Output:
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError("frames must have shape [B,T,3,H,W]")
        backbone_output = self.backbone(frames, prompts)
        return self.forward_from_backbone(
            frames,
            backbone_output,
            counterfactual_background_gt=counterfactual_background_gt,
            use_ground_truth_background=use_ground_truth_background,
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

    def _matter_from_background(
        self,
        frames: Tensor,
        background: Tensor,
        uncertainty: Tensor,
        backbone_output: MAM2BackboneOutput,
    ) -> PhysicsMatterOutput:
        t = frames.shape[1]
        background_video = self._video(background, t)
        uncertainty_video = self._video(uncertainty, t)
        if self.config.detach_background_for_matter:
            background_video = background_video.detach()
            uncertainty_video = uncertainty_video.detach()
        return self.matter(
            frames,
            background_video,
            backbone_output.trimap_logits,
            backbone_output.non_memory_features,
            uncertainty_video,
        )

    def forward_from_backbone(
        self,
        frames: Tensor,
        backbone_output: MAM2BackboneOutput,
        *,
        counterfactual_background_gt: Tensor | None = None,
        use_ground_truth_background: bool = False,
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
        mask_probability = F.interpolate(
            flat_mask, size=(h, w), mode="bilinear", align_corners=False
        ).sigmoid().reshape(b, t, 1, h, w)
        flat_trimap = backbone_output.trimap_logits.reshape(
            b * t, 3, *backbone_output.trimap_logits.shape[-2:]
        )
        trimap_probability = F.interpolate(
            flat_trimap, size=(h, w), mode="bilinear", align_corners=False
        ).softmax(dim=1).reshape(b, t, 3, h, w)
        trimap_support = 1.0 - trimap_probability[:, :, 0:1]
        object_support = torch.maximum(mask_probability, trimap_support)
        semantics_for_background = (
            object_support.detach()
            if self.config.detach_semantics_for_background
            else object_support
        )

        direct_evidence = self.background_model.observe(
            frames, semantics_for_background
        )
        # Fixed-point refinement always uses the deterministic completion.
        # PRISM-Diffusion applies its expensive generative prior once, after
        # the final inverse evidence has been computed.
        estimated_background = self.background_model.fuse(
            direct_evidence,
            completion_variant="base",
        )

        if use_ground_truth_background:
            if counterfactual_background_gt is None:
                raise ValueError("counterfactual_background_gt is required for teacher forcing")
            global_gt = self._global_background_gt(counterfactual_background_gt, frames)
            matter_output = self._matter_from_background(
                frames,
                global_gt,
                torch.zeros_like(estimated_background.uncertainty),
                backbone_output,
            )
            background_for_render = global_gt
        else:
            # Each iteration first predicts an operator, then inverts its
            # transparent interior observations into the same global canvas.
            # No detach occurs, so the render loss jointly updates both sides.
            matter_output: PhysicsMatterOutput | None = None
            refractive_evidence = None
            for _ in range(self.config.joint_refinement_steps):
                matter_output = self._matter_from_background(
                    frames,
                    estimated_background.background,
                    estimated_background.uncertainty,
                    backbone_output,
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
            if self.config.background.completion_variant == "diffusion":
                estimated_background = self.background_model.fuse(
                    direct_evidence,
                    refractive_evidence,
                    completion_variant="diffusion",
                )
            # The loop ends with a background update. Re-evaluate the operator
            # on that final asset so returned assets and the rendered frame do
            # not refer to adjacent fixed-point iterates.
            matter_output = self._matter_from_background(
                frames,
                estimated_background.background,
                estimated_background.uncertainty,
                backbone_output,
            )
            background_for_render = estimated_background.background

        background_video = self._video(background_for_render, t)
        reconstructed, refracted_background = recompose(
            matter_output.alpha,
            matter_output.premultiplied_foreground,
            background_video,
            matter_output.refractive_flow,
            transmittance=matter_output.transmittance,
            residual=matter_output.residual,
        )
        return RefractiveMAM2Output(
            backbone=backbone_output,
            background=estimated_background,
            matter=matter_output,
            reconstructed_frames=reconstructed,
            refracted_background=refracted_background,
        )
