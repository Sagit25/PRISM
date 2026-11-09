"""Frozen MoGe-2 camera-space geometry prior for the PRISM post head.

MoGe sees the original RGB image.  Its first-surface predictions are only a
prior: a separate PRISM-conditioned head must correct transparent surfaces.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MoGe2GeometryPrior(nn.Module):
    """Return camera normals, first-ray distance and validity from MoGe-2."""

    def __init__(self, checkpoint: str, *, num_tokens: int = 1200) -> None:
        super().__init__()
        if not checkpoint:
            raise ValueError("a MoGe-2 normal checkpoint is required")
        if num_tokens < 256:
            raise ValueError("MoGe-2 requires at least 256 image tokens")
        try:
            from moge.model.v2 import MoGeModel
        except ImportError as exc:
            raise ImportError(
                "MoGe-2 geometry prior requires the official microsoft/MoGe package"
            ) from exc
        self.model = MoGeModel.from_pretrained(checkpoint)
        if not hasattr(self.model, "normal_head"):
            raise ValueError("the MoGe-2 checkpoint has no surface-normal head")
        self.model.eval().requires_grad_(False)
        self.num_tokens = num_tokens
        self.checkpoint = str(checkpoint)

    def train(self, mode: bool = True) -> "MoGe2GeometryPrior":
        # The prior is fixed even when the surrounding PRISM adapter trains.
        super().train(False)
        return self

    @staticmethod
    def _to_srgb(linear_rgb: Tensor) -> Tensor:
        linear_rgb = linear_rgb.float().clamp(0.0, 1.0)
        return torch.where(
            linear_rgb <= 0.0031308,
            linear_rgb * 12.92,
            1.055 * linear_rgb.clamp_min(0.0031308).pow(1 / 2.4) - 0.055,
        )

    def forward(self, frames: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError("MoGe-2 expects [B,T,3,H,W] RGB frames")
        batch, time, _, height, width = frames.shape
        image = self._to_srgb(frames.reshape(batch * time, 3, height, width))
        # MoGe.infer runs in inference_mode. Clone its outputs afterwards so
        # the trainable PRISM residual convolutions may save them for backward.
        normals, distances, validities = [], [], []
        # PRISM already holds SAM2, MAM2 and FLUX in GPU memory. Process one
        # frame at a time so MoGe's ViT attention never scales with clip size.
        for frame in image.split(1):
            with torch.inference_mode():
                result = self.model.infer(
                    frame, num_tokens=self.num_tokens, apply_mask=False,
                    use_fp16=frame.device.type == "cuda",
                )
            if "points" not in result or "normal" not in result:
                raise RuntimeError("MoGe-2 normal checkpoint did not return geometry")
            # Clone outside inference_mode so the trainable convolutions can
            # save these priors for their backward pass.
            normals.append(result["normal"].detach().clone().permute(0, 3, 1, 2))
            distances.append(result["points"].detach().clone().norm(dim=-1)[:, None])
            mask = result.get("mask")
            validities.append(
                torch.ones_like(distances[-1]) if mask is None else
                mask.detach().clone().float()[:, None]
            )
        normal = torch.cat(normals)
        distance = torch.cat(distances)
        validity = torch.cat(validities)
        if normal.shape != (batch * time, 3, height, width):
            raise RuntimeError("MoGe-2 normal resolution does not match PRISM")
        if not bool(torch.isfinite(normal).all() and torch.isfinite(distance).all()):
            raise FloatingPointError("non-finite MoGe-2 geometry prior")
        normal = F.normalize(normal, dim=1, eps=1e-6)
        fallback = normal.new_tensor((0.0, 0.0, -1.0)).view(1, 3, 1, 1)
        normal = torch.where(normal.norm(dim=1, keepdim=True) > 0.5, normal, fallback)
        distance = distance.clamp_min(1e-3)
        return normal, distance, validity
