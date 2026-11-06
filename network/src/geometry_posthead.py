"""Checkpoint-compatible visible-surface geometry from final PRISM assets.

This module is deliberately separate from PAM's physical-operator head.  A
trained PRISM checkpoint can therefore be loaded unchanged before the new
geometry weights are initialized or fine-tuned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor, nn

if TYPE_CHECKING:
    from .pipeline import RefractiveMAM2Output


@dataclass
class GeometryPostOutput:
    surface_normal: Tensor  # [B,T,3,H,W], unit world-space normal
    depth: Tensor  # [B,T,1,H,W], positive first-hit camera distance
    geometry_confidence: Tensor  # [B,T,1,H,W]
    object_support: Tensor  # [B,T,1,H,W], predicted, not GT

    def masked(self) -> "GeometryPostOutput":
        """Return display/inference maps confined to predicted object pixels."""

        support = (self.object_support >= 0.5).to(self.depth.dtype)
        return GeometryPostOutput(
            surface_normal=self.surface_normal * support,
            depth=self.depth * support,
            geometry_confidence=self.geometry_confidence * support,
            object_support=self.object_support,
        )


def _block(in_channels: int, out_channels: int, *, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
        nn.GroupNorm(8, out_channels),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.GroupNorm(8, out_channels),
        nn.SiLU(inplace=True),
    )


class PRISMGeometryPostHead(nn.Module):
    """Small video head conditioned on *final jointly refined* PRISM assets.

    It receives I, straight F, premultiplied G, physical alpha, SAM/MAM2
    mask/trimap/alpha, RGB T and tau, flow, R, B, refracted B, render error,
    optical confidence and B uncertainty.  The temporal convolution lets a
    visible surface use evidence from adjacent frames without changing PRISM.
    """

    INPUT_CHANNELS = 37

    def __init__(self, *, width: int = 32, min_depth: float = 1e-3) -> None:
        super().__init__()
        if width < 8 or width % 8:
            raise ValueError("geometry width must be divisible by 8 and at least 8")
        if min_depth <= 0:
            raise ValueError("min_depth must be positive")
        self.min_depth = min_depth
        self.stem = _block(self.INPUT_CHANNELS, width)
        self.down = _block(width, width * 2, stride=2)
        self.bottleneck = _block(width * 2, width * 2)
        self.temporal = nn.Conv3d(
            width * 2, width * 2, (3, 1, 1), padding=(1, 0, 0),
            groups=width * 2, bias=False,
        )
        self.up = _block(width * 3, width)
        self.head = nn.Conv2d(width, 5, 3, padding=1)
        nn.init.zeros_(self.temporal.weight)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[2] = 1.0  # stable initial world-space unit normal
            self.head.bias[3] = 2.0  # initial positive depth ~2.13 scene units

    @staticmethod
    def _resize_video(value: Tensor, size: tuple[int, int]) -> Tensor:
        batch, time, channels = value.shape[:3]
        if value.shape[-2:] == size:
            return value
        return F.interpolate(
            value.reshape(batch * time, channels, *value.shape[-2:]),
            size=size, mode="bilinear", align_corners=False,
        ).reshape(batch, time, channels, *size)

    def forward(
        self, frames: Tensor, prediction: "RefractiveMAM2Output"
    ) -> GeometryPostOutput:
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError("frames must be [B,T,3,H,W]")
        batch, time, _, height, width = frames.shape
        size = (height, width)
        matter = prediction.matter
        backbone = prediction.backbone
        background = prediction.background
        mask = self._resize_video(backbone.mask_logits, size).sigmoid()
        trimap = self._resize_video(backbone.trimap_logits, size).softmax(dim=2)
        upstream_alpha = self._resize_video(backbone.alpha_matte, size).clamp(0, 1)
        if matter.object_support_logits is not None:
            mask = torch.maximum(mask, matter.object_support_logits.sigmoid())
        support = torch.maximum(mask, matter.alpha.clamp(0, 1))
        background_video = background.background[:, None].expand(-1, time, -1, -1, -1)
        uncertainty = background.uncertainty[:, None].expand(-1, time, -1, -1, -1)
        flow_scale = frames.new_tensor((width, height)).view(1, 1, 2, 1, 1)
        conditions = (
            frames,
            torch.asinh(matter.straight_foreground),
            torch.asinh(matter.premultiplied_foreground),
            matter.alpha,
            mask,
            trimap,
            upstream_alpha,
            matter.color_transmission,
            matter.transmittance,
            matter.refractive_flow / flow_scale,
            torch.asinh(matter.residual),
            torch.asinh(background_video),
            torch.asinh(prediction.refracted_background),
            (frames - prediction.reconstructed_frames).clamp(-1, 1),
            matter.confidence,
            uncertainty,
        )
        if any(value.shape[:2] != (batch, time) or value.shape[-2:] != size
               for value in conditions):
            raise ValueError("PRISM assets are not aligned with input video")
        inputs = torch.cat(conditions, dim=2)
        if inputs.shape[2] != self.INPUT_CHANNELS:
            raise RuntimeError(f"expected {self.INPUT_CHANNELS} channels, got {inputs.shape[2]}")
        if not bool(torch.isfinite(inputs).all()):
            raise FloatingPointError("non-finite PRISM geometry conditioning")
        flat = inputs.reshape(batch * time, self.INPUT_CHANNELS, height, width)
        low = self.stem(flat)
        coarse = self.bottleneck(self.down(low))
        coarse_video = coarse.reshape(batch, time, *coarse.shape[1:])
        temporal = self.temporal(coarse_video.permute(0, 2, 1, 3, 4))
        coarse = (coarse_video + temporal.permute(0, 2, 1, 3, 4)).reshape_as(coarse)
        coarse = F.interpolate(coarse, size=size, mode="bilinear", align_corners=False)
        raw = self.head(self.up(torch.cat((coarse, low), dim=1)))
        normal = F.normalize(raw[:, :3], dim=1, eps=1e-6)
        depth = F.softplus(raw[:, 3:4]) + self.min_depth
        confidence = torch.sigmoid(raw[:, 4:5])

        def video(value: Tensor) -> Tensor:
            return value.reshape(batch, time, *value.shape[1:])

        return GeometryPostOutput(
            surface_normal=video(normal),
            depth=video(depth),
            geometry_confidence=video(confidence),
            object_support=support,
        )
