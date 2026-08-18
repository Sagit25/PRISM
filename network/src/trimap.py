from __future__ import annotations

import torch
from torch import Tensor

BG = 0
UNKNOWN = 1
FG = 2


def build_transparency_trimap(
    object_mask: Tensor,
    alpha: Tensor,
    *,
    foreground_threshold: float = 0.95,
    mask_threshold: float = 0.5,
) -> Tensor:
    """Create transparent-object trimaps.

    Unlike a morphology-only trimap, the full transparent object interior is
    UNKNOWN unless it is genuinely near-opaque.

    Args:
        object_mask: Geometry/support mask with shape [...,1,H,W].
        alpha: Opacity/mixing alpha with the same shape.
    Returns:
        Long tensor with shape [...,H,W] and labels BG=0, UNKNOWN=1, FG=2.
    """

    if object_mask.shape != alpha.shape:
        raise ValueError("object_mask and alpha must have identical shapes")
    if object_mask.shape[-3] != 1:
        raise ValueError("object_mask and alpha must have a singleton channel")

    support = object_mask >= mask_threshold
    definite_fg = support & (alpha >= foreground_threshold)
    result = torch.full_like(alpha, BG, dtype=torch.long)
    result = torch.where(support, torch.full_like(result, UNKNOWN), result)
    result = torch.where(definite_fg, torch.full_like(result, FG), result)
    return result.squeeze(-3)
