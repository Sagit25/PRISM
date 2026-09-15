from __future__ import annotations

import torch
from torch import Tensor


def linear_to_srgb(value: Tensor) -> Tensor:
    """Convert normalized linear RGB to display/SAM-compatible sRGB."""

    value = value.clamp(0.0, 1.0)
    return torch.where(
        value <= 0.0031308,
        12.92 * value,
        1.055 * value.pow(1.0 / 2.4) - 0.055,
    )


def srgb_to_linear(value: Tensor) -> Tensor:
    """Convert normalized sRGB to the linear RGB used by PRISM physics."""

    value = value.clamp(0.0, 1.0)
    return torch.where(
        value <= 0.04045,
        value / 12.92,
        ((value + 0.055) / 1.055).pow(2.4),
    )
