"""Shared semantic outputs for the PRISM baseline comparison.

The foreground cutout is the *observed input* inside a binary object mask.
It is not PRISM's straight foreground F or its premultiplied G. Matting alpha
may be used to derive a mask when a model does not expose a semantic mask, but
is never compared with PRISM's physically defined alpha.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


def comparison_mask(values: dict[str, np.ndarray], shape: tuple[int, int]) -> tuple[np.ndarray | None, str | None]:
    if "mask" in values:
        source = "model_mask"
        value = values["mask"]
    elif "alpha" in values:
        source = "alpha_threshold_0.5"
        value = values["alpha"]
    else:
        return None, None
    value = np.asarray(value, dtype=np.float32)
    if value.shape != (1, *shape) or not np.isfinite(value).all():
        raise ValueError(f"invalid {source} shape or values: {value.shape}, expected {(1, *shape)}")
    return (value >= 0.5).astype(np.float32), source


def observed_foreground_cutout(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.float32)
    if frame.ndim != 3 or frame.shape[0] != 3 or mask.shape != (1, *frame.shape[1:]):
        raise ValueError(f"cutout expects RGB frame and matching mask: {frame.shape}, {mask.shape}")
    if not np.isfinite(frame).all() or not np.isfinite(mask).all():
        raise ValueError("cutout contains NaN or Inf")
    return frame * mask


def save_mask_cutout_preview(path, frame_srgb: np.ndarray, gt_mask: np.ndarray, predicted_mask: np.ndarray) -> None:
    """Show exactly the semantic outputs scored in the common comparison."""
    if frame_srgb.ndim != 3 or frame_srgb.shape[2] != 3:
        raise ValueError("preview frame must be HWC RGB")
    height, width = frame_srgb.shape[:2]
    if gt_mask.shape != (1, height, width) or predicted_mask.shape != gt_mask.shape:
        raise ValueError("preview mask shape mismatch")
    rgb = np.asarray(frame_srgb, dtype=np.uint8)
    gt = gt_mask[0] >= 0.5
    pred = predicted_mask[0] >= 0.5
    panels = [rgb, np.repeat(gt[..., None], 3, axis=2) * 255,
              np.repeat(pred[..., None], 3, axis=2) * 255,
              rgb * gt[..., None], rgb * pred[..., None]]
    labels = ("Input", "Mask GT", "Mask pred", "Cutout GT", "Cutout pred")
    scale, label_height = 3, 20
    canvas = Image.new("RGB", (width * scale * len(panels), (height + label_height) * scale), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        x = index * width * scale
        canvas.paste(Image.fromarray(panel.astype(np.uint8), "RGB").resize(
            (width * scale, height * scale), Image.Resampling.NEAREST
        ), (x, label_height * scale))
        draw.text((x + 4, 4), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
