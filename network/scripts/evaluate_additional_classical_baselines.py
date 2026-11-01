#!/usr/bin/env python3
"""Non-learning background and mask controls on the frozen PRISM test split.

All spatial methods receive the frame-4 oracle object mask. Temporal nearest
visible receives all eight oracle masks. GrabCut receives only the oracle
frame-4 bounding box and returns a binary mask/alpha, not a color foreground.
The GT background and GT alpha are never read to form predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.restoration import inpaint_biharmonic

from refractive_mam2.color import linear_to_srgb, srgb_to_linear
from refractive_mam2.dataset import RCTransPRISMDataset

from evaluate_classical_backgrounds import _preview


def _srgb_u8(linear: np.ndarray) -> np.ndarray:
    srgb = linear_to_srgb(torch.from_numpy(np.ascontiguousarray(linear.transpose(2, 0, 1))))
    return np.uint8(np.round(srgb.permute(1, 2, 0).numpy().clip(0, 1) * 255))


def _to_linear(srgb: np.ndarray) -> np.ndarray:
    result = srgb_to_linear(torch.from_numpy(np.ascontiguousarray(srgb.transpose(2, 0, 1))).float())
    return result.permute(1, 2, 0).numpy().clip(0, 1)


def navier_stokes_background(observed: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image = _srgb_u8(observed)
    filled = cv2.inpaint(image, np.uint8(mask) * 255, 3, cv2.INPAINT_NS)
    return np.where(mask[..., None], _to_linear(filled.astype(np.float32) / 255), observed)


def biharmonic_background(observed: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return observed.copy()
    image = _srgb_u8(observed).astype(np.float64) / 255
    filled = inpaint_biharmonic(image, mask, channel_axis=-1).clip(0, 1)
    return np.where(mask[..., None], _to_linear(filled), observed)


def nearest_visible_background(frames: np.ndarray, masks: np.ndarray, frame: int) -> np.ndarray:
    """Copy each hidden pixel from the closest frame where GT marks it visible."""
    result = frames[frame].copy()
    missing = masks[frame].copy()
    for source in sorted(range(len(frames)), key=lambda index: (abs(index - frame), index)):
        choose = missing & ~masks[source]
        result[choose] = frames[source][choose]
        missing[choose] = False
    if missing.any():
        from evaluate_classical_backgrounds import _inpaint_linear
        fallback = _inpaint_linear(frames[frame], masks[frame])
        result[missing] = fallback[missing]
    return result


def grabcut_box_mask(observed: np.ndarray, oracle_mask: np.ndarray) -> np.ndarray:
    """Run standard GrabCut with only the GT object's bounding rectangle."""
    height, width = oracle_mask.shape
    ys, xs = np.where(oracle_mask)
    if not len(xs):
        raise ValueError("empty GT object mask cannot define a GrabCut box")
    x0, y0 = max(1, int(xs.min()) - 1), max(1, int(ys.min()) - 1)
    x1, y1 = min(width - 1, int(xs.max()) + 2), min(height - 1, int(ys.max()) + 2)
    if x1 - x0 < 2 or y1 - y0 < 2:
        raise ValueError("GT object box leaves insufficient GrabCut image support")
    image = np.ascontiguousarray(_srgb_u8(observed))
    state = np.zeros((height, width), dtype=np.uint8)
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(
        image, state, (x0, y0, x1 - x0, y1 - y0),
        background_model, foreground_model, 5, cv2.GC_INIT_WITH_RECT,
    )
    return np.isin(state, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.float32)


def _save(root: Path, index: int, sequence_id: str, **values: np.ndarray) -> None:
    output = root / "predictions"
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / f"{index:04d}.npz", sequence_id=np.array(sequence_id),
        **{key: value.astype(np.float32) for key, value in values.items()},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--preview-count", type=int, default=12)
    args = parser.parse_args()
    if args.max_sequences < 0 or args.preview_count < 0:
        parser.error("sequence and preview limits must be non-negative")
    dataset = RCTransPRISMDataset(args.data_root / "test", clip_length=8, strict_contract=True)
    count = min(len(dataset), args.max_sequences or len(dataset))
    if not count:
        raise RuntimeError("empty test split")
    for index in range(count):
        sample = dataset[index]
        tensors = sample["tensors"]
        frames = tensors["frames"].numpy().transpose(0, 2, 3, 1)
        masks = tensors["object_mask"][:, 0].numpy() >= 0.5
        observed, mask = frames[4], masks[4]
        target = tensors["counterfactual_background"].numpy().transpose(1, 2, 0)
        predictions = {
            "navier_stokes_oracle_mask": navier_stokes_background(observed, mask),
            "biharmonic_oracle_mask": biharmonic_background(observed, mask),
            "nearest_visible_oracle_masks": nearest_visible_background(frames, masks, 4),
        }
        for name, background in predictions.items():
            root = args.output_root / name
            _save(root, index, sample["sequence_id"], background=background.transpose(2, 0, 1))
            if index < args.preview_count:
                _preview(root / "previews" / f"{index:04d}.png", observed, mask, background, target)
        binary = grabcut_box_mask(observed, mask)[None]
        root = args.output_root / "grabcut_oracle_box"
        _save(root, index, sample["sequence_id"], alpha=binary, mask=binary)
        if (index + 1) % 25 == 0 or index + 1 == count:
            print(f"PRISM_CLASSICAL_ADDITIONAL_PROGRESS {index + 1}/{count}", flush=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "protocol.json").write_text(json.dumps({
        "sequences": count, "frame_index": 4,
        "navier_stokes_oracle_mask": "Frame 4 and GT frame-4 object mask; OpenCV INPAINT_NS radius 3.",
        "biharmonic_oracle_mask": "Frame 4 and GT frame-4 object mask; scikit-image biharmonic in sRGB.",
        "nearest_visible_oracle_masks": "Eight observed frames and GT per-frame masks; nearest directly visible pixel, Telea fallback only if never visible; no geometric alignment.",
        "grabcut_oracle_box": "Frame 4 plus GT object bounding box only; binary alpha/mask, five GrabCut iterations.",
    }, indent=2) + "\n")
    print(f"PRISM_CLASSICAL_ADDITIONAL_COMPLETE sequences={count}", flush=True)


if __name__ == "__main__":
    main()
