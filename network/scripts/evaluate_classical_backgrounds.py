#!/usr/bin/env python3
"""Oracle-mask Telea and temporal-visible background controls on held-out test.

These are transparent reference methods, not substitutes for learned models.
They never read the counterfactual background while forming predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from refractive_mam2.color import linear_to_srgb, srgb_to_linear
from refractive_mam2.dataset import RCTransPRISMDataset


def _inpaint_linear(linear: np.ndarray, mask: np.ndarray) -> np.ndarray:
    srgb = linear_to_srgb(torch.from_numpy(linear).permute(2, 0, 1)).permute(1, 2, 0).numpy()
    image = np.uint8(np.round(srgb.clip(0, 1) * 255))
    result = cv2.inpaint(image, np.uint8(mask) * 255, 3, cv2.INPAINT_TELEA)
    return srgb_to_linear(torch.from_numpy(result.astype(np.float32) / 255).permute(2, 0, 1)).permute(1, 2, 0).numpy()


def _preview(path: Path, observed: np.ndarray, mask: np.ndarray, prediction: np.ndarray, target: np.ndarray) -> None:
    panels = []
    for value in (observed, prediction, target):
        srgb = linear_to_srgb(torch.from_numpy(value).permute(2, 0, 1)).permute(1, 2, 0).numpy()
        panels.append(Image.fromarray(np.uint8(np.round(srgb.clip(0, 1) * 255)), "RGB"))
    panels.insert(1, Image.fromarray(np.uint8(mask) * 255, "L").convert("RGB"))
    side = observed.shape[0] * 4
    canvas = Image.new("RGB", (side * 4, side + 24), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(zip(("input", "GT mask", "pred B", "GT B"), panels)):
        canvas.paste(image.resize((side, side), Image.Resampling.NEAREST), (index * side, 24))
        draw.text((index * side + 4, 5), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--clip-length", type=int, default=8)
    parser.add_argument("--frame-index", type=int, default=4)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--preview-count", type=int, default=12)
    args = parser.parse_args()
    if args.max_sequences < 0 or not 0 <= args.frame_index < args.clip_length:
        parser.error("invalid sequence count or frame index")
    dataset = RCTransPRISMDataset(args.data_root / "test", clip_length=args.clip_length, strict_contract=True)
    count = min(len(dataset), args.max_sequences or len(dataset))
    if count == 0:
        raise RuntimeError("empty test split")
    for index in range(count):
        sample = dataset[index]
        tensors = sample["tensors"]
        frames = tensors["frames"].numpy().transpose(0, 2, 3, 1)
        masks = tensors["object_mask"][:, 0].numpy() >= 0.5
        target = tensors["counterfactual_background"].numpy().transpose(1, 2, 0)
        observed, mask = frames[args.frame_index], masks[args.frame_index]
        telea = np.where(mask[..., None], _inpaint_linear(observed, mask), observed)
        visible = np.where(masks[..., None], np.nan, frames)
        with np.errstate(all="ignore"):
            median = np.nanmedian(visible, axis=0)
        never_visible = ~np.isfinite(median).all(axis=2)
        median = np.nan_to_num(median, nan=0.0)
        filled = _inpaint_linear(median, never_visible)
        temporal = np.where(never_visible[..., None], filled, median)
        temporal = np.where(mask[..., None], temporal, observed)
        for name, prediction in (("telea_oracle_mask", telea), ("temporal_visible_oracle_masks", temporal)):
            root = args.output_root / name
            output = root / "predictions"
            output.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output / f"{index:04d}.npz", sequence_id=np.array(sample["sequence_id"]),
                background=prediction.transpose(2, 0, 1).astype(np.float32),
            )
            if index < args.preview_count:
                _preview(root / "previews" / f"{index:04d}.png", observed, mask, prediction, target)
        if (index + 1) % 25 == 0 or index + 1 == count:
            print(f"PRISM_CLASSICAL_PROGRESS sequences={index + 1}/{count}", flush=True)
    (args.output_root / "protocol.json").write_text(json.dumps({
        "sequences": count, "frame_index": args.frame_index,
        "telea_oracle_mask": "Current frame plus oracle current mask; OpenCV Telea fill only inside mask.",
        "temporal_visible_oracle_masks": "All input frames and oracle masks; pixel median of directly visible samples, Telea for never-visible pixels.",
    }, indent=2) + "\n")
    print(f"PRISM_CLASSICAL_COMPLETE sequences={count}", flush=True)


if __name__ == "__main__":
    main()
