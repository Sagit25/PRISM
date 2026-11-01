#!/usr/bin/env python3
"""Render matched test frames across baseline prediction archives."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from refractive_mam2.color import linear_to_srgb
from refractive_mam2.dataset import RCTransPRISMDataset


def _panel(value: np.ndarray | None, channels: int, side: int) -> Image.Image:
    if value is None:
        return Image.new("RGB", (side, side), "#dddddd")
    if channels == 3:
        display = linear_to_srgb(torch.from_numpy(value).float().clamp(0, 1)).permute(1, 2, 0).numpy()
    else:
        display = np.repeat(value.transpose(1, 2, 0).clip(0, 1), 3, axis=2)
    return Image.fromarray(np.uint8(np.round(display * 255)), "RGB").resize((side, side), Image.Resampling.NEAREST)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, default=4)
    parser.add_argument("--clip-length", type=int, default=8)
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    parser.add_argument("--prediction", nargs="+", required=True, metavar="NAME=PATH")
    args = parser.parse_args()
    models = []
    for item in args.prediction:
        name, separator, path = item.partition("=")
        if not separator or not name or not path:
            parser.error("prediction must be NAME=PATH")
        models.append((name, Path(path)))
    dataset = RCTransPRISMDataset(args.data_root / "test", clip_length=args.clip_length, strict_contract=True)
    if not 0 <= args.frame_index < args.clip_length:
        parser.error("frame index must be inside clip")
    args.output_root.mkdir(parents=True, exist_ok=True)
    side, label_width, header = 256, 210, 36
    columns = ("Input", "Background", "Alpha", "Mask", "Foreground")
    for index in args.indices:
        if not 0 <= index < len(dataset):
            parser.error(f"index {index} outside test split")
        sample = dataset[index]
        tensors = sample["tensors"]
        frame = args.frame_index
        rows = [("Ground truth", {
            "input": tensors["frames"][frame].numpy(),
            "background": tensors["counterfactual_background"].numpy(),
            "alpha": tensors["alpha"][frame].numpy(),
            "mask": tensors["object_mask"][frame].numpy(),
            "foreground": tensors["straight_foreground"][frame].numpy(),
        })]
        for name, root in models:
            path = root / f"{index:04d}.npz"
            if not path.is_file():
                rows.append((name + " (missing)", {}))
                continue
            with np.load(path, allow_pickle=False) as archive:
                if str(archive["sequence_id"].item()) != sample["sequence_id"]:
                    raise ValueError(f"sequence mismatch in {path}")
                rows.append((name, {key: archive[key].copy() for key in archive.files if key != "sequence_id"}))
        canvas = Image.new("RGB", (label_width + side * len(columns), header + side * len(rows)), "white")
        draw = ImageDraw.Draw(canvas)
        for column, title in enumerate(columns):
            draw.text((label_width + column * side + 8, 10), title, fill="black")
        for row_index, (name, values) in enumerate(rows):
            y = header + row_index * side
            draw.text((8, y + 10), name, fill="black")
            if row_index > 0:
                values["input"] = rows[0][1]["input"]
            for column, key in enumerate(("input", "background", "alpha", "mask", "foreground")):
                panel = _panel(values.get(key), 1 if key in ("alpha", "mask") else 3, side)
                canvas.paste(panel, (label_width + column * side, y))
        out = args.output_root / f"{index:04d}.png"
        canvas.save(out)
        print(f"PRISM_CONTACT_SHEET {sample['sequence_id']} {out}", flush=True)


if __name__ == "__main__":
    main()
