#!/usr/bin/env python3
"""Convert one official TOM-Net short .flo output to PRISM flow coordinates.

TOM-Net's Lua writer stores signed int16 displacements in [vertical,
horizontal] order. PRISM scores float32 [horizontal, vertical] in pixels at
the 64x64 test resolution. This adapter only converts genuine TOM-Net output;
it does not produce a model prediction or substitute a different model.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


MAGIC = 202021.25


def read_official_short_flow(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) < 12:
        raise ValueError("short TOM-Net flow header")
    magic, width, height = struct.unpack_from("<fii", data)
    if magic != MAGIC or width < 1 or height < 1:
        raise ValueError(f"invalid TOM-Net flow header: {magic}, {width}, {height}")
    expected_bytes = 12 + width * height * 2 * 2
    if len(data) != expected_bytes:
        raise ValueError(f"TOM-Net flow payload has {len(data)} bytes, expected {expected_bytes}")
    return np.frombuffer(data, dtype="<i2", offset=12).reshape(height, width, 2).copy()


def to_prism_flow(official_vu: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    if official_vu.ndim != 3 or official_vu.shape[-1] != 2:
        raise ValueError(f"expected [H,W,2] TOM-Net flow, got {official_vu.shape}")
    source_height, source_width = official_vu.shape[:2]
    if min(source_height, source_width, target_height, target_width) < 1:
        raise ValueError("invalid source or target size")
    # Official flow channel 0 is vertical; channel 1 is horizontal.
    xy = torch.from_numpy(official_vu[..., [1, 0]].astype(np.float32)).permute(2, 0, 1)[None]
    if (source_height, source_width) != (target_height, target_width):
        xy = F.interpolate(xy, (target_height, target_width), mode="bilinear", align_corners=False)
    xy[:, 0].mul_(target_width / source_width)
    xy[:, 1].mul_(target_height / source_height)
    return xy[0].numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow-file", type=Path, required=True)
    parser.add_argument("--mask-file", type=Path, required=True)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-height", type=int, default=64)
    parser.add_argument("--target-width", type=int, default=64)
    args = parser.parse_args()
    from PIL import Image

    flow = to_prism_flow(
        read_official_short_flow(args.flow_file), args.target_height, args.target_width
    )
    mask = np.asarray(Image.open(args.mask_file).convert("L").resize(
        (args.target_width, args.target_height), Image.Resampling.NEAREST
    ), dtype=np.float32)[None] / 255.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, sequence_id=np.array(args.sequence_id), flow=flow, mask=mask)


if __name__ == "__main__":
    main()
