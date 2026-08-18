#!/usr/bin/env python3
"""Backfill reversible 16-bit flow PNGs from canonical u.npy files."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def encode_flow_png(flow: np.ndarray, valid: np.ndarray) -> np.ndarray:
    flow = np.asarray(flow, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if flow.ndim != 3 or flow.shape[-1] != 2:
        raise ValueError(f"Expected HxWx2 flow, got {flow.shape}")
    if valid.shape != flow.shape[:2]:
        raise ValueError(
            f"Mask {valid.shape} does not match flow {flow.shape[:2]}"
        )

    height, width = flow.shape[:2]
    x_range = float(max(width - 1, 1))
    y_range = float(max(height - 1, 1))
    encoded = np.empty((height, width, 3), dtype=np.uint16)
    encoded[..., 0] = np.rint(
        np.clip(0.5 * (flow[..., 0] / x_range + 1.0), 0.0, 1.0)
        * 65535.0
    ).astype(np.uint16)
    encoded[..., 1] = np.rint(
        np.clip(0.5 * (flow[..., 1] / y_range + 1.0), 0.0, 1.0)
        * 65535.0
    ).astype(np.uint16)
    encoded[..., 2] = np.where(valid, 65535, 0).astype(np.uint16)
    encoded[~valid] = 0
    return encoded


def decode_flow_png(encoded_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if encoded_rgb.dtype != np.uint16 or encoded_rgb.ndim != 3:
        raise ValueError("Expected an RGB uint16 PNG")
    height, width = encoded_rgb.shape[:2]
    valid = encoded_rgb[..., 2] == 65535
    flow = np.empty((height, width, 2), dtype=np.float32)
    flow[..., 0] = (
        (encoded_rgb[..., 0].astype(np.float32) / 65535.0) * 2.0 - 1.0
    ) * float(max(width - 1, 1))
    flow[..., 1] = (
        (encoded_rgb[..., 1].astype(np.float32) / 65535.0) * 2.0 - 1.0
    ) * float(max(height - 1, 1))
    flow[~valid] = 0.0
    return flow, valid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "result_dir",
        type=Path,
        help="Dataset result directory to scan recursively",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    flow_paths = sorted(args.result_dir.rglob("*_u.npy"))
    if not flow_paths:
        raise SystemExit(f"No *_u.npy files found under {args.result_dir}")

    written = 0
    skipped = 0
    worst_error = 0.0
    for flow_path in flow_paths:
        stem = flow_path.name[: -len("_u.npy")]
        mask_path = flow_path.with_name(stem + "_phi_valid.png")
        output_path = flow_path.with_name(stem + "_u_uv16.png")
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing validity mask: {mask_path}")

        flow = np.load(flow_path)
        valid_raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if valid_raw is None:
            raise RuntimeError(f"Could not read: {mask_path}")
        valid = valid_raw > 0
        encoded_rgb = encode_flow_png(flow, valid)
        encoded_bgr = np.ascontiguousarray(encoded_rgb[..., ::-1])
        if not cv2.imwrite(str(output_path), encoded_bgr):
            raise IOError(f"Could not write: {output_path}")

        check_bgr = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)
        if check_bgr is None or check_bgr.dtype != np.uint16:
            raise RuntimeError(f"16-bit PNG verification failed: {output_path}")
        decoded, decoded_valid = decode_flow_png(check_bgr[..., ::-1])
        if not np.array_equal(decoded_valid, valid):
            raise RuntimeError(f"Validity round-trip failed: {output_path}")
        if valid.any():
            error = float(np.max(np.abs(decoded[valid] - flow[valid])))
            worst_error = max(worst_error, error)
        written += 1

    print(
        f"written={written} skipped={skipped} "
        f"worst_valid_vector_error_px={worst_error:.8f}"
    )


if __name__ == "__main__":
    main()
