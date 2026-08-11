#!/usr/bin/env python3
"""Backfill exact refractive-correspondence arrow overlays."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np


def preview_rgb(linear_rgb: np.ndarray) -> np.ndarray:
    linear = np.maximum(np.asarray(linear_rgb, dtype=np.float32), 0.0)
    mapped = 1.0 - np.exp(-linear)
    srgb = np.where(
        mapped <= 0.0031308,
        12.92 * mapped,
        1.055 * np.power(mapped, 1.0 / 2.4) - 0.055,
    )
    return np.rint(np.clip(srgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def draw_arrows(
    flow: np.ndarray,
    valid: np.ndarray,
    base_rgb: np.ndarray,
    stride: int,
    canvas_scale: int,
    min_magnitude_px: float,
    max_arrows: int,
) -> tuple[np.ndarray, int]:
    flow = np.asarray(flow, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if flow.ndim != 3 or flow.shape[-1] != 2:
        raise ValueError(f"Expected HxWx2 flow, got {flow.shape}")
    if valid.shape != flow.shape[:2]:
        raise ValueError(f"Mask {valid.shape} does not match {flow.shape[:2]}")

    height, width = flow.shape[:2]
    if base_rgb.shape[:2] != (height, width):
        raise ValueError("Base image size does not match flow")

    scale = int(canvas_scale)
    canvas = cv2.resize(
        np.ascontiguousarray(preview_rgb(base_rgb)[..., ::-1]),
        (width * scale, height * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    canvas = np.rint(canvas.astype(np.float32) * 0.48).astype(np.uint8)

    yy, xx = np.nonzero(valid)
    keep = (xx % stride == 0) & (yy % stride == 0)
    yy, xx = yy[keep], xx[keep]
    if yy.size:
        magnitude = np.linalg.norm(flow[yy, xx], axis=1)
        keep = magnitude >= min_magnitude_px
        yy, xx = yy[keep], xx[keep]
    if yy.size > max_arrows:
        indices = np.linspace(0, yy.size - 1, max_arrows, dtype=np.int64)
        yy, xx = yy[indices], xx[indices]

    thickness = max(1, scale // 2)
    radius = max(1, scale // 2)
    for y, x in zip(yy.tolist(), xx.tolist()):
        dx, dy = flow[y, x]
        start = (
            int(round((x + 0.5) * scale)),
            int(round((y + 0.5) * scale)),
        )
        target = (
            int(round((x + dx + 0.5) * scale)),
            int(round((y + dy + 0.5) * scale)),
        )
        cv2.arrowedLine(
            canvas, start, target, (255, 255, 0), thickness,
            cv2.LINE_AA, 0, 0.22,
        )
        cv2.circle(canvas, start, radius, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, target, radius, (0, 0, 255), -1, cv2.LINE_AA)
    return canvas, int(yy.size)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Draw output-pixel -> sampled-background-pixel arrows from Phi.npy"
        )
    )
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--canvas-scale", type=int, default=4)
    parser.add_argument("--min-magnitude-px", type=float, default=0.25)
    parser.add_argument("--max-arrows", type=int, default=1200)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.stride < 1 or args.canvas_scale < 1 or args.max_arrows < 1:
        parser.error("stride, canvas-scale, and max-arrows must be positive")
    if args.min_magnitude_px < 0.0:
        parser.error("min-magnitude-px must be non-negative")

    phi_paths = sorted(args.result_dir.rglob("*_Phi.npy"))
    if not phi_paths:
        raise SystemExit(f"No *_Phi.npy files found under {args.result_dir}")

    written = skipped = arrow_count = 0
    for phi_path in phi_paths:
        stem = phi_path.name[: -len("_Phi.npy")]
        mask_path = phi_path.with_name(stem + "_phi_valid.png")
        image_path = phi_path.with_name(stem + "_I.exr")
        output_path = phi_path.with_name(stem + "_Phi_arrows.png")
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        if not mask_path.exists() or not image_path.exists():
            raise FileNotFoundError(
                f"Need both {mask_path.name} and {image_path.name}"
            )

        flow = np.load(phi_path)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if mask is None or image_bgr is None:
            raise RuntimeError(f"Could not read inputs for {phi_path}")
        if image_bgr.ndim == 2:
            image_bgr = np.repeat(image_bgr[..., None], 3, axis=-1)
        base_rgb = image_bgr[..., :3][..., ::-1]
        canvas, count = draw_arrows(
            flow,
            mask > 0,
            base_rgb,
            args.stride,
            args.canvas_scale,
            args.min_magnitude_px,
            args.max_arrows,
        )
        if not cv2.imwrite(str(output_path), canvas):
            raise IOError(f"Could not write {output_path}")
        written += 1
        arrow_count += count

    print(
        f"written={written} skipped={skipped} arrows_drawn={arrow_count} "
        f"direction='output pixel -> sampled background pixel'"
    )


if __name__ == "__main__":
    main()
