#!/usr/bin/env python3
"""Backfill explicit I-pixel -> clean-B-pixel correspondence diagnostics."""

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


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise RuntimeError(f"Could not read {path}")
    if bgr.ndim == 2:
        bgr = np.repeat(bgr[..., None], 3, axis=-1)
    return np.asarray(bgr[..., :3][..., ::-1], dtype=np.float32)


def draw_pairs(
    flow: np.ndarray,
    valid: np.ndarray,
    rendered_rgb: np.ndarray,
    background_rgb: np.ndarray,
    stride: int,
    canvas_scale: int,
    min_magnitude_px: float,
    max_arrows: int,
) -> tuple[np.ndarray, int]:
    flow = np.asarray(flow, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if flow.ndim != 3 or flow.shape[-1] != 2:
        raise ValueError(f"Expected HxWx2 flow, got {flow.shape}")
    height, width = flow.shape[:2]
    if valid.shape != (height, width):
        raise ValueError("Mask does not match flow")

    left = preview_rgb(rendered_rgb)
    right = preview_rgb(background_rgb)
    if left.shape[:2] != (height, width) or right.shape[:2] != (
        height,
        width,
    ):
        raise ValueError("I/B sizes do not match flow")

    scale = int(canvas_scale)
    gap = max(8, 6 * scale)
    panel_width = width * scale
    canvas = np.zeros(
        (height * scale, panel_width * 2 + gap, 3), dtype=np.uint8
    )
    canvas[:, :panel_width] = cv2.resize(
        np.ascontiguousarray(left[..., ::-1]),
        (panel_width, height * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    canvas[:, panel_width + gap :] = cv2.resize(
        np.ascontiguousarray(right[..., ::-1]),
        (panel_width, height * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    canvas = np.rint(canvas.astype(np.float32) * 0.55).astype(np.uint8)
    label_scale = max(0.35, scale / 10.0)
    label_y = max(14, 4 * scale)
    cv2.putText(
        canvas,
        "I: rendered pixel (x,y)",
        (5, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        label_scale,
        (255, 255, 255),
        max(1, scale // 2),
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "B: clean source pixel (u,v)",
        (panel_width + gap + 5, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        label_scale,
        (255, 255, 255),
        max(1, scale // 2),
        cv2.LINE_AA,
    )

    yy, xx = np.nonzero(valid)
    keep = (xx % stride == 0) & (yy % stride == 0)
    yy, xx = yy[keep], xx[keep]
    if yy.size:
        keep = np.linalg.norm(flow[yy, xx], axis=1) >= min_magnitude_px
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
            panel_width + gap + int(round((x + dx + 0.5) * scale)),
            int(round((y + dy + 0.5) * scale)),
        )
        cv2.arrowedLine(
            canvas, start, target, (255, 255, 0), thickness,
            cv2.LINE_AA, 0, 0.012,
        )
        cv2.circle(canvas, start, radius, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, target, radius, (0, 0, 255), -1, cv2.LINE_AA)
    return canvas, int(yy.size)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw I-pixel -> clean-B-pixel correspondence pairs"
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
        sequence_stem = stem.rsplit("_frame", 1)[0]
        mask_path = phi_path.with_name(stem + "_phi_valid.png")
        image_path = phi_path.with_name(stem + "_I.exr")
        background_path = phi_path.with_name(sequence_stem + "_background.exr")
        output_path = phi_path.with_name(stem + "_Phi_pairs.png")
        source_path = phi_path.with_name(stem + "_Phi_src.npy")
        if output_path.exists() and source_path.exists() and not args.overwrite:
            skipped += 1
            continue
        for required in (mask_path, image_path, background_path):
            if not required.exists():
                raise FileNotFoundError(required)

        flow = np.load(phi_path)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"Could not read {mask_path}")
        valid = mask > 0
        rendered = read_rgb(image_path)
        background = read_rgb(background_path)
        canvas, count = draw_pairs(
            flow,
            valid,
            rendered,
            background,
            args.stride,
            args.canvas_scale,
            args.min_magnitude_px,
            args.max_arrows,
        )
        if not cv2.imwrite(str(output_path), canvas):
            raise IOError(f"Could not write {output_path}")

        height, width = flow.shape[:2]
        yy, xx = np.meshgrid(
            np.arange(height, dtype=np.float32),
            np.arange(width, dtype=np.float32),
            indexing="ij",
        )
        source = np.stack([xx, yy], axis=-1) + flow
        source[~valid] = 0.0
        np.save(source_path, source.astype(np.float32))
        written += 1
        arrow_count += count

    print(
        f"written={written} skipped={skipped} arrows_drawn={arrow_count} "
        "direction='left I(x,y) -> right clean B(u,v)'"
    )


if __name__ == "__main__":
    main()
