#!/usr/bin/env python3
"""Collect GPU1 ablation test metrics and aligned qualitative image paths."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


VARIANTS = (
    "full", "no_inverse", "no_joint_refinement", "detached_paths",
    "no_residual", "scalar_transmission",
)
PRIMARY = (
    "render_object_psnr", "background_true_hole_psnr",
    "straight_foreground_valid_psnr", "flow_epe", "alpha_boundary_f1",
    "render_object_lpips", "background_true_hole_lpips",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queue_root", type=Path)
    args = parser.parse_args()
    root = args.queue_root / "full"
    rows = []
    images = {}
    for variant in VARIANTS:
        folder = root / variant
        metrics_file = folder / "prism_stage4_metrics.json"
        row = {"variant": variant, "status": "pending"}
        if metrics_file.is_file():
            report = json.loads(metrics_file.read_text())
            row["status"] = "complete" if (folder / ".complete").is_file() else "partial"
            row.update(report["metrics"])
            images[variant] = {
                p.parent.name: str(p)
                for p in sorted((folder / "qualitative").glob(
                    "*/input_reconstruction_bgpred_bggt_alpha.png"
                ))
            }
        elif (folder / "train.log").is_file():
            row["status"] = "running_or_failed"
        rows.append(row)
    all_keys = sorted(set().union(*(row.keys() for row in rows)) - {"variant", "status"})
    with (args.queue_root / "all_metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["variant", "status", *all_keys])
        writer.writeheader()
        writer.writerows(rows)
    with (args.queue_root / "primary_metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["variant", "status", *PRIMARY],
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (args.queue_root / "qualitative_images.json").write_text(
        json.dumps(images, indent=2) + "\n"
    )
    print("PRISM_ABLATION_SUMMARY", [(r["variant"], r["status"]) for r in rows])


if __name__ == "__main__":
    main()
