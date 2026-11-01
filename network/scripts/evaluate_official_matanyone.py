#!/usr/bin/env python3
"""Evaluate released MatAnyone on PRISM clips with an oracle first-frame mask."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from refractive_mam2.color import linear_to_srgb
from refractive_mam2.dataset import RCTransPRISMDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--generation", choices=("1", "2"), default="1")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, default=4)
    parser.add_argument("--clip-length", type=int, default=8)
    parser.add_argument("--max-sequences", type=int, default=16)
    parser.add_argument("--preview-count", type=int, default=12)
    parser.add_argument("--working-side", type=int, default=256,
                        help="Upsample tiny PRISM frames before official memory attention")
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if args.max_sequences < 0 or not 0 <= args.frame_index < args.clip_length or args.working_side < 128:
        parser.error("invalid count or frame index")
    script_name = "inference_matanyone2.py" if args.generation == "2" else "inference_matanyone.py"
    if not (args.source_root / script_name).is_file():
        raise FileNotFoundError(f"source-root is not released MatAnyone {args.generation} checkout")
    dataset = RCTransPRISMDataset(args.data_root / "test", clip_length=args.clip_length, strict_contract=True)
    count = min(len(dataset), args.max_sequences or len(dataset))
    if count == 0:
        raise RuntimeError("empty test split")
    prediction_dir = args.output_root / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    import wandb
    run = wandb.init(
        entity="humangpt", project="PRISM", name=args.run_name,
        group="prism-64-standalone-comparison", job_type="baseline-evaluation",
        config={"model": f"official MatAnyone {args.generation}", "split": "test", "sequences": count,
                "clip_length": args.clip_length, "frame_index": args.frame_index,
                "prompt": "oracle first-frame object mask", "warmup": 10,
                "erosion": 0, "dilation": 0, "working_side": args.working_side,
                "fine_tuning": False},
    )
    for index in range(count):
        sample = dataset[index]
        values = sample["tensors"]
        with tempfile.TemporaryDirectory(prefix="prism-matanyone-") as directory:
            root = Path(directory)
            frames_dir, result_dir = root / "frames", root / "results"
            frames_dir.mkdir()
            for time_index, frame in enumerate(values["frames"]):
                rgb = linear_to_srgb(frame.float().clamp(0, 1)).mul(255).round().byte().permute(1, 2, 0).numpy()
                Image.fromarray(rgb, "RGB").resize(
                    (args.working_side, args.working_side), Image.Resampling.BICUBIC
                ).save(frames_dir / f"{time_index:04d}.png")
            first_mask = values["object_mask"][0, 0].ge(0.5).byte().mul(255).numpy()
            mask_path = root / "first_mask.png"
            Image.fromarray(first_mask, "L").resize(
                (args.working_side, args.working_side), Image.Resampling.NEAREST
            ).save(mask_path)
            command = [
                sys.executable, str(args.source_root / script_name),
                "-i", str(frames_dir), "-m", str(mask_path), "-o", str(result_dir),
                "-w", "10", "-e", "0", "-d", "0", "--save_image",
            ]
            subprocess.run(command, cwd=args.source_root, check=True, env=os.environ.copy())
            result = result_dir / "frames" / "pha" / f"{args.frame_index:04d}.png"
            if not result.is_file():
                raise FileNotFoundError(f"MatAnyone did not save alpha frame: {result}")
            alpha_image = Image.open(result).convert("L").resize(
                tuple(values["alpha"].shape[-2:][::-1]), Image.Resampling.BILINEAR
            )
            alpha = np.asarray(alpha_image, dtype=np.float32) / 255.0
            if alpha.shape != tuple(values["alpha"].shape[-2:]):
                raise ValueError(f"MatAnyone alpha shape {alpha.shape} does not match GT")
            np.savez_compressed(
                prediction_dir / f"{index:04d}.npz", sequence_id=np.array(sample["sequence_id"]),
                alpha=alpha[None].astype(np.float32),
            )
            if index < args.preview_count:
                preview = args.output_root / "previews" / f"{index:04d}.png"
                preview.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.uint8(np.round(alpha * 255)), "L").save(preview)
                wandb.log({"preview": wandb.Image(str(preview), caption=sample["sequence_id"])}, step=index + 1)
        if (index + 1) % 8 == 0 or index + 1 == count:
            print(f"PRISM_MATANYONE_PROGRESS sequences={index + 1}/{count}", flush=True)
            wandb.log({"sequences": index + 1}, step=index + 1)
    weight_name = "matanyone2.pth" if args.generation == "2" else "matanyone.pth"
    weight = args.source_root / "pretrained_models" / weight_name
    if not weight.is_file():
        raise FileNotFoundError(f"MatAnyone official weight missing: {weight}")
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    summary = {"model": f"released MatAnyone {args.generation}", "sequences": count,
               "frame_index": args.frame_index, "weights_sha256": digest, "wandb_url": run.url}
    summary["working_side"] = args.working_side
    (args.output_root / "provenance.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    wandb.summary.update(summary)
    wandb.finish()
    print("PRISM_MATANYONE_COMPLETE " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
