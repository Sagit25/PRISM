#!/usr/bin/env python3
"""Run released ProPainter on PRISM clips with oracle per-frame masks."""

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
import torch.nn.functional as F
from PIL import Image

from refractive_mam2.color import linear_to_srgb, srgb_to_linear
from refractive_mam2.dataset import RCTransPRISMDataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, default=4)
    parser.add_argument("--clip-length", type=int, default=8)
    parser.add_argument("--max-sequences", type=int, default=16)
    parser.add_argument("--working-side", type=int, default=256)
    parser.add_argument("--preview-count", type=int, default=12)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if args.max_sequences < 0 or args.working_side < 64 or args.working_side % 8:
        parser.error("invalid sequence count or working side")
    if not 0 <= args.frame_index < args.clip_length:
        parser.error("frame index outside clip")
    if not (args.source_root / "inference_propainter.py").is_file():
        raise FileNotFoundError("source-root is not the released ProPainter checkout")
    dataset = RCTransPRISMDataset(args.data_root / "test", clip_length=args.clip_length, strict_contract=True)
    count = min(len(dataset), args.max_sequences or len(dataset))
    if count == 0:
        raise RuntimeError("empty test split")
    output_dir = args.output_root / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    import wandb
    run = wandb.init(
        entity="humangpt", project="PRISM", name=args.run_name,
        group="prism-64-standalone-comparison", job_type="baseline-evaluation",
        config={"model": "official ProPainter", "split": "test", "sequences": count,
                "clip_length": args.clip_length, "frame_index": args.frame_index,
                "working_side": args.working_side, "mask_dilation": 0,
                "prompt": "oracle per-frame masks", "fine_tuning": False},
    )
    for index in range(count):
        sample = dataset[index]
        tensors = sample["tensors"]
        frames = tensors["frames"].float().clamp(0, 1)
        masks = tensors["object_mask"] >= 0.5
        with tempfile.TemporaryDirectory(prefix="prism-propainter-") as directory:
            root = Path(directory)
            frame_dir, mask_dir, result_dir = root / "frames", root / "masks", root / "results"
            frame_dir.mkdir()
            mask_dir.mkdir()
            for frame_index in range(args.clip_length):
                rgb = linear_to_srgb(frames[frame_index]).mul(255).round().byte().permute(1, 2, 0).numpy()
                Image.fromarray(rgb, "RGB").save(frame_dir / f"{frame_index:04d}.png")
                binary = masks[frame_index, 0].byte().mul(255).numpy()
                Image.fromarray(binary, "L").save(mask_dir / f"{frame_index:04d}.png")
            command = [
                sys.executable, str(args.source_root / "inference_propainter.py"),
                "--video", str(frame_dir), "--mask", str(mask_dir), "--output", str(result_dir),
                "--height", str(args.working_side), "--width", str(args.working_side),
                "--mask_dilation", "0", "--save_frames", "--fp16",
            ]
            subprocess.run(command, cwd=args.source_root, check=True, env=os.environ.copy())
            result = result_dir / "frames" / "frames" / f"{args.frame_index:04d}.png"
            if not result.is_file():
                raise FileNotFoundError(f"ProPainter did not save expected frame: {result}")
            generated = np.asarray(Image.open(result).convert("RGB"), dtype=np.float32) / 255.0
            prediction = srgb_to_linear(torch.from_numpy(generated).permute(2, 0, 1))
            observed = frames[args.frame_index]
            mask = masks[args.frame_index]
            if prediction.shape[-2:] != observed.shape[-2:]:
                prediction = F.interpolate(
                    prediction.unsqueeze(0), size=observed.shape[-2:],
                    mode="bicubic", align_corners=False,
                ).squeeze(0).clamp(0, 1)
            if prediction.shape != observed.shape:
                raise ValueError(f"ProPainter output shape {prediction.shape} != {observed.shape}")
            prediction = torch.where(mask.expand_as(observed), prediction, observed)
            np.savez_compressed(
                output_dir / f"{index:04d}.npz", sequence_id=np.array(sample["sequence_id"]),
                background=prediction.numpy().astype(np.float32),
            )
            if index < args.preview_count:
                preview = args.output_root / "previews" / f"{index:04d}.png"
                preview.parent.mkdir(parents=True, exist_ok=True)
                preview_rgb = linear_to_srgb(prediction).mul(255).round().byte().permute(1, 2, 0).numpy()
                Image.fromarray(preview_rgb, "RGB").save(preview)
                wandb.log({"preview": wandb.Image(str(preview), caption=sample["sequence_id"])}, step=index + 1)
        if (index + 1) % 8 == 0 or index + 1 == count:
            print(f"PRISM_PROPAINTER_PROGRESS sequences={index + 1}/{count}", flush=True)
            wandb.log({"sequences": index + 1}, step=index + 1)
    weights = {}
    for filename in ("ProPainter.pth", "recurrent_flow_completion.pth", "raft-things.pth"):
        path = args.source_root / "weights" / filename
        if not path.is_file():
            raise FileNotFoundError(f"released ProPainter weight missing: {path}")
        weights[filename] = _sha256(path)
    summary = {"model": "released ProPainter", "sequences": count,
               "frame_index": args.frame_index, "weights_sha256": weights, "wandb_url": run.url}
    (args.output_root / "provenance.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    wandb.summary.update(summary)
    wandb.finish()
    print("PRISM_PROPAINTER_COMPLETE " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
