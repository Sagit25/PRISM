#!/usr/bin/env python3
"""Evaluate frozen FLUX.1-Fill-dev using only frame 4 and an oracle hard mask."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from refractive_mam2.color import linear_to_srgb
from refractive_mam2.dataset import RCTransPRISMDataset
from refractive_mam2.completion import DiffusionCompletionSettings, FrozenDiffusionBackgroundCompleter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--frame-index", type=int, default=4)
    parser.add_argument("--working-side", type=int, default=256)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if args.max_sequences < 0 or args.steps < 1 or args.working_side < 64 or not 0 <= args.frame_index < 8:
        parser.error("invalid evaluation configuration")

    dataset = RCTransPRISMDataset(args.data_root / "test", clip_length=8, strict_contract=True)
    count = min(len(dataset), args.max_sequences or len(dataset))
    if not count:
        raise RuntimeError("empty test split")
    completer = FrozenDiffusionBackgroundCompleter.from_pretrained(
        "black-forest-labs/FLUX.1-Fill-dev",
        settings=DiffusionCompletionSettings(
            inference_steps=args.steps, guidance_scale=30.0,
            min_working_side=args.working_side, image_cache_entries=0,
        ),
        device="cuda", dtype="bfloat16", cpu_offload=True,
    )
    print("PRISM_FLUX_HARDMASK_WEIGHTS_LOADED", flush=True)
    predictions = args.output_root / "predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    import wandb

    run = wandb.init(
        entity="humangpt", project="PRISM", name=args.run_name,
        group="prism-64-standalone-comparison", job_type="baseline-evaluation",
        config={"model": "black-forest-labs/FLUX.1-Fill-dev", "split": "test",
                "sequences": count, "actual_model_input": "one RGB image: frame 4",
                "working_side": args.working_side, "steps": args.steps,
                "prompt": "oracle frame-4 object mask; no temporal or refractive-flow evidence",
                "fine_tuning": False, "scored_output": "background only"},
    )
    for index in range(count):
        sample = dataset[index]
        values = sample["tensors"]
        observed = values["frames"][args.frame_index].float().clamp(0, 1)
        mask = values["object_mask"][args.frame_index, 0].ge(0.5)
        masked = observed * (~mask)[None]
        with torch.no_grad():
            completed = completer(masked[None], (~mask)[None, None].float(), mask[None, None])[0]
        background = torch.where(mask[None], completed.cpu(), observed).clamp(0, 1).numpy()
        if not np.isfinite(background).all():
            raise RuntimeError("FLUX output has NaN or Inf")
        np.savez_compressed(
            predictions / f"{index:04d}.npz", sequence_id=np.array(sample["sequence_id"]),
            background=background.astype(np.float32),
        )
        if index < 12:
            preview = args.output_root / "previews" / f"{index:04d}.png"
            preview.parent.mkdir(exist_ok=True)
            srgb = linear_to_srgb(torch.from_numpy(background)).clamp(0, 1)
            Image.fromarray(np.uint8(np.round(srgb.permute(1, 2, 0).numpy() * 255)), "RGB").save(preview)
            wandb.log({"preview": wandb.Image(str(preview), caption=sample["sequence_id"] + " | one RGB frame + oracle hard mask")}, step=index + 1)
        if (index + 1) % 25 == 0 or index + 1 == count:
            print(f"PRISM_FLUX_HARDMASK_PROGRESS sequences={index + 1}/{count}", flush=True)
            wandb.log({"sequences": index + 1}, step=index + 1)
    provenance = {"model": "black-forest-labs/FLUX.1-Fill-dev frozen", "sequences": count,
                  "frame_index": args.frame_index, "actual_input_frames": 1,
                  "prompt": "oracle frame-4 hard object mask", "working_side": args.working_side,
                  "steps": args.steps, "output": "background only", "wandb_url": run.url}
    (args.output_root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    wandb.summary.update(provenance)
    wandb.finish()


if __name__ == "__main__":
    main()
