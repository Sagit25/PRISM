#!/usr/bin/env python3
"""Evaluate the published LaMa TorchScript port as a single-image removal baseline.

This is the third-party simple-lama-inpainting port, not the original author's
inference script. It receives one RGB image and the oracle mask at frame 4.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from refractive_mam2.color import linear_to_srgb, srgb_to_linear
from refractive_mam2.dataset import RCTransPRISMDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-sequences", type=int, default=1)
    parser.add_argument("--frame-index", type=int, default=4)
    parser.add_argument("--working-side", type=int, default=256)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if args.max_sequences < 0 or not 0 <= args.frame_index < 8 or args.working_side < 64:
        parser.error("invalid sequence count, frame index, or working side")
    from simple_lama_inpainting import SimpleLama

    dataset = RCTransPRISMDataset(args.data_root / "test", clip_length=8, strict_contract=True)
    count = min(len(dataset), args.max_sequences or len(dataset))
    if not count:
        raise RuntimeError("empty test split")
    model = SimpleLama()
    print("PRISM_LAMA_PORT_WEIGHTS_LOADED", flush=True)
    predictions = args.output_root / "predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    import wandb

    run = wandb.init(
        entity="humangpt", project="PRISM", name=args.run_name,
        group="prism-64-standalone-comparison", job_type="baseline-evaluation",
        config={"model": "simple-lama-inpainting 0.1.2 TorchScript port of LaMa",
                "split": "test", "sequences": count, "actual_model_input": "one RGB image: frame 4",
                "working_side": args.working_side, "prompt": "oracle frame-4 object mask",
                "fine_tuning": False, "scored_output": "background only"},
    )
    for index in range(count):
        sample = dataset[index]
        values = sample["tensors"]
        observed = values["frames"][args.frame_index].float().clamp(0, 1)
        mask = values["object_mask"][args.frame_index, 0].ge(0.5)
        srgb = linear_to_srgb(observed).mul(255).round().byte().permute(1, 2, 0).numpy()
        image = Image.fromarray(srgb, "RGB").resize(
            (args.working_side, args.working_side), Image.Resampling.BICUBIC
        )
        hole = Image.fromarray(mask.byte().mul(255).numpy(), "L").resize(
            (args.working_side, args.working_side), Image.Resampling.NEAREST
        )
        filled = model(image, hole).convert("RGB").resize(
            (observed.shape[-1], observed.shape[-2]), Image.Resampling.BICUBIC
        )
        filled_np = np.asarray(filled, dtype=np.float32).transpose(2, 0, 1).copy() / 255.0
        linear_fill = srgb_to_linear(torch.from_numpy(filled_np)).clamp(0, 1)
        background = torch.where(mask[None], linear_fill, observed).numpy()
        if not np.isfinite(background).all():
            raise RuntimeError("LaMa output has NaN or Inf")
        np.savez_compressed(
            predictions / f"{index:04d}.npz", sequence_id=np.array(sample["sequence_id"]),
            background=background.astype(np.float32),
        )
        if index < 12:
            preview = args.output_root / "previews" / f"{index:04d}.png"
            preview.parent.mkdir(exist_ok=True)
            Image.fromarray(np.uint8(np.round(linear_to_srgb(torch.from_numpy(background)).clamp(0, 1).permute(1, 2, 0).numpy() * 255)), "RGB").save(preview)
            wandb.log({"preview": wandb.Image(str(preview), caption=sample["sequence_id"] + " | RGB image 1 frame + oracle mask")}, step=index + 1)
        if (index + 1) % 25 == 0 or index + 1 == count:
            print(f"PRISM_LAMA_PORT_PROGRESS sequences={index + 1}/{count}", flush=True)
            wandb.log({"sequences": index + 1}, step=index + 1)
    provenance = {"model": "simple-lama-inpainting 0.1.2 LaMa TorchScript port",
                  "sequences": count, "frame_index": args.frame_index,
                  "actual_input_frames": 1, "prompt": "oracle frame-4 object mask",
                  "output": "background only", "wandb_url": run.url}
    (args.output_root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    wandb.summary.update(provenance)
    wandb.finish()


if __name__ == "__main__":
    main()
