#!/usr/bin/env python3
"""Evaluate released MAM2 on PRISM test clips with oracle first-frame masks."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from refractive_mam2.color import linear_to_srgb
from refractive_mam2.dataset import RCTransPRISMDataset
from comparison_semantic import save_mask_cutout_preview


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-sequences", type=int, default=16)
    parser.add_argument("--frame-index", type=int, default=4)
    parser.add_argument("--working-side", type=int, default=256)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if args.max_sequences < 0 or not 0 <= args.frame_index < 8 or args.working_side < 128:
        parser.error("invalid count, frame index, or working size")
    config_path = args.source_root / "configs/inference.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"official MAM2 config missing: {config_path}")
    sys.path.insert(0, str(args.source_root))
    from mam2.config import load_config
    from mam2.inference import MattingAnything2
    from omegaconf import OmegaConf
    config = load_config(config_path)
    config.inference.matter_resolution = [args.working_side, args.working_side]
    config.runtime.async_loading_frames = False
    config.runtime.show_progress = False
    config.output.save_masks = True
    config.output.save_trimaps = True
    OmegaConf.resolve(config)
    dataset = RCTransPRISMDataset(args.data_root / "test", clip_length=8, strict_contract=True)
    count = min(len(dataset), args.max_sequences or len(dataset))
    if not count:
        raise RuntimeError("empty test split")
    prediction_dir = args.output_root / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    import wandb
    run = wandb.init(
        entity="humangpt", project="PRISM", name=args.run_name,
        group="prism-64-standalone-comparison", job_type="baseline-evaluation",
        config={"model": "official Matting Anything 2", "split": "test", "sequences": count,
                "frame_index": args.frame_index, "working_side": args.working_side,
                "prompt": "oracle first-frame object mask", "fine_tuning": False,
                "checkpoint_revision": config.huggingface.revision},
    )
    pipeline = MattingAnything2(config)
    print("PRISM_MAM2_WEIGHTS_LOADED", flush=True)
    for index in range(count):
        sample = dataset[index]
        tensors = sample["tensors"]
        with tempfile.TemporaryDirectory(prefix="prism-mam2-") as directory:
            root = Path(directory)
            frames = root / "frames"
            frames.mkdir()
            for time_index, frame in enumerate(tensors["frames"]):
                rgb = linear_to_srgb(frame.float().clamp(0, 1)).mul(255).round().byte().permute(1, 2, 0).numpy()
                if time_index == args.frame_index:
                    target_rgb = rgb
                Image.fromarray(rgb, "RGB").resize(
                    (args.working_side,) * 2, Image.Resampling.BICUBIC
                ).save(frames / f"{time_index:04d}.png")
            first_mask = tensors["object_mask"][0, 0].ge(0.5).byte().mul(255).numpy()
            mask_file = root / "first_mask.png"
            Image.fromarray(first_mask, "L").resize(
                (args.working_side,) * 2, Image.Resampling.NEAREST
            ).save(mask_file)
            results = root / "official_results"
            pipeline.run(frames, results, [{"object_id": 1, "frame_index": 0, "mask": str(mask_file)}],
                         show_progress=False)
            filename = f"{args.frame_index:04d}.png"
            alpha_path = results / "pha" / filename
            mask_path = results / "mask" / filename
            if not alpha_path.is_file() or not mask_path.is_file():
                raise FileNotFoundError(f"MAM2 did not save alpha and mask for {filename}")
            size = tuple(tensors["alpha"].shape[-2:][::-1])
            alpha = np.asarray(Image.open(alpha_path).convert("L").resize(size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
            mask = np.asarray(Image.open(mask_path).convert("L").resize(size, Image.Resampling.NEAREST), dtype=np.float32) / 255.0
            if alpha.shape != tuple(tensors["alpha"].shape[-2:]):
                raise ValueError(f"MAM2 output size mismatch: {alpha.shape}")
            np.savez_compressed(
                prediction_dir / f"{index:04d}.npz", sequence_id=np.array(sample["sequence_id"]),
                alpha=alpha[None], mask=mask[None],
            )
            if index < 12:
                preview = args.output_root / "previews" / f"{index:04d}.png"
                save_mask_cutout_preview(
                    preview, target_rgb, tensors["object_mask"][args.frame_index].numpy(), mask[None]
                )
                wandb.log({"preview": wandb.Image(str(preview), caption=sample["sequence_id"])}, step=index + 1)
        if (index + 1) % 8 == 0 or index + 1 == count:
            print(f"PRISM_MAM2_PROGRESS sequences={index + 1}/{count}", flush=True)
            wandb.log({"sequences": index + 1}, step=index + 1)
    provenance = {"model": "official Matting Anything 2", "sequences": count,
                  "source_commit": "16c3d08a8ffbed497f0b0575b3d5628bec7b64a2",
                  "checkpoint_revision": str(config.huggingface.revision), "wandb_url": run.url,
                  "working_side": args.working_side}
    (args.output_root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    wandb.summary.update(provenance)
    wandb.finish()
    print("PRISM_MAM2_COMPLETE " + json.dumps(provenance, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
