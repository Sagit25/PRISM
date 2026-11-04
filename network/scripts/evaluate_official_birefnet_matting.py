#!/usr/bin/env python3
"""Evaluate released single-image BiRefNet-matting as a mask baseline.

The model receives only test frame 4. Its matte is thresholded for a binary
object-mask comparison; it is not treated as PRISM physical alpha or F.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from huggingface_hub import model_info
from torchvision.transforms.functional import normalize
from transformers import AutoModelForImageSegmentation

from refractive_mam2.color import linear_to_srgb
from refractive_mam2.dataset import RCTransPRISMDataset


MODEL_ID = "ZhengPeng7/BiRefNet-matting"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-sequences", type=int, default=1)
    parser.add_argument("--frame-index", type=int, default=4)
    parser.add_argument("--working-side", type=int, default=1024)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if args.max_sequences < 0 or not 0 <= args.frame_index < 8 or args.working_side < 256:
        parser.error("invalid sequence count, frame index, or working side")

    dataset = RCTransPRISMDataset(args.data_root / "test", clip_length=8, strict_contract=True)
    count = min(len(dataset), args.max_sequences or len(dataset))
    if not count:
        raise RuntimeError("empty test split")
    revision = model_info(MODEL_ID).sha
    model = AutoModelForImageSegmentation.from_pretrained(
        MODEL_ID, revision=revision, trust_remote_code=True
    ).to("cuda").eval()
    print(f"PRISM_BIREFNET_WEIGHTS_LOADED {revision}", flush=True)
    prediction_dir = args.output_root / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    import wandb

    run = wandb.init(
        entity="humangpt", project="PRISM", name=args.run_name,
        group="prism-64-standalone-comparison", job_type="baseline-evaluation",
        config={"model": MODEL_ID, "revision": revision, "split": "test", "sequences": count,
                "actual_model_input": "one RGB image: frame 4", "working_side": args.working_side,
                "prompt": "none", "fine_tuning": False, "scored_output": "binary object mask only"},
    )
    for index in range(count):
        sample = dataset[index]
        frame = sample["tensors"]["frames"][args.frame_index].float().clamp(0, 1)
        srgb = linear_to_srgb(frame).mul(255).round().byte().permute(1, 2, 0).numpy()
        image = Image.fromarray(srgb, "RGB").resize(
            (args.working_side, args.working_side), Image.Resampling.BICUBIC
        )
        x = torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1).copy())[None]
        x = normalize(x.div(255.0), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).cuda()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            matte = model(x)[-1].sigmoid().float()
        if matte.ndim != 4 or matte.shape[1] != 1 or not torch.isfinite(matte).all():
            raise RuntimeError(f"invalid BiRefNet output shape/value: {tuple(matte.shape)}")
        mask = F.interpolate(
            matte, size=frame.shape[-2:], mode="bilinear", align_corners=False
        )[0].ge(0.5).float().cpu().numpy()
        np.savez_compressed(
            prediction_dir / f"{index:04d}.npz",
            sequence_id=np.array(sample["sequence_id"]), mask=mask,
        )
        if index < 12:
            preview = args.output_root / "previews" / f"{index:04d}.png"
            preview.parent.mkdir(exist_ok=True)
            Image.fromarray(np.uint8(mask[0] * 255), "L").save(preview)
            wandb.log({"preview": wandb.Image(str(preview), caption=sample["sequence_id"] + " | RGB image 1 frame | no prompt")}, step=index + 1)
        if (index + 1) % 25 == 0 or index + 1 == count:
            print(f"PRISM_BIREFNET_PROGRESS sequences={index + 1}/{count}", flush=True)
            wandb.log({"sequences": index + 1}, step=index + 1)
    provenance = {"model": MODEL_ID, "revision": revision, "sequences": count,
                  "frame_index": args.frame_index, "actual_input_frames": 1,
                  "prompt": "none", "output": "binary object mask; no independent F or refractive flow",
                  "wandb_url": run.url}
    (args.output_root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    wandb.summary.update(provenance)
    wandb.finish()


if __name__ == "__main__":
    main()
