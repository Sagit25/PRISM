#!/usr/bin/env python3
"""Run the released ROSE video removal model on PRISM held-out sequences.

ROSE requires 16n+1 frames. The eight real test frames are followed by nine
copies of the last frame; only original frame 4 is scored. This temporal
extension is part of the experimental input contract, not unseen test data.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from refractive_mam2.color import linear_to_srgb, srgb_to_linear
from refractive_mam2.dataset import RCTransPRISMDataset


def _load_pipeline(source_root: Path, base_root: Path, weights_root: Path):
    sys.path.insert(0, str(source_root))
    from diffusers import FlowMatchEulerDiscreteScheduler
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from rose.models import AutoencoderKLWan, CLIPModel, WanT5EncoderModel, WanTransformer3DModel
    from rose.pipeline import WanFunInpaintPipeline

    cfg = OmegaConf.load(source_root / "configs/wan2.1/wan_civitai.yaml")
    tokenizer = AutoTokenizer.from_pretrained(
        str(base_root / cfg["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer"))
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        str(base_root / cfg["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder")),
        additional_kwargs=OmegaConf.to_container(cfg["text_encoder_kwargs"]),
        low_cpu_mem_usage=True,
    )
    clip_encoder = CLIPModel.from_pretrained(
        str(base_root / cfg["image_encoder_kwargs"].get("image_encoder_subpath", "image_encoder"))
    )
    kwargs = OmegaConf.to_container(cfg["scheduler_kwargs"])
    allowed = set(inspect.signature(FlowMatchEulerDiscreteScheduler.__init__).parameters)
    scheduler = FlowMatchEulerDiscreteScheduler(**{k: v for k, v in kwargs.items() if k in allowed})
    vae = AutoencoderKLWan.from_pretrained(
        str(base_root / cfg["vae_kwargs"].get("vae_subpath", "vae")),
        additional_kwargs=OmegaConf.to_container(cfg["vae_kwargs"]),
    )
    # The Civitai config uses `./` relative to the original ROSE checkpoint.
    # Our pinned Hugging Face snapshot is materialized under transformer/.
    transformer = WanTransformer3DModel.from_pretrained(
        str(weights_root / "transformer"),
        transformer_additional_kwargs=OmegaConf.to_container(cfg["transformer_additional_kwargs"]),
    )
    return WanFunInpaintPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
        transformer=transformer, scheduler=scheduler, clip_image_encoder=clip_encoder,
    ).to("cuda", torch.float16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--weights-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-sequences", type=int, default=1)
    parser.add_argument("--frame-index", type=int, default=4)
    parser.add_argument("--working-side", type=int, default=256)
    parser.add_argument("--inference-steps", type=int, default=50)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if args.max_sequences < 0 or not 0 <= args.frame_index < 8:
        parser.error("invalid sequence count or frame index")
    if args.working_side < 128 or args.working_side % 16:
        parser.error("working-side must be a multiple of 16 and at least 128")
    if not (args.source_root / "inference.py").is_file():
        raise FileNotFoundError("official ROSE checkout missing")
    if not (args.weights_root / "transformer" / "diffusion_pytorch_model.safetensors").is_file():
        raise FileNotFoundError("official ROSE transformer weights missing")
    if not (args.base_root / "config.json").is_file():
        raise FileNotFoundError("Wan2.1-Fun-1.3B-InP base weights missing")
    dataset = RCTransPRISMDataset(args.data_root / "test", clip_length=8, strict_contract=True)
    count = min(len(dataset), args.max_sequences or len(dataset))
    if not count:
        raise RuntimeError("empty test split")
    args.output_root.mkdir(parents=True, exist_ok=True)
    predictions = args.output_root / "predictions"
    predictions.mkdir(exist_ok=True)
    import wandb
    run = wandb.init(
        entity="humangpt", project="PRISM", name=args.run_name,
        group="prism-64-standalone-comparison", job_type="baseline-evaluation",
        config={"model": "official ROSE", "split": "test", "sequences": count,
                "frame_index": args.frame_index, "working_side": args.working_side,
                "inference_steps": args.inference_steps,
                "temporal_extension": "repeat final frame 9 times to make 17 frames",
                "prompt": "oracle per-frame masks", "fine_tuning": False},
    )
    pipeline = _load_pipeline(args.source_root, args.base_root, args.weights_root)
    print("PRISM_ROSE_WEIGHTS_LOADED", flush=True)
    for index in range(count):
        sample = dataset[index]
        tensors = sample["tensors"]
        frames = tensors["frames"].float().clamp(0, 1)
        masks = tensors["object_mask"] >= 0.5
        color = linear_to_srgb(frames).mul(255).round().byte().permute(0, 2, 3, 1).numpy()
        mask = masks[:, 0].byte().mul(255).numpy()
        color = np.concatenate([color, np.repeat(color[-1:], 9, axis=0)], axis=0)
        mask = np.concatenate([mask, np.repeat(mask[-1:], 9, axis=0)], axis=0)
        color_256 = np.stack([
            np.asarray(Image.fromarray(frame, "RGB").resize((args.working_side,) * 2, Image.Resampling.BICUBIC))
            for frame in color
        ])
        mask_256 = np.stack([
            np.asarray(Image.fromarray(frame, "L").resize((args.working_side,) * 2, Image.Resampling.NEAREST))
            for frame in mask
        ])
        video = torch.from_numpy(color_256).permute(3, 0, 1, 2).unsqueeze(0).float().div(255)
        hole = torch.from_numpy(mask_256 >= 240).unsqueeze(0).unsqueeze(0).float()
        with torch.inference_mode():
            generated = pipeline(
                prompt="", video=video, mask_video=hole, num_frames=17,
                num_inference_steps=args.inference_steps,
            ).videos
        if generated.ndim != 5 or generated.shape[2] != 17:
            raise ValueError(f"unexpected official ROSE output shape: {generated.shape}")
        prediction = srgb_to_linear(generated[0, :, args.frame_index].float().cpu().clamp(0, 1))
        prediction = F.interpolate(
            prediction.unsqueeze(0), size=frames.shape[-2:], mode="bicubic", align_corners=False,
        ).squeeze(0).clamp(0, 1)
        np.savez_compressed(
            predictions / f"{index:04d}.npz", sequence_id=np.array(sample["sequence_id"]),
            background=prediction.numpy().astype(np.float32),
        )
        if index < 12:
            preview = args.output_root / "previews" / f"{index:04d}.png"
            preview.parent.mkdir(exist_ok=True)
            rgb = linear_to_srgb(prediction).mul(255).round().byte().permute(1, 2, 0).numpy()
            Image.fromarray(rgb, "RGB").save(preview)
            wandb.log({"preview": wandb.Image(str(preview), caption=sample["sequence_id"])}, step=index + 1)
        if (index + 1) % 8 == 0 or index + 1 == count:
            print(f"PRISM_ROSE_PROGRESS sequences={index + 1}/{count}", flush=True)
            wandb.log({"sequences": index + 1}, step=index + 1)
    provenance = {"model": "official ROSE", "sequences": count, "wandb_url": run.url,
                  "source_commit": "6be41c5420bf331c6d491277d5a6feaf9b3a779a",
                  "base_repo": "alibaba-pai/Wan2.1-Fun-1.3B-InP", "weights_repo": "Kunbyte/ROSE",
                  "temporal_extension": "repeat final frame 9 times to make 17 frames"}
    (args.output_root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    wandb.summary.update(provenance)
    wandb.finish()
    print("PRISM_ROSE_COMPLETE " + json.dumps(provenance, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
