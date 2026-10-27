#!/usr/bin/env python3
"""Evaluate untouched ObjectClear on PRISM's held-out counterfactual backgrounds.

One deterministic frame per sequence is used. The oracle object mask is given
to ObjectClear, making this an optimistic removal baseline. No PRISM flow,
inverse evidence, learned adapter, or temporal input reaches the model.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from refractive_mam2.color import linear_to_srgb, srgb_to_linear
from refractive_mam2.dataset import RCTransPRISMDataset
from refractive_mam2.objectclear_completion import (
    FlowGuidedObjectClearCompleter,
    ObjectClearSettings,
)


def score_background(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> dict[str, float]:
    """Return additive squared-error totals for consistent subset aggregation."""
    if prediction.shape != target.shape or mask.shape != prediction[:1].shape:
        raise ValueError("background or object mask shape mismatch")
    error = (prediction.float().clamp(0, 1) - target.float().clamp(0, 1)).square()
    region = mask.float().expand_as(error)
    return {
        "all_squared_error": float(error.sum()),
        "all_pixels": float(error.numel()),
        "object_squared_error": float((error * region).sum()),
        "object_pixels": float(region.sum()),
    }


def metrics(totals: dict[str, float]) -> dict[str, float]:
    output: dict[str, float] = {}
    for region in ("all", "object"):
        count = totals[f"{region}_pixels"]
        mse = totals[f"{region}_squared_error"] / max(count, 1.0)
        output[f"{region}_mse"] = mse
        output[f"{region}_psnr"] = -10.0 * math.log10(max(mse, 1e-12))
    return output


def save_preview(
    path: Path, observed: torch.Tensor, mask: torch.Tensor,
    predicted: torch.Tensor, target: torch.Tensor,
) -> None:
    panels = (
        ("input", observed), ("GT mask", mask.expand(3, -1, -1)),
        ("ObjectClear B", predicted), ("GT B", target),
    )
    side = observed.shape[-1] * 4
    image = Image.new("RGB", (side * len(panels), side + 24), "white")
    painter = ImageDraw.Draw(image)
    for index, (label, tensor) in enumerate(panels):
        rgb = linear_to_srgb(tensor.float().clamp(0, 1)) if label != "GT mask" else tensor
        array = rgb.mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
        panel = Image.fromarray(array, "RGB").resize((side, side), Image.Resampling.NEAREST)
        image.paste(panel, (side * index, 24))
        painter.text((side * index + 4, 5), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model", default="jixin0101/ObjectClear")
    parser.add_argument("--revision", default="c73af80888dbd519819d0a521b5c8d0f3cda6859")
    parser.add_argument("--clip-length", type=int, default=8)
    parser.add_argument("--max-sequences", type=int, default=64)
    parser.add_argument("--frame-index", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--preview-count", type=int, default=12)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if args.clip_length < 1 or args.max_sequences < 1 or args.steps < 1:
        parser.error("clip length, maximum sequences and steps must be positive")
    if not 0 <= args.frame_index < args.clip_length:
        parser.error("frame index must fall inside the clip")
    if not torch.cuda.is_available():
        raise RuntimeError("ObjectClear comparison requires CUDA")

    dataset = RCTransPRISMDataset(
        args.data_root / "test", clip_length=args.clip_length,
        strict_contract=True, semantic_only=False,
    )
    count = min(len(dataset), args.max_sequences)
    args.output_root.mkdir(parents=True, exist_ok=True)
    settings = ObjectClearSettings(inference_steps=args.steps, use_fused_output=True)
    model = FlowGuidedObjectClearCompleter.from_pretrained(
        args.model, source_root=args.source_root, revision=args.revision,
        settings=settings, device="cuda", dtype="bfloat16", cpu_offload=True,
    ).pipeline
    import wandb

    run = wandb.init(
        entity="humangpt", project="PRISM", name=args.run_name,
        group="prism-64-standalone-comparison", job_type="baseline-evaluation",
        config={
            "model": "released ObjectClear", "revision": args.revision,
            "fine_tuning": False, "flow_guidance": False,
            "prompt": "oracle per-frame object mask", "split": "test",
            "clip_length": args.clip_length, "frame_index": args.frame_index,
            "max_sequences": count, "steps": args.steps,
            "working_side": settings.min_working_side,
        },
    )
    totals = {name: 0.0 for name in (
        "all_squared_error", "all_pixels", "object_squared_error", "object_pixels"
    )}
    records = args.output_root / "per_sequence.jsonl"
    elapsed = 0.0
    with records.open("w", encoding="utf-8") as stream:
        for index in range(count):
            sample = dataset[index]
            tensors = sample["tensors"]
            observed = tensors["frames"][args.frame_index].float()
            mask = (tensors["object_mask"][args.frame_index] >= 0.5)
            target = tensors["counterfactual_background"].float()
            size = (settings.min_working_side,) * 2
            srgb = linear_to_srgb(observed.clamp(0, 1))
            up = F.interpolate(srgb[None], size=size, mode="bicubic", align_corners=False).clamp(0, 1)[0]
            up_mask = F.interpolate(mask.float()[None], size=size, mode="nearest")[0]
            image = Image.fromarray(up.mul(255).round().byte().permute(1, 2, 0).numpy(), "RGB")
            mask_image = Image.fromarray(up_mask[0].mul(255).byte().numpy(), "L")
            generator = torch.Generator(device="cuda").manual_seed(settings.seed)
            start = time.perf_counter()
            with torch.inference_mode():
                result = model(
                    prompt=settings.prompt, image=image, mask_image=mask_image,
                    generator=generator, num_inference_steps=settings.inference_steps,
                    guidance_scale=settings.guidance_scale,
                    height=size[0], width=size[1], return_attn_map=False,
                    output_type="pil",
                )
            torch.cuda.synchronize()
            elapsed += time.perf_counter() - start
            generated = result.images[0]
            if not isinstance(generated, Image.Image):
                raise RuntimeError("ObjectClear did not return a PIL image")
            array = np.asarray(generated.convert("RGB"), dtype=np.float32).copy()
            pixels = torch.from_numpy(array).permute(2, 0, 1).div(255)
            linear = srgb_to_linear(pixels)
            down = F.interpolate(linear[None], size=observed.shape[-2:], mode="area")[0]
            prediction = torch.where(mask.expand_as(observed), down, observed)
            value = score_background(prediction, target, mask)
            for key, item in value.items():
                totals[key] += item
            stream.write(json.dumps({"sequence_id": sample["sequence_id"], **value}) + "\n")
            stream.flush()
            if index < args.preview_count:
                preview = args.output_root / "previews" / f"{index:04d}.png"
                save_preview(preview, observed, mask.float(), prediction, target)
                wandb.log({"preview": wandb.Image(str(preview), caption=sample["sequence_id"])}, step=index + 1)
            if (index + 1) % 8 == 0 or index + 1 == count:
                current = metrics(totals)
                print(f"PRISM_OBJECTCLEAR_BASELINE_PROGRESS sequences={index + 1}/{count} object_psnr={current['object_psnr']:.4f}", flush=True)
                wandb.log({**current, "sequences": index + 1}, step=index + 1)
    summary = {
        "model": "released ObjectClear (no PRISM modules)", "split": "test",
        "mask": "oracle per-frame object mask", "evaluated_sequences": count,
        "available_sequences": len(dataset), "frame_index": args.frame_index,
        "steps": args.steps, "forward_seconds": elapsed,
        "forward_seconds_per_sequence": elapsed / count,
        **metrics(totals), "wandb_url": run.url,
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    wandb.summary.update(summary)
    wandb.finish()
    print("PRISM_OBJECTCLEAR_BASELINE_COMPLETE " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
