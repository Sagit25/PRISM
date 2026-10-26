#!/usr/bin/env python3
"""Evaluate untouched released SAM2Matting weights on PRISM video clips.

This is a semantic matting baseline, not a PRISM reconstruction run: no flow,
background, optical adapter, or fine-tuning is applied.  The only prompt is
the ground-truth object mask on the first frame of each clip.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from refractive_mam2.color import linear_to_srgb
from refractive_mam2.dataset import RCTransPRISMDataset
from refractive_mam2.official_sam2matting import OfficialSAM2MattingClipBackbone
from refractive_mam2.training import normalize_sam2_training_frames


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score_clip(
    alpha: torch.Tensor,
    alpha_gt: torch.Tensor,
    mask_logits: torch.Tensor,
    mask_gt: torch.Tensor,
    *,
    skip_prompt_frame: bool,
) -> dict[str, float]:
    """Return numerators/counts so the dataset summary is pixel-weighted."""

    if skip_prompt_frame:
        alpha, alpha_gt = alpha[:, 1:], alpha_gt[:, 1:]
        mask_logits, mask_gt = mask_logits[:, 1:], mask_gt[:, 1:]
    if alpha.numel() == 0:
        raise ValueError("the clip needs at least one scored frame")
    error = (alpha.float() - alpha_gt.float()).abs()
    predicted_mask = mask_logits >= 0
    true_mask = mask_gt >= 0.5
    intersection = (predicted_mask & true_mask).sum().item()
    union = (predicted_mask | true_mask).sum().item()
    return {
        "alpha_abs_sum": float(error.sum().item()),
        "alpha_sq_sum": float(error.square().sum().item()),
        "alpha_pixels": float(error.numel()),
        "alpha_sad_sum": float(error.sum(dim=(-1, -2, -3)).sum().item() / 1000),
        "frames": float(alpha.shape[0] * alpha.shape[1]),
        "mask_intersection": float(intersection),
        "mask_union": float(union),
    }


def metrics_from_totals(totals: dict[str, float]) -> dict[str, float]:
    pixels = max(totals["alpha_pixels"], 1)
    frames = max(totals["frames"], 1)
    return {
        "alpha_mae": totals["alpha_abs_sum"] / pixels,
        "alpha_mse": totals["alpha_sq_sum"] / pixels,
        "alpha_sad_per_frame": totals["alpha_sad_sum"] / frames,
        "mask_iou": totals["mask_intersection"] / max(totals["mask_union"], 1),
        "frames": totals["frames"],
    }


def add_totals(destination: dict[str, float], source: dict[str, float]) -> None:
    for key, value in source.items():
        destination[key] = destination.get(key, 0.0) + value


def save_preview(
    path: Path,
    frame: torch.Tensor,
    alpha_gt: torch.Tensor,
    alpha_pred: torch.Tensor,
    mask_gt: torch.Tensor,
    mask_pred: torch.Tensor,
) -> None:
    def grey(value: torch.Tensor) -> Image.Image:
        array = value.detach().float().clamp(0, 1).mul(255).byte().cpu().numpy()
        return Image.fromarray(array, "L").convert("RGB")

    srgb = linear_to_srgb(frame.detach().float().clamp(0, 1))
    rgb = srgb.mul(255).byte().permute(1, 2, 0).cpu().numpy()
    panels = [
        Image.fromarray(rgb, "RGB"),
        grey(mask_gt[0]),
        grey(mask_pred[0]),
        grey(alpha_gt[0]),
        grey(alpha_pred[0]),
    ]
    scale = 3
    width, height = panels[0].size
    canvas = Image.new("RGB", (len(panels) * width * scale, (height + 22) * scale), "white")
    painter = ImageDraw.Draw(canvas)
    for index, (label, panel) in enumerate(zip(
        ("input", "mask GT", "mask pred", "alpha GT", "alpha pred"), panels
    )):
        x = index * width * scale
        canvas.paste(panel.resize((width * scale, height * scale)), (x, 22 * scale))
        painter.text((x + 3, 4), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--variant", choices=("tiny", "base_plus"), default="base_plus")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--clip-length", type=int, default=8)
    parser.add_argument("--max-sequences", type=int, default=0, help="0 evaluates the full split")
    parser.add_argument("--preview-count", type=int, default=12)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if args.clip_length < 2 or args.max_sequences < 0 or args.preview_count < 0:
        parser.error("clip-length must be >=2 and counts must be nonnegative")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("official SAM2Matting baseline requires a CUDA GPU")
    args.output_root.mkdir(parents=True, exist_ok=True)
    dataset = RCTransPRISMDataset(
        args.data_root / args.split,
        clip_length=args.clip_length,
        strict_contract=True,
        semantic_only=True,
        require_split_kind=args.split,
    )
    if not dataset:
        raise RuntimeError(f"empty {args.split} split")
    count = min(len(dataset), args.max_sequences or len(dataset))
    if count != len(dataset):
        print(f"PRISM_SAM2_BASELINE_SUBSET sequences={count}/{len(dataset)}", flush=True)
    checkpoint_sha = sha256_file(args.checkpoint)
    model = OfficialSAM2MattingClipBackbone.from_pretrained(
        args.source_root, args.checkpoint, variant=args.variant, device="cuda"
    ).eval()
    import wandb

    online = wandb.init(
        entity="humangpt", project="PRISM", name=args.run_name,
        group="sam2matting-original-baseline", job_type="baseline-evaluation",
        config={
            "model": "official SAM2Matting", "variant": args.variant,
            "checkpoint_sha256": checkpoint_sha, "split": args.split,
            "clip_length": args.clip_length, "max_sequences": args.max_sequences,
            "prompt": "ground-truth first-frame object mask only",
            "fine_tuning": False, "flow_guidance": False,
        },
    )
    all_totals: dict[str, float] = {}
    propagated_totals: dict[str, float] = {}
    elapsed_seconds = 0.0
    per_sequence = args.output_root / "per_sequence.jsonl"
    with per_sequence.open("w", encoding="utf-8") as records:
        for index in range(count):
            sample = dataset[index]
            values = sample["tensors"]
            frames = values["frames"][None].cuda(non_blocking=True)
            mask_gt = values["object_mask"][None].cuda(non_blocking=True)
            alpha_gt = values["alpha"][None].cuda(non_blocking=True)
            batch, time_count, _, height, width = frames.shape
            srgb = linear_to_srgb(frames).clamp(0, 1)
            resized = F.interpolate(
                srgb.flatten(0, 1), size=(model.image_size, model.image_size),
                mode="bilinear", align_corners=False, antialias=True,
            ).reshape(batch, time_count, 3, model.image_size, model.image_size)
            normalized = normalize_sam2_training_frames(resized)
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                result = model.forward_mam2_clip(
                    normalized, first_frame_mask_inputs=mask_gt[:, 0],
                    original_size=(height, width),
                )
            torch.cuda.synchronize()
            elapsed_seconds += time.perf_counter() - start
            alpha = F.interpolate(
                result.alpha_matte.float().flatten(0, 1),
                size=(height, width), mode="bilinear", align_corners=False,
            ).reshape_as(alpha_gt).clamp(0, 1)
            mask_logits = F.interpolate(
                result.mask_logits.float().flatten(0, 1),
                size=(height, width), mode="bilinear", align_corners=False,
            ).reshape_as(mask_gt)
            all_scores = score_clip(alpha, alpha_gt, mask_logits, mask_gt, skip_prompt_frame=False)
            propagated_scores = score_clip(
                alpha, alpha_gt, mask_logits, mask_gt, skip_prompt_frame=True
            )
            add_totals(all_totals, all_scores)
            add_totals(propagated_totals, propagated_scores)
            entry = {
                "sequence_id": sample["sequence_id"],
                "all_frames": metrics_from_totals(all_scores),
                "propagated_frames": metrics_from_totals(propagated_scores),
            }
            records.write(json.dumps(entry) + "\n")
            records.flush()
            if index < args.preview_count:
                preview = args.output_root / "previews" / f"{index:04d}.png"
                save_preview(
                    preview, frames[0, -1], alpha_gt[0, -1], alpha[0, -1],
                    mask_gt[0, -1], (mask_logits[0, -1] >= 0).float(),
                )
                wandb.log({"preview": wandb.Image(str(preview), caption=sample["sequence_id"])}, step=index + 1)
            if (index + 1) % 10 == 0 or index + 1 == count:
                current = metrics_from_totals(propagated_totals)
                print(
                    "PRISM_SAM2_BASELINE_PROGRESS "
                    f"sequences={index + 1}/{count} "
                    f"propagated_alpha_mae={current['alpha_mae']:.6f} "
                    f"propagated_mask_iou={current['mask_iou']:.6f}", flush=True,
                )
                wandb.log({f"propagated/{key}": value for key, value in current.items()}, step=index + 1)
            model.clear_mam2_cache()
    summary = {
        "model": "released SAM2Matting (no PRISM modules)",
        "variant": args.variant,
        "checkpoint_sha256": checkpoint_sha,
        "split": args.split,
        "prompt": "ground-truth first-frame object mask only",
        "evaluated_sequences": count,
        "available_sequences": len(dataset),
        "clip_length": args.clip_length,
        "all_frames": metrics_from_totals(all_totals),
        "propagated_frames": metrics_from_totals(propagated_totals),
        "forward_seconds": elapsed_seconds,
        "forward_seconds_per_frame": elapsed_seconds / max(all_totals["frames"], 1),
        "wandb_url": online.url,
    }
    if not all(math.isfinite(value) for group in (summary["all_frames"], summary["propagated_frames"]) for value in group.values()):
        raise RuntimeError("nonfinite SAM2Matting baseline metrics")
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    wandb.summary.update(summary)
    wandb.finish()
    print("PRISM_SAM2_BASELINE_COMPLETE " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
