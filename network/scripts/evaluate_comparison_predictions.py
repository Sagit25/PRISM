#!/usr/bin/env python3
"""Score aligned PRISM test predictions without inventing unavailable outputs.

Each prediction file is ``predictions/{index:04d}.npz`` with ``sequence_id``
and any of ``background``, ``alpha``, ``mask``, ``foreground``, ``flow`` or
``recomposition`` in CHW float32 format. ``foreground`` is reserved for a
physical straight-F prediction; generic RGBA/omnimatte layers must use a
different key. Alpha is used only as a mask fallback, never as a GT metric.
All methods use the same ordered test sequences and frame index.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.metrics import structural_similarity

from refractive_mam2.color import linear_to_srgb
from refractive_mam2.dataset import RCTransPRISMDataset
from comparison_semantic import comparison_mask, observed_foreground_cutout


def _region_metrics(pred: np.ndarray, gt: np.ndarray, region: np.ndarray) -> dict[str, float] | None:
    region = region.astype(bool)
    if not region.any():
        return None
    difference = pred.astype(np.float64) - gt.astype(np.float64)
    selected = difference[:, region]
    l1 = float(np.abs(selected).mean())
    mse = float(np.square(selected).mean())
    return {
        "l1": l1, "l2_mse": mse, "rmse": math.sqrt(mse),
        "psnr_db": -10.0 * math.log10(max(mse, 1e-12)),
        "pixel_count": int(region.sum()),
    }


def _ssim_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    # Structural similarity is computed in sRGB display space. Pixel errors
    # and PSNR are computed in linear RGB; both conventions are explicit.
    p = linear_to_srgb(torch.from_numpy(pred).clamp(0, 1)).permute(1, 2, 0).numpy()
    g = linear_to_srgb(torch.from_numpy(gt).clamp(0, 1)).permute(1, 2, 0).numpy()
    _, score = structural_similarity(p, g, channel_axis=2, data_range=1.0, win_size=7, full=True)
    return score.mean(axis=2)


def score_rgb(pred: np.ndarray, gt: np.ndarray, regions: dict[str, np.ndarray]) -> dict:
    if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[0] != 3:
        raise ValueError(f"RGB shapes do not match: {pred.shape}, {gt.shape}")
    if not np.isfinite(pred).all():
        raise ValueError("prediction contains NaN or Inf")
    pred, gt = pred.clip(0, 1), gt.clip(0, 1)
    ssim_map = _ssim_map(pred, gt)
    result = {}
    for name, region in regions.items():
        row = _region_metrics(pred, gt, region)
        if row is not None:
            row["ssim_srgb"] = float(ssim_map[region].mean())
        result[name] = row
    return result


def score_mask(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    p, g = pred[0] >= 0.5, gt[0] >= 0.5
    intersection = int((p & g).sum())
    return {
        "iou": intersection / max(int((p | g).sum()), 1),
        "f1": 2 * intersection / max(int(p.sum() + g.sum()), 1),
        "precision": intersection / max(int(p.sum()), 1),
        "recall": intersection / max(int(g.sum()), 1),
    }


def _lpips_model(device: torch.device):
    import lpips
    return lpips.LPIPS(net="alex").to(device).eval()


def lpips_score(model, pred: np.ndarray, gt: np.ndarray, device: torch.device) -> float:
    p = linear_to_srgb(torch.from_numpy(pred).to(device).clamp(0, 1))[None]
    g = linear_to_srgb(torch.from_numpy(gt).to(device).clamp(0, 1))[None]
    with torch.no_grad():
        return float(model(2 * p - 1, 2 * g - 1).item())


def region_composited_lpips(model, pred: np.ndarray, gt: np.ndarray, region: np.ndarray, device: torch.device) -> float | None:
    """LPIPS after copying GT outside the region; not a native masked LPIPS."""
    if not region.any():
        return None
    isolated = np.where(region[None], pred, gt)
    return lpips_score(model, isolated, gt, device)


def _as_chw(value: np.ndarray, channels: int) -> np.ndarray:
    if value.ndim != 3 or value.shape[0] != channels:
        raise ValueError(f"expected [{channels},H,W], got {value.shape}")
    return np.asarray(value, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, default=4)
    parser.add_argument("--clip-length", type=int, default=8)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--prompt-policy", required=True)
    args = parser.parse_args()
    if args.max_sequences < 0 or not 0 <= args.frame_index < args.clip_length:
        parser.error("invalid sequence count or frame index")
    dataset = RCTransPRISMDataset(args.data_root / "test", clip_length=args.clip_length, strict_contract=True)
    count = min(len(dataset), args.max_sequences or len(dataset))
    args.output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    perceptual = None
    metrics_by_key: dict[str, list[float]] = {}
    mask_sources: dict[str, int] = {}
    records = args.output_root / "per_sequence.jsonl"
    with records.open("w", encoding="utf-8") as stream:
        for index in range(count):
            path = args.prediction_root / f"{index:04d}.npz"
            if not path.is_file():
                raise FileNotFoundError(f"missing prediction for sequence {index}: {path}")
            sample = dataset[index]
            gt = sample["tensors"]
            with np.load(path, allow_pickle=False) as archive:
                if str(archive["sequence_id"].item()) != sample["sequence_id"]:
                    raise ValueError(f"sequence mismatch at {index}")
                values = {key: archive[key].copy() for key in archive.files if key != "sequence_id"}
            if "foreground" in values and "foreground_semantics" in values:
                semantics = str(values["foreground_semantics"].item())
                if semantics != "physical_straight_F":
                    raise ValueError(f"{path}: foreground must be physical_straight_F, got {semantics}")
            frame = args.frame_index
            object_mask = gt["object_mask"][frame, 0].numpy() >= 0.5
            masks = gt["object_mask"][:, 0].numpy() >= 0.5
            boundary = binary_dilation(object_mask, iterations=2) ^ binary_erosion(object_mask, iterations=2)
            regions = {
                "whole": np.ones_like(object_mask, dtype=bool),
                "object": object_mask,
                "boundary": boundary,
                "outside": ~object_mask,
                "never_directly_visible": object_mask & masks.all(axis=0),
                "visible_in_other_frame": object_mask & ~masks.all(axis=0),
            }
            row: dict = {"index": index, "sequence_id": sample["sequence_id"], "frame_index": frame}
            for key, target_key in (
                ("background", "counterfactual_background"),
                ("foreground", "straight_foreground"),
                ("recomposition", "frames"),
            ):
                if key not in values:
                    continue
                prediction = _as_chw(values[key], 3)
                target_tensor = gt[target_key] if key == "background" else gt[target_key][frame]
                target = target_tensor.numpy()
                row[key] = score_rgb(prediction, target, regions)
                if perceptual is None:
                    perceptual = _lpips_model(device)
                row[key]["whole"]["lpips_srgb_alex"] = lpips_score(perceptual, prediction, target, device)
                for region_name in ("object", "never_directly_visible"):
                    if row[key][region_name] is not None:
                        row[key][region_name]["lpips_srgb_region_composited"] = region_composited_lpips(
                            perceptual, prediction, target, regions[region_name], device
                        )
            predicted_mask, mask_source = comparison_mask(values, object_mask.shape)
            if predicted_mask is not None:
                gt_mask = (gt["object_mask"][frame].numpy() >= 0.5).astype(np.float32)
                row["mask"] = score_mask(predicted_mask, gt_mask)
                row["mask_source"] = mask_source
                mask_sources[mask_source] = mask_sources.get(mask_source, 0) + 1
                observed = gt["frames"][frame].numpy()
                cutout = observed_foreground_cutout(observed, predicted_mask)
                cutout_gt = observed_foreground_cutout(observed, gt_mask)
                row["observed_foreground_cutout"] = score_rgb(cutout, cutout_gt, regions)
                if perceptual is None:
                    perceptual = _lpips_model(device)
                row["observed_foreground_cutout"]["whole"]["lpips_srgb_alex"] = lpips_score(
                    perceptual, cutout, cutout_gt, device
                )
                for region_name in ("object", "boundary", "outside"):
                    if row["observed_foreground_cutout"][region_name] is not None:
                        row["observed_foreground_cutout"][region_name]["lpips_srgb_region_composited"] = region_composited_lpips(
                            perceptual, cutout, cutout_gt, regions[region_name], device
                        )
            if "flow" in values:
                pred = _as_chw(values["flow"], 2)
                target = gt["refractive_flow"][frame].numpy()
                valid = object_mask & (gt["confidence"][frame, 0].numpy() > 0)
                epe = np.linalg.norm(pred - target, axis=0)[valid]
                row["flow"] = None if not len(epe) else {
                    "epe_px": float(epe.mean()), "bad_1px": float((epe > 1).mean()),
                    "bad_3px": float((epe > 3).mean()), "valid_pixels": int(len(epe)),
                }
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            def collect(prefix: str, value):
                if isinstance(value, dict):
                    for child, item in value.items():
                        collect(f"{prefix}/{child}" if prefix else child, item)
                elif isinstance(value, (int, float)) and math.isfinite(value) and not prefix.endswith("pixel_count"):
                    metrics_by_key.setdefault(prefix, []).append(float(value))
            collect("", row)
            if (index + 1) % 25 == 0 or index + 1 == count:
                print(f"PRISM_COMPARISON_PROGRESS model={args.model_name} sequences={index + 1}/{count}", flush=True)
    aggregate = {key: float(np.mean(values)) for key, values in metrics_by_key.items() if key not in ("index", "frame_index")}
    summary = {
        "model": args.model_name, "prompt_policy": args.prompt_policy,
        "split": "test", "sequences": count, "available_sequences": len(dataset),
        "frame_index": args.frame_index, "metric_aggregation": "mean of per-sequence scores",
        "rgb_pixel_metrics_space": "linear", "ssim_lpips_space": "sRGB",
        "matting_protocol": "No alpha-vs-alpha score. Compare binary object masks and observed RGB input cut out by those masks; cutout is not physical straight F.",
        "mask_fallback": "If a model lacks a semantic mask, binarize its own alpha at 0.5; record mask_source per sequence.",
        "mask_source_counts": mask_sources,
        "masked_lpips_definition": "Prediction is replaced by GT outside the named region before full-frame LPIPS; receptive fields may cross its boundary.",
        "metrics": aggregate,
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("PRISM_COMPARISON_COMPLETE " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
