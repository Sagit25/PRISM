#!/usr/bin/env python3
"""Add color, luminance, edge, and tail-error scores to frozen test predictions.

The original summary and per-sequence records are preserved. This tool writes
``summary_extended.json`` and ``per_sequence_extended.jsonl`` alongside them.
Every score uses the same sequence ID, frame, and region definitions as the
common comparison evaluator. Missing outputs remain missing.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_erosion, sobel
from skimage.color import deltaE_ciede2000, rgb2lab

from refractive_mam2.color import linear_to_srgb
from refractive_mam2.dataset import RCTransPRISMDataset


def _rgb_hwc(linear: np.ndarray) -> np.ndarray:
    value = torch.from_numpy(np.ascontiguousarray(linear.clip(0, 1)))
    return linear_to_srgb(value).permute(1, 2, 0).numpy().astype(np.float64)


def binary_f1_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Precision/recall/F1 for two binary supports, with empty/empty = 1."""
    pred, gt = prediction.astype(bool), target.astype(bool)
    true_positive = int((pred & gt).sum())
    pred_count, gt_count = int(pred.sum()), int(gt.sum())
    precision = true_positive / pred_count if pred_count else float(gt_count == 0)
    recall = true_positive / gt_count if gt_count else float(pred_count == 0)
    f1 = 2 * true_positive / (pred_count + gt_count) if pred_count + gt_count else 1.0
    return {"precision": precision, "recall": recall, "f1": f1}


def boundary_f1(prediction: np.ndarray, target: np.ndarray, tolerance_px: int = 2) -> float:
    """Symmetric edge F1, matching edges within a pixel tolerance."""
    pred = prediction.astype(bool)
    gt = target.astype(bool)
    pred_edge = pred ^ binary_erosion(pred)
    gt_edge = gt ^ binary_erosion(gt)
    pred_count, gt_count = int(pred_edge.sum()), int(gt_edge.sum())
    if pred_count + gt_count == 0:
        return 1.0
    if pred_count == 0 or gt_count == 0:
        return 0.0
    precision = float((pred_edge & binary_dilation(gt_edge, iterations=tolerance_px)).sum()) / pred_count
    recall = float((gt_edge & binary_dilation(pred_edge, iterations=tolerance_px)).sum()) / gt_count
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def extended_rgb_metrics(
    prediction: np.ndarray, target: np.ndarray, regions: dict[str, np.ndarray],
) -> dict[str, dict[str, float] | None]:
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[0] != 3:
        raise ValueError("RGB prediction and target must match [3,H,W]")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("RGB prediction or target contains NaN/Inf")
    pred = np.asarray(prediction.clip(0, 1), dtype=np.float64)
    gt = np.asarray(target.clip(0, 1), dtype=np.float64)
    pred_srgb, gt_srgb = _rgb_hwc(pred), _rgb_hwc(gt)
    lab_pred, lab_gt = rgb2lab(pred_srgb), rgb2lab(gt_srgb)
    color_difference = deltaE_ciede2000(lab_pred, lab_gt)
    chroma_difference = np.linalg.norm(lab_pred[..., 1:] - lab_gt[..., 1:], axis=-1)
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
    y_pred = np.tensordot(weights, pred, axes=(0, 0))
    y_gt = np.tensordot(weights, gt, axes=(0, 0))
    pred_gradient = np.hypot(sobel(y_pred, axis=0) / 8, sobel(y_pred, axis=1) / 8)
    gt_gradient = np.hypot(sobel(y_gt, axis=0) / 8, sobel(y_gt, axis=1) / 8)
    scores = {}
    for name, region in regions.items():
        region = region.astype(bool)
        if not region.any():
            scores[name] = None
            continue
        srgb_difference = pred_srgb[region] - gt_srgb[region]
        srgb_mse = float(np.square(srgb_difference).mean())
        luminance_difference = y_pred[region] - y_gt[region]
        luminance_mse = float(np.square(luminance_difference).mean())
        linear_error = np.abs(pred[:, region] - gt[:, region])
        scores[name] = {
            "l1_srgb": float(np.abs(srgb_difference).mean()),
            "l2_mse_srgb": srgb_mse,
            "rmse_srgb": math.sqrt(srgb_mse),
            "psnr_db_srgb": -10 * math.log10(max(srgb_mse, 1e-12)),
            "delta_e2000_srgb": float(color_difference[region].mean()),
            "lab_chroma_error": float(chroma_difference[region].mean()),
            "luminance_l1_linear": float(np.abs(luminance_difference).mean()),
            "luminance_psnr_db_linear": -10 * math.log10(max(luminance_mse, 1e-12)),
            "sobel_gradient_l1_linear": float(
                np.abs(pred_gradient[region] - gt_gradient[region]).mean()
            ),
            "absolute_error_p95_linear": float(np.percentile(linear_error, 95)),
        }
    return scores


def extend_one(
    data_root: Path, prediction_root: Path, output_root: Path,
) -> dict:
    original_path = output_root / "summary.json"
    original = json.loads(original_path.read_text())
    count = original["sequences"]
    frame = original["frame_index"]
    dataset = RCTransPRISMDataset(data_root / "test", clip_length=8, strict_contract=True)
    if count > len(dataset):
        raise ValueError("summary sequence count exceeds test dataset")
    records = (output_root / "per_sequence.jsonl").open(encoding="utf-8")
    metric_values: dict[str, list[float]] = {}
    out_path = output_root / "per_sequence_extended.jsonl"
    with records, out_path.open("w", encoding="utf-8") as output:
        for index in range(count):
            line = records.readline()
            if not line:
                raise ValueError(f"missing original per-sequence record {index}")
            row = json.loads(line)
            sample = dataset[index]
            if row["index"] != index or row["sequence_id"] != sample["sequence_id"]:
                raise ValueError(f"original score sequence mismatch at {index}")
            path = prediction_root / f"{index:04d}.npz"
            with np.load(path, allow_pickle=False) as archive:
                if str(archive["sequence_id"].item()) != sample["sequence_id"]:
                    raise ValueError(f"prediction sequence mismatch at {index}")
                prediction = {key: archive[key] for key in archive.files if key != "sequence_id"}
            gt = sample["tensors"]
            object_mask = gt["object_mask"][frame, 0].numpy() >= 0.5
            all_masks = gt["object_mask"][:, 0].numpy() >= 0.5
            boundary = binary_dilation(object_mask, iterations=2) ^ binary_erosion(
                object_mask, iterations=2
            )
            regions = {
                "whole": np.ones_like(object_mask, dtype=bool),
                "object": object_mask,
                "boundary": boundary,
                "outside": ~object_mask,
                "never_directly_visible": object_mask & all_masks.all(axis=0),
                "visible_in_other_frame": object_mask & ~all_masks.all(axis=0),
            }
            for key, gt_name in (
                ("background", "counterfactual_background"),
                ("foreground", "straight_foreground"),
                ("recomposition", "frames"),
            ):
                if key not in prediction:
                    continue
                target_tensor = gt[gt_name] if key == "background" else gt[gt_name][frame]
                target = target_tensor.numpy()
                key_regions = dict(regions)
                if key == "foreground":
                    alpha_valid = gt["alpha"][frame, 0].numpy() >= 0.05
                    key_regions["valid_object"] = object_mask & alpha_valid
                    key_regions["valid_boundary"] = boundary & object_mask & alpha_valid
                extended = extended_rgb_metrics(prediction[key], target, key_regions)
                for region_name, values in extended.items():
                    if values is None:
                        continue
                    row.setdefault(key, {}).setdefault(region_name, {}).update(values)
                    for metric_name, value in values.items():
                        metric_values.setdefault(f"{key}/{region_name}/{metric_name}", []).append(value)
            if "alpha" in prediction:
                pred_alpha = prediction["alpha"][0].clip(0, 1)
                gt_alpha = gt["alpha"][frame, 0].numpy().clip(0, 1)
                if pred_alpha.shape != gt_alpha.shape or not np.isfinite(pred_alpha).all():
                    raise ValueError("alpha prediction has invalid shape or values")
                for threshold, label in ((0.1, "t0p1"), (0.5, "t0p5"), (0.9, "t0p9")):
                    pred_support, gt_support = pred_alpha >= threshold, gt_alpha >= threshold
                    values = binary_f1_metrics(pred_support, gt_support)
                    values["boundary_f1_2px"] = boundary_f1(pred_support, gt_support)
                    row.setdefault("alpha_binary", {})[label] = values
                    for metric_name, value in values.items():
                        metric_values.setdefault(f"alpha_binary/{label}/{metric_name}", []).append(value)
            if "mask" in prediction:
                pred_mask = prediction["mask"][0]
                if pred_mask.shape != object_mask.shape or not np.isfinite(pred_mask).all():
                    raise ValueError("mask prediction has invalid shape or values")
                pred_support = pred_mask >= 0.5
                values = binary_f1_metrics(pred_support, object_mask)
                values["boundary_f1_2px"] = boundary_f1(pred_support, object_mask)
                row.setdefault("mask", {}).update({
                    key: value for key, value in values.items() if key != "f1"
                })
                for metric_name, value in values.items():
                    if metric_name != "f1":
                        metric_values.setdefault(f"mask/{metric_name}", []).append(value)
            output.write(json.dumps(row, sort_keys=True) + "\n")
            if (index + 1) % 100 == 0 or index + 1 == count:
                print(f"PRISM_EXTENDED_METRICS_PROGRESS model={original['model']} {index + 1}/{count}", flush=True)
        if records.readline():
            raise ValueError("original per-sequence file has extra records")
    summary = dict(original)
    summary["metrics"] = dict(original["metrics"])
    summary["metrics"].update({key: float(np.mean(values)) for key, values in metric_values.items()})
    summary["extended_metrics_definition"] = (
        "CIEDE2000 and Lab chroma use clipped sRGB/D65; luminance and Sobel use "
        "clipped linear RGB. Region boundary gradients use neighboring pixels. "
        "Foreground valid_object requires GT alpha >= 0.05."
    )
    summary["f1_definition"] = (
        "Mask F1 uses object-mask support at threshold 0.5. Alpha support F1 "
        "uses separate prediction and GT thresholds 0.1, 0.5, 0.9; boundary "
        "F1 matches edge pixels within two pixels. Both empty supports score 1. "
        "RGB background and foreground intensities have no threshold-free F1."
    )
    (output_root / "summary_extended.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n"
    )
    print(f"PRISM_EXTENDED_METRICS_COMPLETE model={original['model']} output={output_root}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    extend_one(args.data_root, args.prediction_root, args.output_root)


if __name__ == "__main__":
    main()
