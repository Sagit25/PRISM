#!/usr/bin/env python3
"""Score saved Stage-4 traces on one common 128px test frame and GT support.

This CPU-only postprocessor does not rerun the model. All ablation arms use
frame 4 and the eight-frame ground-truth visibility history, including arms
that inferred from only one frame. Missing predictions are never imputed.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import torch
from PIL import Image

from refractive_mam2.dataset import RCTransPRISMDataset
from refractive_mam2.renderer import recompose

METRICS = (
    "render_full_psnr_db",
    "render_object_psnr_db",
    "background_gt_true_hole_psnr_db",
    "foreground_valid_psnr_db",
    "flow_epe_px",
)


def _psnr(prediction: np.ndarray, truth: np.ndarray, support: np.ndarray) -> float | None:
    if not np.any(support):
        return None
    error = prediction.astype(np.float64) - truth.astype(np.float64)
    mse = float(np.square(error[:, support]).mean())
    return -10.0 * math.log10(max(mse, 1e-12))


def _mean_flow_magnitude(flow: np.ndarray, valid: np.ndarray) -> float | None:
    if not np.any(valid):
        return None
    return float(np.linalg.norm(flow, axis=0)[valid].mean())


def _score_one(prediction: dict[str, np.ndarray], gt: dict, metadata: dict,
               frame: int, sequence_id: str) -> dict:
    required = {"background", "straight_foreground", "refractive_flow", "reconstruction"}
    missing = required.difference(prediction)
    if missing:
        raise ValueError(f"{sequence_id}: missing trace tensors {sorted(missing)}")
    video = gt["frames"].numpy()
    object_mask = gt["object_mask"].numpy()[:, 0] >= 0.5
    alpha = gt["alpha"].numpy()[:, 0]
    # The support is fixed by GT for *every* arm. Single-frame inference must
    # not enlarge its evaluation hole merely because it saw fewer frames.
    occupied = object_mask | (alpha > 1e-3)
    true_hole = occupied.all(axis=0)
    object_roi = occupied[frame]
    foreground_valid = (alpha[frame] >= 0.05) & object_roi
    flow_valid = (gt["refractive_validity"].numpy()[frame, 0] >= 0.5) & object_roi

    def frame_tensor(name: str, channels: int) -> np.ndarray:
        value = prediction[name]
        if value.ndim != 4 or value.shape[1] != channels:
            raise ValueError(f"{sequence_id}: invalid {name} shape {value.shape}")
        selected = 0 if value.shape[0] == 1 else frame
        if selected >= value.shape[0]:
            raise ValueError(f"{sequence_id}: {name} lacks requested frame {frame}")
        output = value[selected]
        if not np.isfinite(output).all():
            raise ValueError(f"{sequence_id}: nonfinite {name}")
        return output

    render = frame_tensor("reconstruction", 3)
    foreground = frame_tensor("straight_foreground", 3)
    flow = frame_tensor("refractive_flow", 2)
    background = prediction["background"]
    if background.shape != video[frame].shape or not np.isfinite(background).all():
        raise ValueError(f"{sequence_id}: invalid background shape or values")
    gt_flow = gt["refractive_flow"].numpy()[frame]
    error = np.linalg.norm(flow - gt_flow, axis=0)
    transmission_rgb = metadata.get("material_transmission_rgb")
    neutral = False
    if isinstance(transmission_rgb, (list, tuple)) and len(transmission_rgb) == 3:
        channels = np.asarray(transmission_rgb, dtype=np.float64)
        neutral = bool(np.isfinite(channels).all() and channels.mean() >= 0.8
                       and np.ptp(channels) <= 0.05)
    elif np.any(object_roi):
        # Older manifests may omit material parameters. The rendered GT color
        # transmission is then the source of the same neutral-color criterion.
        channels = np.median(gt["color_transmission"].numpy()[frame, :, object_roi], axis=0)
        neutral = bool(np.isfinite(channels).all() and channels.mean() >= 0.8
                       and np.ptp(channels) <= 0.05)
    return {
        "sequence_id": sequence_id,
        "neutral_transparent": neutral,
        "gt_flow_mean_magnitude_px": _mean_flow_magnitude(gt_flow, flow_valid),
        "render_full_psnr_db": _psnr(render, video[frame], np.ones_like(object_roi)),
        "render_object_psnr_db": _psnr(render, video[frame], object_roi),
        "background_gt_true_hole_psnr_db": _psnr(
            background, gt["counterfactual_background"].numpy(), true_hole
        ),
        "foreground_valid_psnr_db": _psnr(
            foreground, gt["straight_foreground"].numpy()[frame], foreground_valid
        ),
        "flow_epe_px": float(error[flow_valid].mean()) if np.any(flow_valid) else None,
        "pixel_counts": {
            "object": int(object_roi.sum()),
            "gt_true_hole": int(true_hole.sum()),
            "foreground_valid": int(foreground_valid.sum()),
            "flow_valid": int(flow_valid.sum()),
        },
    }


def _aggregate(rows: list[dict]) -> dict:
    return {
        "sequences": len(rows),
        "metrics": {
            name: {
                "mean": float(np.mean(values)) if values else None,
                "valid_sequences": len(values),
            }
            for name in METRICS
            for values in [[row[name] for row in rows if row[name] is not None]]
        },
    }


def _selected_trace(trace_root: Path, sequence_id: str) -> dict[str, np.ndarray]:
    sample_dir = trace_root / sequence_id
    with (sample_dir / "steps.json").open(encoding="utf-8") as stream:
        steps = json.load(stream)["steps"]
    if not steps:
        raise ValueError(f"{sequence_id}: empty refinement trace")
    last = steps[-1]
    trace = sample_dir / f"{last['step']:02d}_{last['phase']}.npz"
    with np.load(trace, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _display_png_uri(linear_rgb: np.ndarray) -> str:
    rgb = np.clip(linear_rgb, 0.0, 1.0)
    srgb = np.where(rgb <= 0.0031308, 12.92 * rgb,
                    1.055 * np.power(rgb, 1 / 2.4) - 0.055)
    image = Image.fromarray(np.uint8(np.rint(np.moveaxis(srgb, 0, -1) * 255)))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _novel_background_examples(dataset: RCTransPRISMDataset, rows: list[dict],
                               trace_root: Path, output_root: Path,
                               frame: int) -> list[dict]:
    """Transfer one prediction to a held-out paired background, same object/pose."""
    indices = {row["sequence_id"]: index for index, row in enumerate(rows)}
    candidates = []
    for group_id, members in sorted(dataset.pair_groups.items()):
        available = [index for index in members
                     if str(dataset.records[index].prefix.relative_to(dataset.root)) in indices]
        if len(available) < 2:
            continue
        source_index, target_index = available[:2]
        sequence_id = str(dataset.records[source_index].prefix.relative_to(dataset.root))
        source_row = rows[indices[sequence_id]]
        magnitude = source_row["gt_flow_mean_magnitude_px"]
        if magnitude is not None:
            candidates.append((magnitude, source_row["neutral_transparent"],
                               group_id, source_index, target_index))
    if not candidates:
        return []
    candidates.sort(key=lambda item: (item[0], item[2]))
    chosen = []
    for candidate in (
        candidates[-1],
        next((item for item in reversed(candidates) if item[1]), None),
        candidates[len(candidates) // 2],
        candidates[len(candidates) // 4],
    ):
        if candidate is not None and candidate[2] not in {item[2] for item in chosen}:
            chosen.append(candidate)
    figures = output_root / "novel_background"
    figures.mkdir(parents=True, exist_ok=True)
    records = []
    for rank, (magnitude, neutral, group_id, source_index, target_index) in enumerate(chosen):
        source = dataset[source_index]
        target = dataset[target_index]
        sequence_id = source["sequence_id"]
        prediction = _selected_trace(trace_root, sequence_id)

        def channel(name: str) -> np.ndarray:
            value = prediction[name]
            index = 0 if value.shape[0] == 1 else frame
            return value[index]

        alpha = channel("alpha")
        foreground = channel("straight_foreground")
        tau = channel("transmittance")
        flow = channel("refractive_flow")
        residual = channel("residual")
        new_background = target["tensors"]["counterfactual_background"].numpy()
        with torch.no_grad():
            composite, _ = recompose(
                torch.from_numpy(alpha[None]),
                torch.from_numpy((alpha * foreground)[None]),
                torch.from_numpy(new_background[None]),
                torch.from_numpy(flow[None]),
                transmittance=torch.from_numpy(tau[None]),
                residual=torch.from_numpy(residual[None]),
            )
        novel = composite[0].numpy()
        # Match the pipeline's direct-observation behavior without an oracle
        # object mask: keep the replacement background outside predicted optics.
        support = ((alpha[0] > 0.01)
                   | (np.linalg.norm(flow, axis=0) > 0.1)
                   | (np.max(np.abs(residual), axis=0) > 0.01))
        novel = np.where(support[None], novel, new_background)
        paired_gt = target["tensors"]["frames"].numpy()[frame]
        source_input = source["tensors"]["frames"].numpy()[frame]
        object_roi = source["tensors"]["object_mask"].numpy()[frame, 0] >= 0.5
        score = _psnr(novel, paired_gt, object_roi)
        panels = (
            ("Input I", source_input),
            ("Novel B", new_background),
            ("Transferred render", novel),
            ("Paired GT", paired_gt),
            ("Error x4", np.abs(novel - paired_gt) * 4),
        )
        tile = 146
        gap = 12
        width = len(panels) * tile + (len(panels) + 1) * gap
        parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="190" viewBox="0 0 {width} 190">',
                 f'<rect width="{width}" height="190" fill="white"/>']
        for column, (label, image) in enumerate(panels):
            x = gap + column * (tile + gap)
            parts.append(f'<text x="{x}" y="17" font-family="Times New Roman" font-size="12">{escape(label)}</text>')
            parts.append(f'<image x="{x}" y="25" width="{tile}" height="{tile}" href="{_display_png_uri(image)}"/>')
        parts.append('</svg>')
        svg = figures / f"{rank + 1:02d}_{sequence_id}.svg"
        svg.write_text("\n".join(parts) + "\n", encoding="utf-8")
        records.append({"source_sequence_id": sequence_id,
                        "target_sequence_id": target["sequence_id"],
                        "paired_background_group_id": group_id,
                        "gt_flow_mean_magnitude_px": magnitude,
                        "neutral_transparent": neutral,
                        "novel_render_object_psnr_db": score,
                        "svg": str(svg)})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="dataset root containing test/")
    parser.add_argument("--trace-root", type=Path, required=True,
                        help="results/prism-diffusion-flux-fill/refinement_trace")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, default=4)
    parser.add_argument("--clip-length", type=int, default=8,
                        help="GT history length, even for single-frame inference")
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--wandb-mode", choices=("disabled", "online"), default="disabled")
    parser.add_argument("--wandb-entity", default="sukhuny-seoul-national-university")
    parser.add_argument("--wandb-project", default="PRISM")
    parser.add_argument("--wandb-group", default="prism-128-ablation")
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()
    if not 0 <= args.frame_index < args.clip_length or args.max_sequences < 0:
        parser.error("invalid frame or sequence count")

    dataset = RCTransPRISMDataset(
        args.data_root / "test", clip_length=args.clip_length, strict_contract=True,
        verify_numeric_contract=False,
    )
    count = min(len(dataset), args.max_sequences or len(dataset))
    rows: list[dict] = []
    for index in range(count):
        sample = dataset[index]
        sequence_id = sample["sequence_id"]
        values = _selected_trace(args.trace_root, sequence_id)
        rows.append(_score_one(values, sample["tensors"], sample["metadata"],
                               args.frame_index, sequence_id))
        if (index + 1) % 50 == 0 or index + 1 == count:
            print(f"PRISM_ABLATION_REGION_PROGRESS {index + 1}/{count}", flush=True)

    magnitudes = [row["gt_flow_mean_magnitude_px"] for row in rows
                  if row["gt_flow_mean_magnitude_px"] is not None]
    threshold = float(np.quantile(magnitudes, 0.75)) if magnitudes else None
    groups = {
        "all": rows,
        "large_refraction_gt_top_quartile": [
            row for row in rows if threshold is not None
            and row["gt_flow_mean_magnitude_px"] is not None
            and row["gt_flow_mean_magnitude_px"] >= threshold
        ],
        "neutral_transparent": [row for row in rows if row["neutral_transparent"]],
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    novel_examples = _novel_background_examples(
        dataset, rows, args.trace_root, args.output_root,
        args.frame_index,
    )
    summary = {
        "frame_index": args.frame_index,
        "gt_clip_length": args.clip_length,
        "requested_sequences": count,
        "aggregation": "unweighted mean of per-sequence metrics, fixed GT supports",
        "large_refraction_definition": "top quartile of per-sequence GT mean flow magnitude on valid object pixels",
        "large_refraction_threshold_px": threshold,
        "neutral_transparent_definition": "material RGB transmission mean >= 0.8 and channel range <= 0.05",
        "true_hole_definition": "object mask or alpha>0.001 in every one of eight GT frames",
        "foreground_valid_definition": "GT alpha>=0.05 inside GT object",
        "groups": {name: _aggregate(group) for name, group in groups.items()},
        "novel_background_examples": novel_examples,
    }
    with (args.output_root / "per_sequence.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_root / "table.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("group", "sequences", *METRICS))
        for name, result in summary["groups"].items():
            writer.writerow((name, result["sequences"], *(
                result["metrics"][metric]["mean"] for metric in METRICS
            )))
    if args.wandb_mode == "online":
        import wandb
        if not args.wandb_run_name:
            parser.error("--wandb-run-name is required for online logging")
        run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                         group=args.wandb_group, name=args.wandb_run_name,
                         job_type="ablation_region_analysis", config={
                             "frame_index": args.frame_index,
                             "gt_clip_length": args.clip_length,
                             "large_refraction_threshold_px": threshold,
                         })
        for group_name, result in summary["groups"].items():
            run.log({f"{group_name}/{metric}": data["mean"]
                     for metric, data in result["metrics"].items()
                     if data["mean"] is not None})
        run.log({"ablation_regions": wandb.Table(
            columns=["group", "sequences", *METRICS], data=[
                [name, result["sequences"], *(
                    result["metrics"][metric]["mean"] for metric in METRICS
                )] for name, result in summary["groups"].items()
            ])})
        for index, record in enumerate(novel_examples):
            run.log({f"novel_background/example_{index + 1}_object_psnr_db":
                     record["novel_render_object_psnr_db"]})
            run.save(record["svg"], policy="now")
        run.save(str(args.output_root / "summary.json"), policy="now")
        run.save(str(args.output_root / "table.csv"), policy="now")
        run.save(str(args.output_root / "per_sequence.jsonl"), policy="now")
        run.finish()
    print(f"PRISM_ABLATION_REGION_COMPLETE output={args.output_root}", flush=True)


if __name__ == "__main__":
    main()
