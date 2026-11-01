#!/usr/bin/env python3
"""Write an honest, progressively updated held-out comparison report."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


METHODS = (
    ("Telea", "prism-kubectl-baseline-comparison-v1-full/telea_oracle_mask"),
    ("Temporal", "prism-kubectl-baseline-comparison-v1-full/temporal_visible_oracle_masks"),
    ("Navier-Stokes", "prism-kubectl-classical-additional-v1-full/navier_stokes_oracle_mask"),
    ("Biharmonic", "prism-kubectl-classical-additional-v1-full/biharmonic_oracle_mask"),
    ("Nearest-visible", "prism-kubectl-classical-additional-v1-full/nearest_visible_oracle_masks"),
    ("GrabCut", "prism-kubectl-classical-additional-v1-full/grabcut_oracle_box"),
    ("SAM2Matting", "prism-kubectl-baseline-comparison-v1-full/sam2matting_baseplus"),
    ("ObjectClear", "prism-kubectl-objectclear-comparison-v1-full/objectclear"),
    ("MatAnyone", "prism-kubectl-matanyone-comparison-v1-full/matanyone"),
    ("ProPainter", "prism-kubectl-propainter-comparison-v1-full/propainter"),
    ("PRISM_v26", "prism-kubectl-v26-comparison-full-v2/prism_v26"),
    ("PRISM_joint_v15", "prism-kubectl-joint-v15-comparison-v1-full/joint_v15_flow"),
    ("MatAnyone2", "prism-kubectl-matanyone2-comparison-v2-full/matanyone2"),
    ("MAM2", "prism-kubectl-mam2-comparison-v2-full/mam2"),
    ("ROSE", "prism-kubectl-rose-comparison-v2-full/rose"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    metrics: set[str] = set()
    status: list[tuple[str, str, str]] = []
    predictions: list[str] = []
    for name, relative in METHODS:
        root = args.root / relative
        source = None
        for candidate in (root / "summary_extended.json", root / "summary.json"):
            if not candidate.is_file():
                continue
            payload = json.loads(candidate.read_text())
            if payload.get("sequences") == 960:
                source = candidate
                break
        if source is None:
            status.append((name, "pending or incomplete", str(root)))
            continue
        payload = json.loads(source.read_text())
        if (payload.get("split"), payload.get("frame_index"), payload.get("available_sequences")) != ("test", 4, 960):
            status.append((name, "incompatible protocol", str(source)))
            continue
        metrics.update(payload["metrics"])
        rows.append({
            "method": name,
            "source": str(source),
            "prompt_policy": payload["prompt_policy"],
            "evaluated_sequences": 960,
            **payload["metrics"],
        })
        predictions.append(f"{name}={root / 'predictions'}")
        status.append((name, "full 960 complete", str(source)))
    fields = ["method", "source", "prompt_policy", "evaluated_sequences", *sorted(metrics)]
    with (args.output / "metrics_all_available.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    primary = "background/object/psnr_db"
    ranking = sorted(
        ((row["method"], float(row[primary])) for row in rows if row.get(primary) is not None),
        key=lambda item: item[1], reverse=True,
    )
    with (args.output / "prespecified_background_psnr.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("method", primary))
        writer.writerows(ranking)
    lines = [
        "# PRISM held-out comparison status",
        "",
        f"Updated: {datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')}",
        f"Complete full-test methods: {len(rows)}/{len(METHODS)}. Each complete row covers all 960 test sequences at frame 4.",
        "",
        "Primary background metric was fixed before seeing the final test scores: mean per-sequence linear-RGB object-region PSNR.",
        "Methods that do not output a background have no background PSNR; see metrics_all_available.csv for their alpha/mask results.",
        "Oracle masks, boxes, and trimaps are disclosed in prompt_policy; ranks are descriptive because input privileges differ.",
        "",
        "| Method | Full test | Source |",
        "| --- | --- | --- |",
    ]
    lines += [f"| {name} | {state} | `{source}` |" for name, state, source in status]
    lines += ["", "## Prespecified background PSNR", "", "| Method | Object PSNR (dB) |", "| --- | ---: |"]
    lines += [f"| {name} | {score:.3f} |" for name, score in ranking]
    lines += [
        "", "ROSE is first checked on one sequence; only a passing smoke starts its full test in the queued batch.",
        "Omnimatte, LaMa, ViTMatte, FBA, E2FGVI-HQ, and optical controls lack a verified runner for this test contract and are not represented as completed runs.",
        "The report updates as the queued comparisons complete; a queued job is not a finished result.", "",
    ]
    (args.output / "status.md").write_text("\n".join(lines))
    (args.output / "available_predictions.txt").write_text("\n".join(predictions) + "\n")
    print(f"PRISM_MORNING_REPORT_UPDATED complete={len(rows)} output={args.output}", flush=True)


if __name__ == "__main__":
    main()
