#!/usr/bin/env python3
"""Refresh full-test metrics and matched images as queued methods finish."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from build_comparison_morning_report import METHODS


def full_summary(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return (
        payload.get("split") == "test"
        and payload.get("frame_index") == 4
        and payload.get("available_sequences") == 960
        and payload.get("sequences") == 960
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).resolve().parent
    summaries: list[str] = []
    predictions: list[str] = []
    statuses: list[str] = ["method\tstate\tsource"]
    for name, relative in METHODS:
        root = args.root / relative
        original = root / "summary.json"
        if not full_summary(original):
            statuses.append(f"{name}\tpending_or_incomplete\t{original}")
            continue
        extended = root / "summary_extended.json"
        if not full_summary(extended):
            try:
                subprocess.run(
                    [sys.executable, str(scripts / "extend_comparison_metrics.py"),
                     "--data-root", str(args.data_root),
                     "--prediction-root", str(root / "predictions"),
                     "--output-root", str(root)],
                    check=True,
                )
            except subprocess.CalledProcessError as exc:
                print(f"PRISM_EXTENDED_METRICS_FAILED method={name} code={exc.returncode}", flush=True)
        source = extended if full_summary(extended) else original
        summaries.append(f"{name}={source}")
        predictions.append(f"{name}={root / 'predictions'}")
        statuses.append(f"{name}\t{'extended' if source == extended else 'base_only'}\t{source}")
    if not summaries:
        raise RuntimeError("no complete full-test summary is available")
    table = args.output / "metrics_all_available.csv"
    subprocess.run(
        [sys.executable, str(scripts / "merge_comparison_summaries.py"),
         "--summary", *summaries, "--output", str(table)],
        check=True,
    )
    args.output.joinpath("sources.txt").write_text("\n".join(summaries) + "\n")
    args.output.joinpath("status.tsv").write_text("\n".join(statuses) + "\n")
    image_manifest = args.output / "matched_images_sources.txt"
    source_text = "\n".join(predictions) + "\n"
    if not image_manifest.is_file() or image_manifest.read_text() != source_text:
        subprocess.run(
            [sys.executable, str(scripts / "make_comparison_contact_sheets.py"),
             "--data-root", str(args.data_root),
             "--output-root", str(args.output / "matched_images"),
             "--frame-index", "4", "--indices", "0", "1", "2", "3", "4", "5", "6", "7", "120", "240", "480", "720",
             "--prediction", *predictions],
            check=True,
        )
        image_manifest.write_text(source_text)
    print(f"PRISM_REFRESH_COMPLETE methods={len(summaries)}/{len(METHODS)} table={table}", flush=True)


if __name__ == "__main__":
    main()
