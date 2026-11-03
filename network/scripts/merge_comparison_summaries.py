#!/usr/bin/env python3
"""Build a complete, explicit metric table from finished common evaluations.

Every input must describe the same split and frame. Different sequence counts
are retained as separate cohorts instead of being silently compared.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", nargs="+", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    metric_names = set()
    seen = set()
    reference: tuple[str, int, int] | None = None
    for item in args.summary:
        name, separator, value = item.partition("=")
        if not separator or not name or not value or name in seen:
            parser.error("summary must be unique NAME=PATH")
        seen.add(name)
        source = Path(value)
        payload = json.loads(source.read_text())
        identity = (payload["split"], payload["frame_index"], payload["available_sequences"])
        if reference is None:
            reference = identity
        elif identity != reference:
            raise ValueError(f"test split/frame/available count mismatch in {source}: {identity}")
        # Historical common-evaluator summaries contain alpha scores against
        # PRISM's physical transmittance alpha. They are not comparable to
        # semantic matting alpha and must not appear in the paper table.
        comparable_metrics = {
            key: value for key, value in payload["metrics"].items()
            if not (key.startswith("alpha/") or key.startswith("alpha_"))
        }
        metric_names.update(comparable_metrics)
        rows.append({
            "method": name, "source": str(source), "prompt_policy": payload["prompt_policy"],
            "matting_protocol": "mask_and_observed_cutout" if "matting_protocol" in payload else "legacy_unrescored",
            "evaluated_sequences": payload["sequences"],
            **comparable_metrics,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["method", "source", "prompt_policy", "matting_protocol", "evaluated_sequences", *sorted(metric_names)]
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"PRISM_COMPARISON_TABLE methods={len(rows)} metrics={len(metric_names)} output={args.output}")


if __name__ == "__main__":
    main()
