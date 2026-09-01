"""Validate and compare matched PRISM-Base and PRISM-Diffusion metric files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def compare_metric_files(
    base_path: str | Path,
    diffusion_path: str | Path,
    *,
    preservation_tolerance: float = 0.0,
) -> dict[str, object]:
    base_path = Path(base_path)
    diffusion_path = Path(diffusion_path)
    base = json.loads(base_path.read_text(encoding="utf-8"))
    diffusion = json.loads(diffusion_path.read_text(encoding="utf-8"))
    for key in ("dataset",):
        if base.get(key) != diffusion.get(key):
            raise ValueError(f"Base and Diffusion disagree on {key}")
    base_repro = base.get("reproducibility", {})
    diffusion_repro = diffusion.get("reproducibility", {})
    for key in (
        "checkpoint_sha256",
        "sam2_checkpoint_sha256",
        "dataset_artifact_manifest_sha256",
        "prompt_mode",
        "seed",
    ):
        if base_repro.get(key) != diffusion_repro.get(key):
            raise ValueError(f"Base and Diffusion disagree on reproducibility.{key}")
    for label, payload in (("base", base), ("diffusion", diffusion)):
        preservation = payload.get("metrics", {}).get("evidence_preservation_l1")
        if preservation is None:
            raise ValueError(f"{label} metrics omit evidence_preservation_l1")
        if float(preservation) > preservation_tolerance:
            raise ValueError(
                f"{label} changes supported evidence: {preservation} > "
                f"{preservation_tolerance}"
            )
    base_metrics = base["metrics"]
    diffusion_metrics = diffusion["metrics"]
    common = sorted(set(base_metrics) & set(diffusion_metrics))
    return {
        "base": str(base_path),
        "diffusion": str(diffusion_path),
        "matched_reproducibility": {
            key: base_repro.get(key)
            for key in (
                "checkpoint_sha256",
                "sam2_checkpoint_sha256",
                "dataset_artifact_manifest_sha256",
                "prompt_mode",
                "seed",
            )
        },
        "metrics": {
            name: {
                "base": float(base_metrics[name]),
                "diffusion": float(diffusion_metrics[name]),
                "diffusion_minus_base": float(diffusion_metrics[name])
                - float(base_metrics[name]),
            }
            for name in common
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("diffusion", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preservation-tolerance", type=float, default=0.0)
    args = parser.parse_args(argv)
    report = compare_metric_files(
        args.base,
        args.diffusion,
        preservation_tolerance=args.preservation_tolerance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"comparison written to {args.output}")


if __name__ == "__main__":
    main()
