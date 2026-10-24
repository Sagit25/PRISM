"""Reject a completed-stage marker whose checkpoint belongs to another variant.

The multi-stage launcher may intentionally reuse Stage 1 semantics, but it
must never silently reuse a completed Stage 2--4 physics checkpoint trained
with different operators, priors, losses, or clip schedules.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _normalized(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def verify_checkpoint(
    path: Path,
    expected: list[tuple[str, str, str]],
) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != 8:
        raise ValueError("completed checkpoint is not PRISM format v8")
    sections = {
        "matter": payload.get("pipeline_config", {}).get("matter", {}),
        "background": payload.get("pipeline_config", {}).get("background", {}),
        "arguments": payload.get("metadata", {}).get("arguments", {}),
    }
    for section, key, wanted in expected:
        if section not in sections or not isinstance(sections[section], dict):
            raise ValueError(f"unknown or missing checkpoint section: {section}")
        if key not in sections[section]:
            raise ValueError(f"completed checkpoint omits {section}.{key}")
        actual = _normalized(sections[section][key])
        if actual != wanted:
            raise ValueError(
                f"completed checkpoint mismatch for {section}.{key}: "
                f"saved={actual!r}, requested={wanted!r}; use a new output root"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--expect", nargs=3, metavar=("SECTION", "KEY", "VALUE"),
        action="append", default=[],
    )
    args = parser.parse_args()
    verify_checkpoint(args.checkpoint, [tuple(item) for item in args.expect])
    print(f"PRISM_COMPLETED_STAGE_CHECKPOINT_MATCHES_CONFIG {args.checkpoint}")


if __name__ == "__main__":
    main()
