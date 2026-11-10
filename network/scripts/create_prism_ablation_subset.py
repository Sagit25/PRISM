#!/usr/bin/env python3
"""Create a reproducible, stratified, read-only view of a PRISM split."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def select(metadata: list[Path], count: int) -> list[Path]:
    if count < 1 or count > len(metadata):
        raise ValueError("subset count must be in [1, number of sequences]")
    groups = {
        "objaverse": sorted(p for p in metadata if p.name.startswith("objaverse_")),
        "procedural": sorted(p for p in metadata if not p.name.startswith("objaverse_")),
    }
    allocation = {
        key: min(len(paths), round(count * len(paths) / len(metadata)))
        for key, paths in groups.items()
    }
    while sum(allocation.values()) < count:
        key = max(groups, key=lambda k: len(groups[k]) - allocation[k])
        allocation[key] += 1
    while sum(allocation.values()) > count:
        key = max(allocation, key=allocation.get)
        allocation[key] -= 1
    chosen = []
    for key, paths in groups.items():
        amount = allocation[key]
        chosen.extend(paths[int((index + 0.5) * len(paths) / amount)]
                      for index in range(amount))
    return sorted(chosen)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    destination = args.destination.resolve()
    if destination == source or source in destination.parents:
        raise ValueError("subset destination must be outside the source split")
    metadata = sorted(source.glob("*_sequence_meta.json"))
    chosen = select(metadata, args.count)
    names = {p.name.removesuffix("_sequence_meta.json") for p in chosen}
    destination.mkdir(parents=True, exist_ok=True)
    linked = 0
    for item in os.scandir(source):
        name = item.name
        sequence = name.split("_frame", 1)[0]
        if sequence not in names:
            sequence = name.removesuffix("_sequence_meta.json")
        if sequence not in names:
            sequence = next((key for key in names if name.startswith(key + "_")), "")
        if sequence not in names:
            continue
        target = destination / name
        if target.is_symlink():
            if target.resolve() != Path(item.path).resolve():
                raise ValueError(f"existing link points elsewhere: {target}")
        elif target.exists():
            raise ValueError(f"existing non-link file: {target}")
        else:
            target.symlink_to(item.path)
        linked += 1
    selected = [p.name.removesuffix("_sequence_meta.json") for p in chosen]
    manifest = destination / "ablation_subset.json"
    manifest.write_text(json.dumps({
        "source": str(source), "count": len(selected), "files": linked,
        "sequences": selected,
    }, indent=2) + "\n")
    if len(list(destination.glob("*_sequence_meta.json"))) != args.count:
        raise RuntimeError("not every selected sequence was linked")
    print(f"PRISM_ABLATION_SUBSET_READY count={len(selected)} files={linked} path={destination}")


if __name__ == "__main__":
    main()
