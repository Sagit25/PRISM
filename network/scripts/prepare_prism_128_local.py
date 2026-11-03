#!/usr/bin/env python3
"""Verify seven native 128px render jobs and hardlink them into one dataset.

The merged directory contains no copied image tensors and is accepted by the
normal PRISM local-directory loader. Source manifests remain immutable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path


SPLITS = ("train", "validation", "test")
PREFIX = "prism_main_128_20261103"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def projects(generation_root: Path) -> list[tuple[str, Path, str]]:
    return [
        ("train", generation_root / f"{PREFIX}_train_s{index:02d}of05", str(index))
        for index in range(5)
    ] + [
        (split, generation_root / f"{PREFIX}_{split}", split)
        for split in ("validation", "test")
    ]


def wait_for_projects(
    generation_root: Path, *, poll_seconds: int, timeout_seconds: int
) -> list[tuple[str, Path, str]]:
    selected = projects(generation_root)
    start = time.monotonic()
    while True:
        missing = [str(root / ".generation_complete") for _, root, _ in selected
                   if not (root / ".generation_complete").is_file()]
        if not missing:
            return selected
        if poll_seconds == 0 or (
            timeout_seconds and time.monotonic() - start >= timeout_seconds
        ):
            raise RuntimeError(f"128px generation is incomplete: {missing}")
        print(f"PRISM_128_WAIT remaining={len(missing)}", flush=True)
        time.sleep(poll_seconds)


def inspect_sources(
    selected: list[tuple[str, Path, str]], *, verify_content: bool
) -> tuple[dict, list[tuple[Path, dict]], list[dict]]:
    base: dict | None = None
    seen_train_shapes: set[str] = set()
    sources: list[tuple[Path, dict]] = []
    provenance: list[dict] = []
    for split, root, shard_index in selected:
        manifest_path = root / "dataset_manifest.json"
        artifact_path = root / "artifact_manifest.json"
        if not manifest_path.is_file() or not artifact_path.is_file():
            raise FileNotFoundError(f"missing frozen manifests in {root}")
        if verify_content:
            freezer = Path(__file__).resolve().parents[2] / "RCDatasetCreation" / "tools" / "freeze_prism_manifest.py"
            subprocess.run(
                [sys.executable, str(freezer), str(root), "--verify"], check=True
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        if artifact.get("format") != "prism-artifact-manifest-v1":
            raise ValueError(f"unsupported artifact manifest: {root}")
        if artifact.get("dataset_manifest_sha256") != sha256(manifest_path):
            raise ValueError(f"dataset manifest changed after freeze: {root}")
        if artifact.get("file_count") != len(artifact.get("files", [])):
            raise ValueError(f"artifact file inventory is incomplete: {root}")
        if manifest.get("run_splits") != [split] or artifact.get("splits") != [split]:
            raise ValueError(f"wrong split in source manifest: {root}")
        if manifest.get("generator_version") != "v17_pose_aligned_trace":
            raise ValueError(f"unexpected generator version: {root}")
        configuration = manifest.get("configuration", {})
        film = configuration.get("Camera", {}).get("film", {})
        if (film.get("width"), film.get("height")) != (128, 128):
            raise ValueError(f"source is not native 128x128: {root}")
        if configuration.get("frames_per_sequence") != 8:
            raise ValueError(f"source sequence length is not 8: {root}")
        if split == "train":
            shard = manifest.get("shard", {})
            if (shard.get("enabled"), shard.get("count"), shard.get("index")) != (
                True, 5, int(shard_index)
            ):
                raise ValueError(f"wrong train shard contract: {root}")
            shapes = set(manifest["resources"]["train"]["shapes"])
            if not shapes or shapes & seen_train_shapes:
                raise ValueError(f"empty or overlapping train shape shard: {root}")
            seen_train_shapes.update(shapes)
        if base is None:
            base = deepcopy(manifest)
        else:
            for key in ("generator_version", "dataset_seed", "validation_seed", "split_kind", "output_profile"):
                if manifest.get(key) != base.get(key):
                    raise ValueError(f"source manifests disagree on {key}: {root}")
            for other in ("validation", "test"):
                if manifest["resources"][other] != base["resources"][other]:
                    raise ValueError(f"source resource partition differs: {other} {root}")
            if manifest["resources"]["train"]["backgrounds"] != base["resources"]["train"]["backgrounds"]:
                raise ValueError(f"source train backgrounds differ: {root}")
        if not list((root / split).glob("*_sequence_meta.json")):
            raise ValueError(f"source split contains no sequences: {root / split}")
        sources.append((root, artifact))
        provenance.append({
            "split": split,
            "project": root.name,
            "dataset_manifest_sha256": sha256(manifest_path),
            "artifact_manifest_sha256": sha256(artifact_path),
            "file_count": artifact.get("file_count"),
        })
    assert base is not None
    if seen_train_shapes & set(base["resources"]["test"]["shapes"]):
        raise ValueError("train/test shape leakage")
    if seen_train_shapes & set(base["resources"]["validation"]["shapes"]):
        raise ValueError("train/validation shape leakage")
    base["resources"]["train"]["shapes"] = sorted(seen_train_shapes)
    base["counts"]["train"]["shapes"] = len(seen_train_shapes)
    base["run_splits"] = list(SPLITS)
    base["shard"] = {"enabled": False, "count": 1, "index": 0, "splits": list(SPLITS), "unit": "shape"}
    base["configuration"]["project_name"] = "prism_main_128_20261103_merged"
    return base, sources, provenance


def materialize(
    output_root: Path, manifest: dict, sources: list[tuple[Path, dict]],
    provenance: list[dict],
) -> None:
    marker = output_root / ".prism_128_ready"
    if output_root.exists():
        if not marker.is_file():
            raise FileExistsError(f"incomplete output requires inspection: {output_root}")
        if marker.read_text(encoding="utf-8").strip() != sha256(
            output_root / "artifact_manifest.json"
        ):
            raise ValueError(f"merged dataset marker does not match artifact manifest: {output_root}")
        print(f"PRISM_128_ALREADY_READY {output_root}", flush=True)
        return
    partial = output_root.with_name(output_root.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"incomplete staging directory requires inspection: {partial}")
    partial.mkdir(parents=True)
    records: dict[str, dict] = {}
    try:
        for root, artifact in sources:
            for item in artifact["files"]:
                relative = Path(item["path"])
                if (relative.is_absolute() or ".." in relative.parts or not relative.parts
                        or relative.parts[0] not in SPLITS):
                    raise ValueError(f"unsafe source artifact path: {relative}")
                if str(relative) in records:
                    raise ValueError(f"duplicate 128px artifact path: {relative}")
                source = root / relative
                if not source.is_file() or source.stat().st_size != item["bytes"]:
                    raise ValueError(f"source artifact changed: {source}")
                target = partial / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(source, target)
                records[str(relative)] = item
        (partial / "dataset_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        artifact = {
            "format": "prism-artifact-manifest-v1",
            "dataset_root": ".",
            "dataset_manifest_sha256": sha256(partial / "dataset_manifest.json"),
            "splits": list(SPLITS),
            "file_count": len(records),
            "total_bytes": sum(item["bytes"] for item in records.values()),
            "files": [records[key] for key in sorted(records)],
        }
        (partial / "artifact_manifest.json").write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (partial / "source_provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        marker_path = partial / ".prism_128_ready"
        marker_path.write_text(
            sha256(partial / "artifact_manifest.json") + "\n", encoding="utf-8"
        )
        partial.rename(output_root)
    except Exception:
        # Preserve the staging tree for inspection instead of silently
        # replacing a previous verified dataset.
        raise
    print(f"PRISM_128_LOCAL_READY root={output_root} files={len(records)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=0)
    parser.add_argument("--skip-content-verification", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds < 0 or args.timeout_seconds < 0:
        parser.error("poll and timeout seconds must be nonnegative")
    selected = wait_for_projects(
        args.generation_root, poll_seconds=args.poll_seconds,
        timeout_seconds=args.timeout_seconds,
    )
    manifest, sources, provenance = inspect_sources(
        selected, verify_content=not args.skip_content_verification
    )
    if args.check_only:
        print(f"PRISM_128_SOURCE_CHECK_OK sources={len(sources)}", flush=True)
        return
    materialize(args.dataset_root, manifest, sources, provenance)


if __name__ == "__main__":
    main()
