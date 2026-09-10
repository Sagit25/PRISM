#!/usr/bin/env python3
"""Freeze generated PRISM artifacts into a content-addressed manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_splits(root: Path) -> tuple[str, ...]:
    """Return the splits selected for this render invocation.

    A sharded cloud run intentionally renders only one split, while the
    ``resources`` section still records the complete train/validation/test
    partition.  Prefer ``run_splits`` so an artifact manifest can be frozen
    independently for each shard.  The resource-based fallback preserves
    compatibility with manifests written before ``run_splits`` was added.
    """

    dataset_manifest = root / "dataset_manifest.json"
    if not dataset_manifest.is_file():
        raise FileNotFoundError(dataset_manifest)
    payload = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    canonical = ("train", "validation", "test")
    run_splits = payload.get("run_splits")
    if run_splits is not None:
        if not isinstance(run_splits, list) or not run_splits:
            raise ValueError("dataset manifest run_splits must be a non-empty list")
        unknown = sorted(set(run_splits) - set(canonical))
        if unknown:
            raise ValueError(f"dataset manifest contains unknown run_splits: {unknown}")
        if len(run_splits) != len(set(run_splits)):
            raise ValueError("dataset manifest run_splits contains duplicates")
        return tuple(name for name in canonical if name in run_splits)

    resources = payload.get("resources")
    if not isinstance(resources, dict):
        return canonical
    selected = tuple(
        name
        for name in canonical
        if resources.get(name, {}).get("shapes")
    )
    if "train" not in selected or "test" not in selected:
        raise ValueError("dataset manifest must contain non-empty train and test splits")
    return selected


def verify(root: Path, manifest_path: Path) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("format") != "prism-artifact-manifest-v1":
        raise ValueError("unsupported artifact manifest format")
    dataset_manifest = root / "dataset_manifest.json"
    expected_dataset_hash = payload.get("dataset_manifest_sha256")
    if not dataset_manifest.is_file() or sha256(dataset_manifest) != expected_dataset_hash:
        raise RuntimeError("dataset_manifest.json is missing or has changed")
    recorded = {item["path"]: item for item in payload.get("files", [])}
    split_names = tuple(payload.get("splits", ("train", "validation", "test")))
    if split_names != expected_splits(root):
        raise RuntimeError("artifact-manifest splits disagree with dataset manifest")
    actual = {
        str(path.relative_to(root)): path
        for split in split_names
        for path in sorted((root / split).rglob("*"))
        if path.is_file()
    }
    if set(recorded) != set(actual):
        missing = sorted(set(recorded) - set(actual))
        added = sorted(set(actual) - set(recorded))
        raise RuntimeError(
            f"artifact inventory changed: missing={missing[:10]}, added={added[:10]}"
        )
    for relative, path in actual.items():
        item = recorded[relative]
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise RuntimeError(f"artifact changed: {relative}")
    print(
        f"VERIFIED files={len(actual)} "
        f"manifest_sha256={sha256(manifest_path)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="defaults to DATASET_ROOT/artifact_manifest.json",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify the existing manifest instead of replacing it",
    )
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    split_names = expected_splits(root)
    missing = [name for name in split_names if not (root / name).is_dir()]
    if missing:
        raise FileNotFoundError(f"missing split directories: {missing}")
    output = args.output or root / "artifact_manifest.json"
    if args.verify:
        if not output.is_file():
            raise FileNotFoundError(output)
        verify(root, output)
        return
    files = []
    for split in split_names:
        for path in sorted((root / split).rglob("*")):
            if path.is_file():
                files.append(
                    {
                        "path": str(path.relative_to(root)),
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
    dataset_manifest = root / "dataset_manifest.json"
    if not dataset_manifest.is_file():
        raise FileNotFoundError(dataset_manifest)
    payload = {
        "format": "prism-artifact-manifest-v1",
        # Keep the digest portable when an immutable dataset is moved to a
        # different machine or mounted at a different path.
        "dataset_root": ".",
        "dataset_manifest_sha256": sha256(dataset_manifest),
        "splits": list(split_names),
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"FROZEN files={payload['file_count']} bytes={payload['total_bytes']} "
        f"manifest_sha256={sha256(output)} output={output}"
    )


if __name__ == "__main__":
    main()
