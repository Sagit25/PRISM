#!/usr/bin/env python3
"""Safely materialize PRISM tar shards into the training dataset layout."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import inspect
import json
import os
import pathlib
import tarfile
from typing import Any


MARKER_NAME = ".prism_materialized_complete"


def sha256_file(path: pathlib.Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_members(archive: tarfile.TarFile, output_root: pathlib.Path) -> None:
    root = output_root.resolve()
    for member in archive.getmembers():
        if member.issym() or member.islnk() or member.isdev():
            raise RuntimeError(
                f"Unsafe tar member type in {archive.name}: {member.name}"
            )
        destination = (root / member.name).resolve()
        try:
            destination.relative_to(root)
        except ValueError as error:
            raise RuntimeError(
                f"Path traversal in {archive.name}: {member.name}"
            ) from error


def extract_one(shard: pathlib.Path, output_root: pathlib.Path) -> str:
    print(f"PRISM_EXTRACT_START shard={shard}", flush=True)
    with tarfile.open(shard, mode="r:") as archive:
        validate_members(archive, output_root)
        extract_kwargs = {}
        if "filter" in inspect.signature(archive.extractall).parameters:
            # All paths and member types were explicitly validated above.
            extract_kwargs["filter"] = "fully_trusted"
        archive.extractall(output_root, **extract_kwargs)
    print(f"PRISM_EXTRACT_COMPLETE shard={shard}", flush=True)
    return shard.name


def load_manifest(archive_root: pathlib.Path) -> dict[str, Any]:
    path = archive_root / "archive_manifest.json"
    if not path.is_file():
        matches = list(archive_root.rglob("archive_manifest.json"))
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one archive_manifest.json below {archive_root}, "
                f"found {len(matches)}"
            )
        path = matches[0]
    manifest = json.loads(path.read_text())
    if not manifest.get("complete"):
        raise RuntimeError("Archive manifest is not marked complete")
    return manifest


def expected_shards(manifest: dict[str, Any]) -> dict[str, str]:
    expected: dict[str, str] = {}
    for component in manifest.get("components", {}).values():
        for shard in component.get("shards", []):
            expected[shard["name"]] = shard["sha256"]
    if not expected:
        raise RuntimeError("Archive manifest contains no shards")
    return expected


def materialize(
    archive_root: pathlib.Path,
    output_root: pathlib.Path,
    workers: int,
    verify_sha256: bool,
) -> None:
    manifest = load_manifest(archive_root)
    expected = expected_shards(manifest)
    available = {path.name: path for path in archive_root.rglob("*.tar")}
    missing = sorted(set(expected) - set(available))
    if missing:
        raise RuntimeError(f"Missing {len(missing)} tar shards: {missing[:5]}")

    signature = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()
    marker = output_root / MARKER_NAME
    if marker.is_file() and marker.read_text().strip() == signature:
        print(f"PRISM_ARCHIVES_ALREADY_MATERIALIZED output={output_root}", flush=True)
        return

    output_root.mkdir(parents=True, exist_ok=True)
    shards = [available[name] for name in sorted(expected)]
    if verify_sha256:
        for shard in shards:
            actual = sha256_file(shard)
            if actual != expected[shard.name]:
                raise RuntimeError(
                    f"SHA-256 mismatch for {shard.name}: expected "
                    f"{expected[shard.name]}, got {actual}"
                )
            print(f"PRISM_ARCHIVE_VERIFIED shard={shard.name}", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(extract_one, shard, output_root) for shard in shards]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    for required in (
        output_root / "train",
        output_root / "validation",
        output_root / "test",
        output_root / "dataset_manifest.json",
    ):
        if not required.exists():
            raise RuntimeError(f"Materialized dataset is missing {required}")
    marker.write_text(signature + "\n")
    print(f"PRISM_MATERIALIZATION_COMPLETE output={output_root}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=pathlib.Path, required=True)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--skip-sha256",
        action="store_true",
        help="Skip the full tar checksum pass before extraction.",
    )
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    return args


def main() -> int:
    args = parse_args()
    materialize(
        args.archive_root,
        args.output_root,
        args.workers,
        verify_sha256=not args.skip_sha256,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
