#!/usr/bin/env python3
"""Safely materialize PRISM tar shards into the training dataset layout."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import inspect
import json
import os
import pathlib
import tarfile
import time
from typing import Any


MARKER_NAME = ".prism_materialized_complete"
SPLITS = ("train", "validation", "test")


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


def extract_one(
    shard: pathlib.Path, output_root: pathlib.Path, delete_after_extract: bool
) -> str:
    print(f"PRISM_EXTRACT_START shard={shard}", flush=True)
    with tarfile.open(shard, mode="r:") as archive:
        validate_members(archive, output_root)
        extract_kwargs = {}
        if "filter" in inspect.signature(archive.extractall).parameters:
            # All paths and member types were explicitly validated above.
            extract_kwargs["filter"] = "fully_trusted"
        archive.extractall(output_root, **extract_kwargs)
    if delete_after_extract:
        shard.unlink()
        print(f"PRISM_ARCHIVE_LOCAL_COPY_REMOVED shard={shard}", flush=True)
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


def selected_shards(
    manifest: dict[str, Any], max_shards_per_component: int | None
) -> dict[str, str]:
    """Return the complete archive, or a balanced small subset for a smoke run."""

    selected: dict[str, str] = {}
    for component, payload in manifest.get("components", {}).items():
        shards = payload.get("shards", [])
        if max_shards_per_component is not None and component != "metadata":
            shards = shards[:max_shards_per_component]
        for shard in shards:
            selected[shard["name"]] = shard["sha256"]
    if not selected:
        raise RuntimeError("Archive manifest contains no selected shards")
    return selected


def rebuild_resource_manifest(output_root: pathlib.Path) -> bool:
    """Rebuild split resource lists from the materialized sequence metadata.

    Distributed renderer workers write shard-local ``dataset_manifest.json``
    files.  A packed multi-worker dataset can consequently retain the manifest
    from only one worker even though its sequence shards are all valid.  The
    per-sequence metadata is authoritative for the materialized subset, so
    merge it here and still reject real resource overlap between splits.

    Returns ``True`` when a PRISM sequence manifest was rebuilt.  Non-PRISM
    archives without sequence metadata are left unchanged.
    """

    manifest_path = output_root / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    resources: dict[str, dict[str, list[str]]] = {}
    counts: dict[str, dict[str, int]] = {}
    sequence_counts: dict[str, int] = {}
    total_sequences = 0
    for split in SPLITS:
        metadata_paths = sorted(
            (output_root / split).rglob("*_sequence_meta.json")
        )
        total_sequences += len(metadata_paths)
        sequence_counts[split] = len(metadata_paths)
        shapes: set[str] = set()
        backgrounds: set[str] = set()
        for metadata_path in metadata_paths:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            shape = metadata.get("shape_path")
            background = metadata.get("background_path")
            if not isinstance(shape, str) or not shape:
                raise ValueError(f"{metadata_path}: missing shape_path")
            if not isinstance(background, str) or not background:
                raise ValueError(f"{metadata_path}: missing background_path")
            shapes.add(shape)
            backgrounds.add(background)
        resources[split] = {
            "shapes": sorted(shapes),
            "backgrounds": sorted(backgrounds),
        }
        counts[split] = {
            "shapes": len(shapes),
            "backgrounds": len(backgrounds),
        }

    if total_sequences == 0:
        return False
    empty_splits = [split for split, count in sequence_counts.items() if count == 0]
    if empty_splits:
        raise RuntimeError(
            "Materialized PRISM subset has no sequence metadata for: "
            + ", ".join(empty_splits)
        )

    for kind in ("shapes", "backgrounds"):
        for index, first in enumerate(SPLITS):
            first_values = set(resources[first][kind])
            for second in SPLITS[index + 1 :]:
                overlap = first_values & set(resources[second][kind])
                if overlap:
                    examples = sorted(overlap)[:5]
                    raise ValueError(
                        f"Actual {kind} leakage between {first} and {second}: "
                        f"{examples}"
                    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["resources"] = resources
    manifest["counts"] = counts
    manifest["materialization"] = {
        "resource_manifest_source": "sequence_metadata",
        "sequence_counts": sequence_counts,
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    print(
        "PRISM_RESOURCE_MANIFEST_REBUILT "
        + " ".join(
            f"{split}_sequences={sequence_counts[split]}" for split in SPLITS
        ),
        flush=True,
    )
    return True


def _download_remote_shard(
    store: Any,
    obj: Any,
    destination: pathlib.Path,
    expected_sha256: str,
    retries: int,
) -> None:
    """Download one shard with fresh credentials and safe whole-file retries."""

    partial = destination.with_suffix(destination.suffix + ".partial")
    for attempt in range(1, retries + 1):
        with contextlib.suppress(FileNotFoundError):
            partial.unlink()
        try:
            # Federated VESSL credentials are temporary.  Refresh before every
            # large object so the token lifetime is never shared by the full
            # multi-terabyte materialization job.
            store._refresh_source()
            store.download(obj, partial)
            actual_size = partial.stat().st_size
            if actual_size != obj.size:
                raise IOError(
                    f"short read for {obj.relative_path}: expected {obj.size}, "
                    f"got {actual_size}"
                )
            actual_sha256 = sha256_file(partial)
            if actual_sha256 != expected_sha256:
                raise IOError(
                    f"SHA-256 mismatch for {obj.relative_path}: expected "
                    f"{expected_sha256}, got {actual_sha256}"
                )
            partial.replace(destination)
            print(f"PRISM_ARCHIVE_DOWNLOADED shard={obj.relative_path}", flush=True)
            return
        except Exception as error:
            if attempt == retries:
                raise RuntimeError(
                    f"Failed to download {obj.relative_path} after {retries} attempts"
                ) from error
            delay = min(2 ** (attempt - 1), 30)
            print(
                f"PRISM_ARCHIVE_DOWNLOAD_RETRY shard={obj.relative_path} "
                f"attempt={attempt}/{retries} delay={delay}s error={error}",
                flush=True,
            )
            time.sleep(delay)


def materialize_remote(
    storage_name: str,
    archive_volume: str,
    output_root: pathlib.Path,
    download_root: pathlib.Path,
    max_shards_per_component: int | None = None,
    retries: int = 5,
) -> None:
    """Stream VESSL archive shards into local disk without a VESSL import."""

    # Imported lazily so local archive materialization keeps working without
    # the VESSL SDK installed.
    from repack_vessl_dataset import MANIFEST_NAME, VesslObjectStore, atomic_write_json

    store = VesslObjectStore(
        storage_name=storage_name,
        source_volume_name=archive_volume,
        destination_volume_name=archive_volume,
    )
    manifest = store.read_json(MANIFEST_NAME)
    if manifest is None or not manifest.get("complete"):
        raise RuntimeError("Remote archive manifest is missing or incomplete")
    expected = selected_shards(manifest, max_shards_per_component)
    signature_payload = {
        "manifest": manifest,
        "selected_shards": sorted(expected),
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    marker = output_root / MARKER_NAME
    if marker.is_file() and marker.read_text().strip() == signature:
        print(f"PRISM_ARCHIVES_ALREADY_MATERIALIZED output={output_root}", flush=True)
        return

    available = {
        obj.relative_path: obj
        for obj in store.iter_objects("metadata")
        if obj.relative_path in expected
    }
    missing = sorted(set(expected) - set(available))
    if missing:
        raise RuntimeError(f"Missing {len(missing)} remote tar shards: {missing[:5]}")

    output_root.mkdir(parents=True, exist_ok=True)
    download_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / ".prism_materialization_state.json"
    state: dict[str, Any] = {"signature": signature, "completed": []}
    if state_path.is_file():
        candidate = json.loads(state_path.read_text())
        if candidate.get("signature") == signature:
            state = candidate
    completed = set(state.get("completed", []))

    for name in sorted(expected):
        if name in completed:
            print(f"PRISM_ARCHIVE_ALREADY_EXTRACTED shard={name}", flush=True)
            continue
        local_shard = download_root / pathlib.Path(name).name
        _download_remote_shard(
            store,
            available[name],
            local_shard,
            expected[name],
            retries,
        )
        extract_one(local_shard, output_root, delete_after_extract=True)
        completed.add(name)
        state["completed"] = sorted(completed)
        atomic_write_json(state_path, state)

    for required in (
        output_root / "train",
        output_root / "validation",
        output_root / "test",
        output_root / "dataset_manifest.json",
    ):
        if not required.exists():
            raise RuntimeError(f"Materialized dataset is missing {required}")
    rebuild_resource_manifest(output_root)
    marker.write_text(signature + "\n")
    print(
        f"PRISM_REMOTE_MATERIALIZATION_COMPLETE output={output_root} "
        f"shards={len(expected)}",
        flush=True,
    )


def materialize(
    archive_root: pathlib.Path,
    output_root: pathlib.Path,
    workers: int,
    verify_sha256: bool,
    delete_after_extract: bool = False,
) -> None:
    manifest = load_manifest(archive_root)
    expected = expected_shards(manifest)
    signature = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()
    marker = output_root / MARKER_NAME
    if marker.is_file() and marker.read_text().strip() == signature:
        print(f"PRISM_ARCHIVES_ALREADY_MATERIALIZED output={output_root}", flush=True)
        return

    available = {path.name: path for path in archive_root.rglob("*.tar")}
    missing = sorted(set(expected) - set(available))
    if missing:
        raise RuntimeError(f"Missing {len(missing)} tar shards: {missing[:5]}")

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
        futures = [
            executor.submit(extract_one, shard, output_root, delete_after_extract)
            for shard in shards
        ]
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
    rebuild_resource_manifest(output_root)
    marker.write_text(signature + "\n")
    print(f"PRISM_MATERIALIZATION_COMPLETE output={output_root}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive-root", type=pathlib.Path)
    source.add_argument("--archive-volume")
    parser.add_argument("--storage-name", default="vessl-storage")
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument(
        "--download-root",
        type=pathlib.Path,
        default=pathlib.Path("/tmp/prism-archive-downloads"),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument(
        "--max-shards-per-component",
        type=int,
        help="Materialize only the first N train/validation/test shards for a smoke run.",
    )
    parser.add_argument(
        "--skip-sha256",
        action="store_true",
        help="Skip the full tar checksum pass before extraction.",
    )
    parser.add_argument(
        "--delete-after-extract",
        action="store_true",
        help="Delete each verified local tar copy after successful extraction.",
    )
    args = parser.parse_args()
    if args.workers <= 0 or args.retries <= 0:
        parser.error("--workers and --retries must be positive")
    if args.max_shards_per_component is not None and args.max_shards_per_component <= 0:
        parser.error("--max-shards-per-component must be positive")
    if args.archive_root is not None and args.max_shards_per_component is not None:
        parser.error("--max-shards-per-component is only supported with --archive-volume")
    return args


def main() -> int:
    args = parse_args()
    if args.archive_volume is not None:
        materialize_remote(
            args.storage_name,
            args.archive_volume,
            args.output_root,
            args.download_root,
            max_shards_per_component=args.max_shards_per_component,
            retries=args.retries,
        )
    else:
        materialize(
            args.archive_root,
            args.output_root,
            args.workers,
            verify_sha256=not args.skip_sha256,
            delete_after_extract=args.delete_after_extract,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
