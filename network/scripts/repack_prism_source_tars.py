#!/usr/bin/env python3
"""Turn verified renderer-export tar files into streaming PRISM shards.

Each input tar contains one renderer project directory. The output strips that
directory, preserves whole sequences in one shard, and publishes the archive
manifest only after every output tar has been hashed and verified.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import pathlib
import tarfile
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any


SPLITS = ("train", "validation", "test")
GENERATOR_VERSION = "v17_pose_aligned_trace"


@dataclasses.dataclass(frozen=True)
class Source:
    split: str
    stem: str
    directory: pathlib.Path
    expected_sequences: int


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_source(value: str) -> Source:
    try:
        split, stem, directory, count = value.split(":", 3)
        source = Source(split, stem, pathlib.Path(directory), int(count))
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(
            "--source must be SPLIT:STEM:DIRECTORY:EXPECTED_SEQUENCES"
        ) from error
    if split not in SPLITS or not stem or source.expected_sequences < 1:
        raise argparse.ArgumentTypeError("invalid source split, stem, or count")
    return source


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _source_files(source: Source) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    archive = source.directory / f"{source.stem}.tar"
    checksum = source.directory / f"{source.stem}.tar.sha256"
    manifest = source.directory / f"{source.stem}.dataset_manifest.json"
    marker = source.directory / f"{source.stem}.generation_complete"
    for required in (archive, checksum, manifest, marker):
        if not required.is_file():
            raise FileNotFoundError(required)
    if "split=" + source.split not in marker.read_text(encoding="utf-8"):
        raise ValueError(f"generation marker has wrong split: {marker}")
    return archive, checksum, manifest


def _source_manifest(source: Source) -> tuple[pathlib.Path, dict[str, Any], str]:
    archive, checksum, manifest_path = _source_files(source)
    expected = checksum.read_text(encoding="utf-8").split()[0]
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError(f"invalid SHA-256 sidecar: {checksum}")
    actual = sha256_file(archive)
    if actual != expected:
        raise RuntimeError(f"source archive SHA-256 mismatch: {archive}")
    manifest = _read_json(manifest_path)
    film = manifest.get("configuration", {}).get("Camera", {}).get("film", {})
    if (film.get("width"), film.get("height")) != (64, 64):
        raise ValueError(f"source is not native 64x64: {manifest_path}")
    if source.split == "train":
        shard = manifest.get("shard", {})
        if not shard.get("enabled") or int(shard.get("count", 0)) != 5:
            raise ValueError(f"unexpected train shard manifest: {manifest_path}")
    print(
        f"PRISM_SOURCE_TAR_VERIFIED stem={source.stem} split={source.split} "
        f"sha256={actual}",
        flush=True,
    )
    return archive, manifest, actual


def _copy_source(
    source: Source,
    archive_path: pathlib.Path,
    output_path: pathlib.Path,
    seen_paths: set[str],
    split_resources: dict[str, dict[str, set[str]]],
) -> tuple[int, int, int]:
    prefix = f"{source.stem}/{source.split}/"
    sequence_prefixes: set[str] = set()
    file_count = 0
    uncompressed_bytes = 0
    with tarfile.open(archive_path, "r:") as original, tarfile.open(
        output_path, "w", format=tarfile.PAX_FORMAT
    ) as repacked:
        for member in original:
            if member.isdir():
                continue
            if not member.isfile():
                raise RuntimeError(f"unsafe source tar member: {member.name}")
            if not member.name.startswith(prefix):
                # Source manifest and completion markers live beside the split.
                if member.name.startswith(f"{source.stem}/"):
                    continue
                raise RuntimeError(f"unexpected source tar path: {member.name}")
            relative = member.name[len(prefix) :]
            if not relative or pathlib.PurePosixPath(relative).name != relative:
                raise RuntimeError(f"unexpected nested source path: {member.name}")
            if relative in {".", ".."} or ".." in pathlib.PurePosixPath(relative).parts:
                raise RuntimeError(f"unsafe source tar path: {member.name}")
            output_name = f"{source.split}/{relative}"
            if output_name in seen_paths:
                raise RuntimeError(f"duplicate dataset member: {output_name}")
            seen_paths.add(output_name)

            body = original.extractfile(member)
            if body is None:
                raise RuntimeError(f"unreadable source tar member: {member.name}")
            if relative.endswith("_sequence_meta.json"):
                payload = body.read()
                metadata = json.loads(payload)
                if metadata.get("generator_version") != GENERATOR_VERSION:
                    raise ValueError(f"wrong generator version: {member.name}")
                shape = metadata.get("shape_path")
                background = metadata.get("background_path")
                if not isinstance(shape, str) or not shape:
                    raise ValueError(f"missing shape_path: {member.name}")
                if not isinstance(background, str) or not background:
                    raise ValueError(f"missing background_path: {member.name}")
                sequence_prefixes.add(relative[: -len("_sequence_meta.json")])
                split_resources[source.split]["shapes"].add(shape)
                split_resources[source.split]["backgrounds"].add(background)
                body = io.BytesIO(payload)
                member_size = len(payload)
            else:
                member_size = member.size
            info = tarfile.TarInfo(output_name)
            info.size = member_size
            info.mode = 0o644
            info.mtime = member.mtime
            repacked.addfile(info, body)
            file_count += 1
            uncompressed_bytes += member_size

    if len(sequence_prefixes) != source.expected_sequences:
        raise RuntimeError(
            f"{source.stem}: expected {source.expected_sequences} sequences, "
            f"found {len(sequence_prefixes)}"
        )
    if file_count < len(sequence_prefixes) * 100:
        raise RuntimeError(f"{source.stem}: extracted file count is incomplete")
    return file_count, uncompressed_bytes, len(sequence_prefixes)


def _shard_record(
    path: pathlib.Path, file_count: int, uncompressed_bytes: int, aligned: bool
) -> dict[str, Any]:
    digest = sha256_file(path)
    with tarfile.open(path, "r:") as archive:
        if not any(member.isfile() for member in archive):
            raise RuntimeError(f"empty output tar: {path}")
    return {
        "name": path.name,
        "sha256": digest,
        "tar_bytes": path.stat().st_size,
        "uncompressed_bytes": uncompressed_bytes,
        "file_count": file_count,
        "sequence_aligned": aligned,
    }


def repack(sources: list[Source], output: pathlib.Path) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    if len(sources) != 7 or [source.split for source in sources].count("train") != 5:
        raise ValueError("expected exactly five train, one validation, and one test source")
    if [source.split for source in sources].count("validation") != 1 or [
        source.split for source in sources
    ].count("test") != 1:
        raise ValueError("expected one validation and one test source")
    output.mkdir(parents=True, exist_ok=True)
    components: dict[str, dict[str, Any]] = {
        name: {"complete": True, "file_count": 0, "uncompressed_bytes": 0, "shards": []}
        for name in ("metadata", *SPLITS)
    }
    resources: dict[str, dict[str, set[str]]] = {
        split: {"shapes": set(), "backgrounds": set()} for split in SPLITS
    }
    seen_paths: set[str] = set()
    sequence_counts = defaultdict(int)
    source_hashes: dict[str, str] = {}
    first_manifest: dict[str, Any] | None = None
    for index, source in enumerate(sources):
        archive, source_manifest, digest = _source_manifest(source)
        if first_manifest is None:
            first_manifest = source_manifest
        elif source_manifest.get("configuration") != first_manifest.get("configuration"):
            raise ValueError(f"source configuration mismatch: {source.stem}")
        source_hashes[source.stem] = digest
        shard_name = f"{source.split}-{index:05d}.tar"
        output_path = output / shard_name
        file_count, byte_count, sequence_count = _copy_source(
            source, archive, output_path, seen_paths, resources
        )
        sequence_counts[source.split] += sequence_count
        record = _shard_record(output_path, file_count, byte_count, True)
        component = components[source.split]
        component["shards"].append(record)
        component["file_count"] += file_count
        component["uncompressed_bytes"] += byte_count
        _write_json(output / f"{shard_name}.json", record)
        print(
            f"PRISM_PACK_SHARD_COMPLETE split={source.split} name={shard_name} "
            f"sequences={sequence_count} sha256={record['sha256']}",
            flush=True,
        )

    assert first_manifest is not None
    for kind in ("shapes", "backgrounds"):
        for index, first in enumerate(SPLITS):
            for second in SPLITS[index + 1 :]:
                overlap = resources[first][kind] & resources[second][kind]
                if overlap:
                    raise RuntimeError(
                        f"{kind} leakage between {first} and {second}: "
                        f"{sorted(overlap)[:5]}"
                    )
    dataset_manifest = dict(first_manifest)
    dataset_manifest["resources"] = {
        split: {kind: sorted(values) for kind, values in resources[split].items()}
        for split in SPLITS
    }
    dataset_manifest["counts"] = {
        split: {kind: len(values) for kind, values in resources[split].items()}
        for split in SPLITS
    }
    dataset_manifest["run_splits"] = list(SPLITS)
    dataset_manifest["shard"] = {
        "enabled": False,
        "count": 1,
        "index": 0,
        "splits": list(SPLITS),
        "unit": "verified_source_tar",
    }
    dataset_manifest["materialization"] = {
        "resource_manifest_source": "sequence_metadata",
        "sequence_counts": dict(sequence_counts),
    }
    dataset_manifest["source_tar_sha256"] = source_hashes
    manifest_path = output / "dataset_manifest.json"
    _write_json(manifest_path, dataset_manifest)
    metadata_tar = output / "metadata-00000.tar"
    with tarfile.open(metadata_tar, "w", format=tarfile.PAX_FORMAT) as archive:
        archive.add(manifest_path, arcname=manifest_path.name)
    metadata_record = _shard_record(
        metadata_tar, 1, manifest_path.stat().st_size, False
    )
    components["metadata"]["shards"].append(metadata_record)
    components["metadata"]["file_count"] = 1
    components["metadata"]["uncompressed_bytes"] = manifest_path.stat().st_size
    _write_json(output / "metadata-00000.tar.json", metadata_record)

    now = datetime.now(timezone.utc).isoformat()
    archive_manifest = {
        "format": "prism-uncompressed-tar-shards-v1",
        "complete": True,
        "created_at": now,
        "completed_at": now,
        "source_volume_id": "seven-verified-v64singleflow-renderer-exports",
        "destination_volume_id": "vessl-export",
        "components": components,
        "totals": {
            "file_count": sum(c["file_count"] for c in components.values()),
            "uncompressed_bytes": sum(
                c["uncompressed_bytes"] for c in components.values()
            ),
            "shard_count": sum(len(c["shards"]) for c in components.values()),
            "sequence_count": sum(sequence_counts.values()),
        },
    }
    _write_json(output / "archive_manifest.json", archive_manifest)
    print(
        "PRISM_FULL_64_ARCHIVE_COMPLETE "
        f"sequences={archive_manifest['totals']['sequence_count']} "
        f"shards={archive_manifest['totals']['shard_count']} "
        f"files={archive_manifest['totals']['file_count']}",
        flush=True,
    )
    return archive_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=parse_source, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    repack(args.source, args.output)


if __name__ == "__main__":
    main()
