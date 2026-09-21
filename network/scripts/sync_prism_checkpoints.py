#!/usr/bin/env python3
"""Upload and restore PRISM checkpoints with exact VESSL object keys.

The VESSL ``storage copy-file`` command treats a URI ending in a filename as a
directory and appends the local filename.  Older PRISM runs therefore contain
objects such as ``best.pt/best.pt``.  This helper uses the VESSL object-store
API directly, writes new objects at exact keys, and can restore both the exact
and the legacy doubled-filename layouts.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
from collections.abc import Iterable

from repack_vessl_dataset import ObjectInfo, VesslObjectStore


VOLUME_URI = re.compile(
    r"^volume://(?P<storage>[^/]+)/(?P<volume>[^/]+)(?:/(?P<prefix>.*))?$"
)


def parse_volume_uri(uri: str) -> tuple[str, str, str]:
    match = VOLUME_URI.fullmatch(uri.rstrip("/"))
    if match is None:
        raise ValueError(
            "VESSL URI must be volume://STORAGE/VOLUME[/PREFIX], " f"received {uri!r}"
        )
    return (
        match.group("storage"),
        match.group("volume"),
        (match.group("prefix") or "").strip("/"),
    )


def join_key(*parts: str) -> str:
    return "/".join(part.strip("/") for part in parts if part.strip("/"))


def select_remote_object(
    objects: Iterable[ObjectInfo], logical_key: str
) -> ObjectInfo | None:
    """Select an exact object, falling back to the old ``name/name`` layout."""

    candidates = list(objects)
    exact = [obj for obj in candidates if obj.relative_path == logical_key]
    if len(exact) == 1:
        return exact[0]
    basename = pathlib.PurePosixPath(logical_key).name
    legacy_key = f"{logical_key}/{basename}"
    legacy = [obj for obj in candidates if obj.relative_path == legacy_key]
    if len(legacy) == 1:
        return legacy[0]
    return None


def validate_checkpoint(path: pathlib.Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Restored checkpoint is not a non-empty file: {path}")
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != 6:
        raise RuntimeError(f"Unsupported or corrupt PRISM checkpoint: {path}")


def restore_stages(
    store: VesslObjectStore,
    *,
    source_prefix: str,
    output_root: pathlib.Path,
    stages: Iterable[str],
) -> list[str]:
    restored: list[str] = []
    for stage in stages:
        stage_dir = output_root / "checkpoints" / f"stage{stage}"
        stage_prefix = join_key(source_prefix, "checkpoints", f"stage{stage}")
        remote_objects = list(store.iter_prefix(stage_prefix))
        checkpoint_name = f"prism_stage{stage}_best.pt"
        checkpoint_key = join_key(stage_prefix, checkpoint_name)
        marker_key = join_key(stage_prefix, ".training_complete")
        resume_name = f"prism_stage{stage}_resume.pt"
        resume_key = join_key(stage_prefix, resume_name)
        checkpoint = select_remote_object(remote_objects, checkpoint_key)
        marker = select_remote_object(remote_objects, marker_key)
        resume = select_remote_object(remote_objects, resume_key)
        if checkpoint is None and resume is None:
            print(
                f"PRISM_CHECKPOINT_NOT_FOUND stage={stage}; stage will be trained",
                flush=True,
            )
            continue

        stage_dir.mkdir(parents=True, exist_ok=True)
        restored_files: list[pathlib.Path] = []
        for remote, name in ((checkpoint, checkpoint_name), (resume, resume_name)):
            if remote is None:
                continue
            destination = stage_dir / name
            temporary = destination.with_suffix(destination.suffix + ".partial")
            try:
                store.download(remote, temporary)
                validate_checkpoint(temporary)
                os.replace(temporary, destination)
                restored_files.append(destination)
            finally:
                temporary.unlink(missing_ok=True)
        if marker is not None and checkpoint is not None:
            (stage_dir / ".training_complete").touch()
        restored.append(stage)
        print(
            f"PRISM_CHECKPOINT_RESTORED stage={stage} "
            f"complete={marker is not None and checkpoint is not None} "
            f"files={','.join(path.name for path in restored_files)}",
            flush=True,
        )
    return restored


def build_store(storage: str, volume: str) -> VesslObjectStore:
    return VesslObjectStore(
        storage_name=storage,
        source_volume_name=volume,
        destination_volume_name=volume,
    )


def upload_file(source: pathlib.Path, destination_uri: str) -> None:
    storage, volume, key = parse_volume_uri(destination_uri)
    if not key:
        raise ValueError("Upload destination must include an exact object key")
    if not source.is_file():
        raise ValueError(f"Upload source must be a regular file: {source}")
    store = build_store(storage, volume)
    store.upload(source, key)
    print(f"PRISM_CHECKPOINT_UPLOADED {destination_uri}", flush=True)


def restore_from_uri(
    source_uri: str, output_root: pathlib.Path, stages: Iterable[str]
) -> list[str]:
    storage, volume, prefix = parse_volume_uri(source_uri)
    store = build_store(storage, volume)
    return restore_stages(
        store,
        source_prefix=prefix,
        output_root=output_root,
        stages=stages,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    upload = subparsers.add_parser("upload")
    upload.add_argument("--source", type=pathlib.Path, required=True)
    upload.add_argument("--destination-uri", required=True)

    restore = subparsers.add_parser("restore")
    restore.add_argument("--source-uri", required=True)
    restore.add_argument("--output-root", type=pathlib.Path, required=True)
    restore.add_argument("--stage", action="append", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "upload":
        upload_file(args.source, args.destination_uri)
    else:
        restore_from_uri(args.source_uri, args.output_root, args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
