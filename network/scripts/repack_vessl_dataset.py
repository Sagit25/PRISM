#!/usr/bin/env python3
"""Repack a many-file VESSL volume into resumable, uncompressed tar shards.

The VESSL volume importer downloads every object separately and performs an S3
HEAD request for each one.  That is fragile for datasets containing hundreds of
thousands of small files.  This utility reads source objects with GET requests,
packs them into a small number of POSIX tar files, and uploads each completed
shard directly to another VESSL volume.

The source volume is never modified.  A state file is uploaded after every
shard, so a restarted run resumes after the last completely uploaded object.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import dataclasses
import hashlib
import io
import json
import os
import pathlib
import shutil
import sys
import tarfile
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from datetime import datetime, timezone
from typing import Any, BinaryIO, Protocol


GIB = 1024**3
STATE_NAME = "repack_state.json"
MANIFEST_NAME = "archive_manifest.json"
COMPONENTS = ("metadata", "train", "validation", "test")


@dataclasses.dataclass(frozen=True)
class ObjectInfo:
    key: str
    relative_path: str
    size: int
    etag: str = ""
    mtime: int = 0


class ObjectStore(Protocol):
    def iter_objects(self, component: str) -> Iterator[ObjectInfo]: ...

    def download(self, obj: ObjectInfo, destination: pathlib.Path) -> None: ...

    def upload(self, source: pathlib.Path, key: str) -> None: ...

    def read_json(self, key: str) -> dict[str, Any] | None: ...


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: pathlib.Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def default_state(source_volume_id: int, destination_volume_id: int) -> dict[str, Any]:
    return {
        "version": 1,
        "source_volume_id": source_volume_id,
        "destination_volume_id": destination_volume_id,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "complete": False,
        "components": {
            name: {
                "complete": False,
                "last_key": None,
                "next_shard_index": 0,
                "file_count": 0,
                "uncompressed_bytes": 0,
                "shards": [],
            }
            for name in COMPONENTS
        },
    }


def validate_state(
    state: Mapping[str, Any], source_volume_id: int, destination_volume_id: int
) -> None:
    if state.get("source_volume_id") != source_volume_id:
        raise RuntimeError("Existing state belongs to a different source volume")
    if state.get("destination_volume_id") != destination_volume_id:
        raise RuntimeError("Existing state belongs to a different destination volume")
    if state.get("version") != 1:
        raise RuntimeError(f"Unsupported repack state version: {state.get('version')}")


def iter_pending(
    objects: Iterable[ObjectInfo], last_key: str | None
) -> Iterator[ObjectInfo]:
    for obj in objects:
        if last_key is None or obj.key > last_key:
            yield obj


def iter_shard_groups(
    objects: Iterable[ObjectInfo], target_bytes: int
) -> Iterator[list[ObjectInfo]]:
    group: list[ObjectInfo] = []
    size = 0
    for obj in objects:
        if group and size + obj.size > target_bytes:
            yield group
            group = []
            size = 0
        group.append(obj)
        size += obj.size
    if group:
        yield group


def _temp_name(obj: ObjectInfo) -> str:
    return hashlib.sha256(obj.key.encode("utf-8")).hexdigest()


def _download_with_retry(
    store: ObjectStore,
    obj: ObjectInfo,
    temporary_dir: pathlib.Path,
    retries: int,
) -> pathlib.Path:
    destination = temporary_dir / _temp_name(obj)
    partial = destination.with_suffix(".partial")
    for attempt in range(1, retries + 1):
        with contextlib.suppress(FileNotFoundError):
            partial.unlink()
        try:
            store.download(obj, partial)
            actual_size = partial.stat().st_size
            if actual_size != obj.size:
                raise IOError(
                    f"short read for {obj.relative_path}: expected {obj.size}, got {actual_size}"
                )
            partial.replace(destination)
            return destination
        except Exception as error:
            if attempt == retries:
                raise RuntimeError(
                    f"Failed to download {obj.relative_path} after {retries} attempts"
                ) from error
            delay = min(2 ** (attempt - 1), 30)
            print(
                f"RETRY_DOWNLOAD path={obj.relative_path} attempt={attempt}/{retries} "
                f"delay={delay}s error={error}",
                flush=True,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def build_tar_shard(
    store: ObjectStore,
    objects: list[ObjectInfo],
    tar_path: pathlib.Path,
    temporary_dir: pathlib.Path,
    workers: int,
    retries: int,
) -> None:
    """Download objects concurrently and append them to a deterministic tar."""

    tar_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir.mkdir(parents=True, exist_ok=True)
    window_size = max(workers * 4, 1)
    with tarfile.open(tar_path, mode="w", format=tarfile.PAX_FORMAT) as archive:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            for start in range(0, len(objects), window_size):
                window = objects[start : start + window_size]
                futures = {
                    obj.key: executor.submit(
                        _download_with_retry, store, obj, temporary_dir, retries
                    )
                    for obj in window
                }
                for obj in window:
                    local_path = futures[obj.key].result()
                    info = tarfile.TarInfo(name=obj.relative_path)
                    info.size = obj.size
                    info.mtime = obj.mtime
                    info.mode = 0o644
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    with local_path.open("rb") as stream:
                        archive.addfile(info, stream)
                    local_path.unlink()


def _upload_with_retry(
    store: ObjectStore, source: pathlib.Path, key: str, retries: int
) -> None:
    for attempt in range(1, retries + 1):
        try:
            store.upload(source, key)
            return
        except Exception as error:
            if attempt == retries:
                raise RuntimeError(f"Failed to upload {key} after {retries} attempts") from error
            delay = min(2 ** (attempt - 1), 30)
            print(
                f"RETRY_UPLOAD key={key} attempt={attempt}/{retries} "
                f"delay={delay}s error={error}",
                flush=True,
            )
            time.sleep(delay)


def _persist_json(
    store: ObjectStore,
    work_dir: pathlib.Path,
    key: str,
    payload: Mapping[str, Any],
    retries: int,
) -> None:
    local_path = work_dir / key
    atomic_write_json(local_path, payload)
    _upload_with_retry(store, local_path, key, retries)


def repack(
    store: ObjectStore,
    state: dict[str, Any],
    work_dir: pathlib.Path,
    target_bytes: int,
    workers: int,
    retries: int,
) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    object_dir = work_dir / "objects"
    object_dir.mkdir(parents=True, exist_ok=True)

    for component in COMPONENTS:
        component_state = state["components"][component]
        if component_state["complete"]:
            print(f"REPACK_COMPONENT_REUSED component={component}", flush=True)
            continue

        pending = iter_pending(store.iter_objects(component), component_state["last_key"])
        groups = iter_shard_groups(pending, target_bytes)
        found_any = False
        for objects in groups:
            found_any = True
            shard_index = int(component_state["next_shard_index"])
            shard_name = f"{component}-{shard_index:05d}.tar"
            shard_path = work_dir / shard_name
            print(
                f"REPACK_SHARD_START component={component} shard={shard_name} "
                f"files={len(objects)} bytes={sum(obj.size for obj in objects)}",
                flush=True,
            )
            with contextlib.suppress(FileNotFoundError):
                shard_path.unlink()
            build_tar_shard(
                store,
                objects,
                shard_path,
                object_dir,
                workers=workers,
                retries=retries,
            )
            digest = sha256_file(shard_path)
            shard_record = {
                "name": shard_name,
                "sha256": digest,
                "tar_bytes": shard_path.stat().st_size,
                "uncompressed_bytes": sum(obj.size for obj in objects),
                "file_count": len(objects),
                "first_key": objects[0].key,
                "last_key": objects[-1].key,
            }
            sidecar_path = work_dir / f"{shard_name}.json"
            atomic_write_json(sidecar_path, shard_record)
            _upload_with_retry(store, shard_path, shard_name, retries)
            _upload_with_retry(store, sidecar_path, sidecar_path.name, retries)

            component_state["last_key"] = objects[-1].key
            component_state["next_shard_index"] = shard_index + 1
            component_state["file_count"] += len(objects)
            component_state["uncompressed_bytes"] += shard_record["uncompressed_bytes"]
            component_state["shards"].append(shard_record)
            state["updated_at"] = utc_now()
            _persist_json(store, work_dir, STATE_NAME, state, retries)
            print(
                f"REPACK_SHARD_COMPLETE component={component} shard={shard_name} "
                f"sha256={digest}",
                flush=True,
            )
            shard_path.unlink()
            sidecar_path.unlink()

        component_state["complete"] = True
        state["updated_at"] = utc_now()
        _persist_json(store, work_dir, STATE_NAME, state, retries)
        if not found_any and component_state["file_count"] == 0:
            print(f"REPACK_COMPONENT_EMPTY component={component}", flush=True)
        else:
            print(
                f"REPACK_COMPONENT_COMPLETE component={component} "
                f"files={component_state['file_count']} "
                f"bytes={component_state['uncompressed_bytes']}",
                flush=True,
            )

    state["complete"] = True
    state["updated_at"] = utc_now()
    totals = {
        "file_count": sum(c["file_count"] for c in state["components"].values()),
        "uncompressed_bytes": sum(
            c["uncompressed_bytes"] for c in state["components"].values()
        ),
        "shard_count": sum(len(c["shards"]) for c in state["components"].values()),
    }
    manifest = {
        "format": "prism-uncompressed-tar-shards-v1",
        "complete": True,
        "created_at": state["created_at"],
        "completed_at": state["updated_at"],
        "source_volume_id": state["source_volume_id"],
        "destination_volume_id": state["destination_volume_id"],
        "totals": totals,
        "components": state["components"],
    }
    _persist_json(store, work_dir, STATE_NAME, state, retries)
    _persist_json(store, work_dir, MANIFEST_NAME, manifest, retries)
    print(
        f"PRISM_REPACK_COMPLETE shards={totals['shard_count']} "
        f"files={totals['file_count']} bytes={totals['uncompressed_bytes']}",
        flush=True,
    )
    return manifest


class VesslObjectStore:
    """Direct federated S3 access for a source and destination VESSL volume."""

    def __init__(self, source_volume_id: int, destination_volume_id: int):
        try:
            from boto3.s3.transfer import TransferConfig
            from botocore.exceptions import ClientError
            from vessl.util.volume import VolumeFileTransfer
        except ImportError as error:
            raise RuntimeError(
                "Install the VESSL SDK first: python -m pip install -U vessl"
            ) from error

        self._client_error = ClientError
        self._transfer_config = TransferConfig(
            multipart_threshold=64 * 1024 * 1024,
            multipart_chunksize=64 * 1024 * 1024,
            max_concurrency=8,
            use_threads=True,
        )
        self.source = VolumeFileTransfer(source_volume_id)
        self.destination = VolumeFileTransfer(destination_volume_id)
        if self.destination.volume.is_read_only:
            raise RuntimeError("Destination VESSL volume is read-only")
        self.source_client = self.source._get_s3_client()
        self.destination_client = self.destination._get_s3_client()
        self.source_prefix = self.source.prefix.strip("/")
        self.destination_prefix = self.destination.prefix.strip("/")

    @staticmethod
    def _join(prefix: str, relative: str) -> str:
        if prefix and relative:
            return f"{prefix}/{relative.lstrip('/')}"
        return prefix or relative.lstrip("/")

    def _relative(self, absolute_key: str) -> str:
        prefix = f"{self.source_prefix}/" if self.source_prefix else ""
        if prefix and not absolute_key.startswith(prefix):
            raise RuntimeError(f"Object outside source volume prefix: {absolute_key}")
        return absolute_key[len(prefix) :]

    def _list(self, relative_prefix: str, delimiter: str | None = None) -> Iterator[ObjectInfo]:
        absolute_prefix = self._join(self.source_prefix, relative_prefix)
        if self.source_prefix and not relative_prefix:
            absolute_prefix += "/"
        kwargs: dict[str, Any] = {
            "Bucket": self.source.bucket_name,
            "Prefix": absolute_prefix,
            "MaxKeys": 1000,
        }
        if delimiter is not None:
            kwargs["Delimiter"] = delimiter
        while True:
            response = self.source_client.list_objects_v2(**kwargs)
            for item in response.get("Contents", []):
                if item["Key"].endswith("/") and item["Size"] == 0:
                    continue
                relative = self._relative(item["Key"])
                mtime = item.get("LastModified")
                yield ObjectInfo(
                    key=item["Key"],
                    relative_path=relative,
                    size=int(item["Size"]),
                    etag=str(item.get("ETag", "")).strip('"'),
                    mtime=int(mtime.timestamp()) if mtime else 0,
                )
            if not response.get("IsTruncated"):
                break
            kwargs["ContinuationToken"] = response["NextContinuationToken"]

    def iter_objects(self, component: str) -> Iterator[ObjectInfo]:
        if component == "metadata":
            yield from self._list("", delimiter="/")
            return
        yield from self._list(f"{component}/")

    def download(self, obj: ObjectInfo, destination: pathlib.Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        response = self.source_client.get_object(
            Bucket=self.source.bucket_name, Key=obj.key
        )
        body: BinaryIO = response["Body"]
        try:
            with destination.open("wb") as stream:
                shutil.copyfileobj(body, stream, length=8 * 1024 * 1024)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            body.close()

    def upload(self, source: pathlib.Path, key: str) -> None:
        absolute_key = self._join(self.destination_prefix, key)
        self.destination_client.upload_file(
            str(source),
            self.destination.bucket_name,
            absolute_key,
            Config=self._transfer_config,
        )

    def read_json(self, key: str) -> dict[str, Any] | None:
        absolute_key = self._join(self.destination_prefix, key)
        try:
            response = self.destination_client.get_object(
                Bucket=self.destination.bucket_name, Key=absolute_key
            )
        except self._client_error as error:
            code = str(error.response.get("Error", {}).get("Code", ""))
            if code in {"404", "NoSuchKey", "NotFound"}:
                return None
            raise
        try:
            return json.loads(response["Body"].read().decode("utf-8"))
        finally:
            response["Body"].close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-volume-id", type=int, required=True)
    parser.add_argument("--destination-volume-id", type=int, required=True)
    parser.add_argument(
        "--work-dir", type=pathlib.Path, default=pathlib.Path("/tmp/prism-repack")
    )
    parser.add_argument("--shard-size-gb", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore remote resume state. Existing shard names will be overwritten.",
    )
    args = parser.parse_args(argv)
    if args.shard_size_gb <= 0:
        parser.error("--shard-size-gb must be positive")
    if args.workers <= 0 or args.retries <= 0:
        parser.error("--workers and --retries must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    store = VesslObjectStore(args.source_volume_id, args.destination_volume_id)
    remote_state = None if args.fresh else store.read_json(STATE_NAME)
    state = remote_state or default_state(
        args.source_volume_id, args.destination_volume_id
    )
    validate_state(state, args.source_volume_id, args.destination_volume_id)
    if state.get("complete"):
        manifest = store.read_json(MANIFEST_NAME)
        if manifest and manifest.get("complete"):
            print("PRISM_REPACK_ALREADY_COMPLETE", flush=True)
            return 0
        raise RuntimeError("State is complete but archive manifest is missing")
    repack(
        store,
        state,
        args.work_dir,
        target_bytes=max(1, int(args.shard_size_gb * GIB)),
        workers=args.workers,
        retries=args.retries,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
