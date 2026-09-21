"""Bounded-disk streaming of PRISM tar shards from VESSL storage.

The archive packer preserves the source object-key order but may split a
sequence at a tar boundary.  This dataset therefore extracts one shard at a
time into a private cache, keeps only the unfinished tail sequence, eagerly
loads every completed sequence into tensors, and immediately removes its
files.  At no point does the worker retain the fully materialized split.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import inspect
import json
import random
import shutil
import sys
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

from torch.utils.data import IterableDataset, get_worker_info

from .dataset import RCTransPRISMDataset


def _load_archive_helpers():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    name = "prism_repack_vessl_dataset"
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(
        name, scripts / "repack_vessl_dataset.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the PRISM VESSL archive helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _safe_extract(archive_path: Path, output_root: Path) -> None:
    root = output_root.resolve()
    with tarfile.open(archive_path, mode="r:") as archive:
        for member in archive.getmembers():
            if member.issym() or member.islnk() or member.isdev():
                raise RuntimeError(
                    f"Unsafe tar member type in {archive_path}: {member.name}"
                )
            destination = (root / member.name).resolve()
            try:
                destination.relative_to(root)
            except ValueError as error:
                raise RuntimeError(
                    f"Path traversal in {archive_path}: {member.name}"
                ) from error
        kwargs = {}
        if "filter" in inspect.signature(archive.extractall).parameters:
            kwargs["filter"] = "fully_trusted"
        archive.extractall(output_root, **kwargs)


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _estimated_sequence_count(manifest: dict[str, Any], split: str) -> int:
    """Estimate split length without loading the multi-gigabyte file inventory.

    A fully materialized manifest has an exact sequence count.  Older packed
    datasets instead retain one renderer worker's resource manifest.  In that
    case the shape-shard multiplier recovers the intended full-split scale;
    the iterator still consumes every shard and reports the observed count, so
    this estimate affects only DataLoader length and schedule calibration.
    """

    materialized = (
        manifest.get("materialization", {}).get("sequence_counts", {}).get(split)
    )
    if materialized is not None:
        return int(materialized)
    configuration = manifest.get("configuration", {})
    shapes = manifest.get("resources", {}).get(split, {}).get("shapes", ())
    if not shapes:
        shapes = range(int(manifest.get("counts", {}).get(split, {}).get("shapes", 0)))
    backgrounds_per_shape = int(configuration.get("backgrounds_per_shape", 0))
    sequences_per_background = int(
        configuration.get("sequence_num_per_background", 0)
    )
    shape_count = len(shapes)
    shard = manifest.get("shard", {})
    if (
        shard.get("enabled")
        and split in shard.get("splits", ())
        and int(shard.get("count", 1)) > 1
    ):
        shape_count *= int(shard["count"])
    count = shape_count * backgrounds_per_shape * sequences_per_background
    if count < 1:
        raise ValueError(
            "dataset_manifest.json cannot determine the number of sequences for "
            f"{split!r}; add materialization.sequence_counts before streaming"
        )
    return count


def _remove_sequence_files(prefix: Path) -> None:
    for path in prefix.parent.glob(prefix.name + "_*"):
        if path.is_file():
            with contextlib.suppress(FileNotFoundError):
                path.unlink()


def _discard_structurally_invalid_sequences(split_root: Path) -> int:
    """Drop completed metadata records whose rendered files are incomplete.

    Sequence metadata is the final lexicographic member emitted for a rendered
    sequence.  Once it is present, a frame-count mismatch cannot be repaired by
    a later tar shard.  Skipping only that record prevents one interrupted
    renderer output from terminating a multi-day training run while the normal
    dataset contract remains strict for every retained sequence.
    """

    discarded = 0
    for metadata_path in sorted(split_root.rglob("*_sequence_meta.json")):
        prefix = Path(str(metadata_path)[: -len("_sequence_meta.json")])
        with metadata_path.open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        expected = metadata.get("frame_count")
        if expected is None:
            continue
        frame_count = len(
            list(prefix.parent.glob(prefix.name + "_frame*_I.exr"))
        )
        if frame_count == int(expected):
            continue
        print(
            "PRISM_STREAM_SEQUENCE_SKIPPED "
            f"split={split_root.name} reason=frame_count_mismatch "
            f"expected={int(expected)} found={frame_count} "
            f"metadata={metadata_path.name}",
            flush=True,
        )
        _remove_sequence_files(prefix)
        discarded += 1
    return discarded


class VesslShardCyclingDataset(IterableDataset[dict[str, Any]]):
    """Yield one PRISM split while retaining at most one tar shard on disk."""

    def __init__(
        self,
        *,
        dataset_manifest: str | Path,
        split: str,
        storage_name: str,
        archive_volume: str,
        cache_root: str | Path,
        clip_length: int | None,
        frame_stride: int,
        strict_contract: bool,
        random_temporal_crop: bool = False,
        random_horizontal_flip: bool = False,
        augmentation_seed: int = 0,
        paired_backgrounds: bool = False,
        pair_size: int = 2,
        shuffle_buffer: int = 16,
        max_shards: int | None = None,
        download_retries: int = 5,
    ) -> None:
        super().__init__()
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"unsupported PRISM split: {split}")
        if paired_backgrounds and pair_size < 2:
            raise ValueError("paired shard streaming requires pair_size >= 2")
        if shuffle_buffer < 1:
            raise ValueError("shuffle_buffer must be positive")
        if download_retries < 1:
            raise ValueError("download_retries must be positive")
        self.dataset_manifest = Path(dataset_manifest)
        self.split = split
        self.storage_name = storage_name
        self.archive_volume = archive_volume
        self.cache_root = Path(cache_root) / split
        self.clip_length = clip_length
        self.frame_stride = frame_stride
        self.strict_contract = strict_contract
        self.random_temporal_crop = random_temporal_crop
        self.random_horizontal_flip = random_horizontal_flip
        self.augmentation_seed = augmentation_seed
        self.paired_backgrounds = paired_backgrounds
        self.pair_size = pair_size
        self.shuffle_buffer = shuffle_buffer
        self.max_shards = max_shards
        self.download_retries = download_retries
        self.epoch = 0

        manifest = json.loads(self.dataset_manifest.read_text(encoding="utf-8"))
        full_sequence_count = _estimated_sequence_count(manifest, split)
        self.backgrounds_per_group = int(
            manifest.get("configuration", {}).get("backgrounds_per_shape", 1)
        )
        if paired_backgrounds and self.backgrounds_per_group < pair_size:
            raise ValueError(
                "paired shard streaming requires at least pair_size backgrounds "
                "per physical-operator group"
            )
        if paired_backgrounds:
            # Match PairedBackgroundBatchSampler: one batch containing
            # pair_size distinct backgrounds per physical-operator group.
            full_sequence_count = (
                full_sequence_count // self.backgrounds_per_group * pair_size
            )
        archive = self._store().read_json("archive_manifest.json")
        if archive is None or not archive.get("complete"):
            raise RuntimeError("remote PRISM archive manifest is missing or incomplete")
        shards = archive["components"][split]["shards"]
        if max_shards is not None:
            shards = shards[:max_shards]
        self.shards = tuple(shards)
        if not self.shards:
            raise RuntimeError(f"remote PRISM archive has no {split} shards")
        if max_shards is None:
            self.sequence_count = full_sequence_count
        else:
            selected_files = sum(int(shard.get("file_count", 0)) for shard in self.shards)
            total_files = int(archive["components"][split].get("file_count", 0))
            self.sequence_count = max(
                1,
                round(full_sequence_count * selected_files / max(total_files, 1)),
            )

    def _store(self):
        helper = _load_archive_helpers()
        return helper.VesslObjectStore(
            storage_name=self.storage_name,
            source_volume_name=self.archive_volume,
            destination_volume_name=self.archive_volume,
        )

    def __len__(self) -> int:
        return self.sequence_count

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _download(self, store: Any, objects: dict[str, Any], shard: dict[str, Any]) -> Path:
        name = shard["name"]
        destination = self.cache_root / name
        partial = destination.with_suffix(".tar.partial")
        for attempt in range(1, self.download_retries + 1):
            with contextlib.suppress(FileNotFoundError):
                partial.unlink()
            try:
                store._refresh_source()
                store.download(objects[name], partial)
                if partial.stat().st_size != int(shard["tar_bytes"]):
                    raise IOError(f"short read while streaming {name}")
                actual = _sha256(partial)
                if actual != shard["sha256"]:
                    raise IOError(f"SHA-256 mismatch while streaming {name}")
                partial.replace(destination)
                return destination
            except Exception as error:
                if attempt == self.download_retries:
                    raise RuntimeError(
                        f"failed to stream {name} after {self.download_retries} attempts"
                    ) from error
                print(
                    f"PRISM_STREAM_SHARD_RETRY split={self.split} name={name} "
                    f"attempt={attempt}/{self.download_retries} error={error}",
                    flush=True,
                )
        raise AssertionError("unreachable")

    def _ready_samples(self, processed: set[str]) -> list[tuple[dict[str, Any], Path]]:
        split_root = self.cache_root / self.split
        if not split_root.exists():
            return []
        _discard_structurally_invalid_sequences(split_root)
        metadata_paths = sorted(split_root.rglob("*_sequence_meta.json"))
        if metadata_paths:
            # Distributed renderer workers may each have emitted a shard-local
            # resource list.  Extend the cache-local manifest from the actual
            # sequence metadata visible in this window, while leaving the
            # immutable source manifest untouched.
            manifest_path = self.cache_root / "dataset_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            resources = manifest.setdefault("resources", {}).setdefault(
                self.split, {"shapes": [], "backgrounds": []}
            )
            shapes = set(resources.get("shapes", ()))
            backgrounds = set(resources.get("backgrounds", ()))
            for metadata_path in metadata_paths:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata.get("shape_path"):
                    shapes.add(str(metadata["shape_path"]))
                if metadata.get("background_path"):
                    backgrounds.add(str(metadata["background_path"]))
            resources["shapes"] = sorted(shapes)
            resources["backgrounds"] = sorted(backgrounds)
            temporary = manifest_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(manifest_path)
        try:
            dataset = RCTransPRISMDataset(
                split_root,
                clip_length=self.clip_length,
                frame_stride=self.frame_stride,
                strict_contract=self.strict_contract,
                random_temporal_crop=self.random_temporal_crop,
                random_horizontal_flip=self.random_horizontal_flip,
                augmentation_seed=self.augmentation_seed,
            )
        except FileNotFoundError as error:
            if "No *_sequence_meta.json" in str(error):
                return []
            raise
        dataset.set_epoch(self.epoch)
        ready: list[tuple[dict[str, Any], Path]] = []
        for index, record in enumerate(dataset.records):
            identity = str(record.metadata_path)
            if identity in processed:
                continue
            try:
                sample = dataset[index]
            except FileNotFoundError:
                # The only legitimate incomplete record is the tail cut by a
                # tar boundary.  It remains in the cache for the next shard.
                continue
            processed.add(identity)
            ready.append((sample, record.prefix))
        return ready

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if get_worker_info() is not None:
            raise RuntimeError(
                "VESSL shard cycling requires DataLoader num_workers=0 to avoid "
                "duplicate remote downloads"
            )
        shutil.rmtree(self.cache_root, ignore_errors=True)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.dataset_manifest, self.cache_root / "dataset_manifest.json")
        store = self._store()
        objects = {
            obj.relative_path: obj
            for obj in store.iter_objects("metadata")
            if obj.relative_path in {shard["name"] for shard in self.shards}
        }
        missing = [shard["name"] for shard in self.shards if shard["name"] not in objects]
        if missing:
            raise RuntimeError(f"missing remote PRISM shards: {missing[:5]}")

        processed: set[str] = set()
        rng = random.Random(self.augmentation_seed + self.epoch)
        shuffle: list[dict[str, Any]] = []
        pair_shuffle: list[list[dict[str, Any]]] = []
        pairs: dict[str, list[tuple[dict[str, Any], Path]]] = defaultdict(list)
        emitted = 0
        try:
            for shard_index, shard in enumerate(self.shards):
                archive_path = self._download(store, objects, shard)
                print(
                    f"PRISM_STREAM_SHARD_START split={self.split} "
                    f"index={shard_index + 1}/{len(self.shards)} name={shard['name']}",
                    flush=True,
                )
                _safe_extract(archive_path, self.cache_root)
                archive_path.unlink()
                for sample, prefix in self._ready_samples(processed):
                    _remove_sequence_files(prefix)
                    if self.paired_backgrounds:
                        group = sample["paired_background_group_id"]
                        bucket = pairs[group]
                        backgrounds = {
                            str(item[0]["metadata"].get("background_path"))
                            for item in bucket
                        }
                        background = str(sample["metadata"].get("background_path"))
                        if background not in backgrounds:
                            bucket.append((sample, prefix))
                        if len(bucket) >= self.backgrounds_per_group:
                            candidates = list(bucket)
                            del pairs[group]
                            if self.split == "train":
                                rng.shuffle(candidates)
                            pair = [
                                entry[0] for entry in candidates[: self.pair_size]
                            ]
                            pair_shuffle.append(pair)
                    else:
                        shuffle.append(sample)
                    if self.paired_backgrounds:
                        pair_limit = max(1, self.shuffle_buffer // self.pair_size)
                        while len(pair_shuffle) >= pair_limit:
                            index = (
                                rng.randrange(len(pair_shuffle))
                                if self.split == "train"
                                else 0
                            )
                            for paired_sample in pair_shuffle.pop(index):
                                emitted += 1
                                yield paired_sample
                    else:
                        while len(shuffle) >= self.shuffle_buffer:
                            index = (
                                rng.randrange(len(shuffle))
                                if self.split == "train"
                                else 0
                            )
                            emitted += 1
                            yield shuffle.pop(index)
                print(
                    f"PRISM_STREAM_SHARD_COMPLETE split={self.split} "
                    f"name={shard['name']} emitted={emitted}",
                    flush=True,
                )
            if self.paired_backgrounds and pairs:
                if self.max_shards is None:
                    raise RuntimeError(
                        f"{len(pairs)} paired-background groups were incomplete after "
                        f"streaming the {self.split} split"
                    )
                # A smoke run may intentionally stop in the middle of a group.
                # Keep only partial groups that already contain a valid pair.
                for bucket in pairs.values():
                    if len(bucket) < self.pair_size:
                        continue
                    candidates = list(bucket)
                    if self.split == "train":
                        rng.shuffle(candidates)
                    pair_shuffle.append(
                        [entry[0] for entry in candidates[: self.pair_size]]
                    )
            while pair_shuffle:
                index = rng.randrange(len(pair_shuffle)) if self.split == "train" else 0
                for paired_sample in pair_shuffle.pop(index):
                    emitted += 1
                    yield paired_sample
            while shuffle:
                index = rng.randrange(len(shuffle)) if self.split == "train" else 0
                emitted += 1
                yield shuffle.pop(index)
            if self.max_shards is None and emitted != self.sequence_count:
                print(
                    f"PRISM_STREAM_COUNT_ADJUSTED split={self.split} "
                    f"estimated={self.sequence_count} observed={emitted}",
                    flush=True,
                )
        finally:
            shutil.rmtree(self.cache_root, ignore_errors=True)
