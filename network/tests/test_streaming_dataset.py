import hashlib
import json
import shutil
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from refractive_mam2 import streaming_dataset as streaming


def _tar(path: Path, files: dict[str, bytes]) -> dict[str, object]:
    source = path.parent / (path.stem + "-source")
    for relative, payload in files.items():
        member = source / relative
        member.parent.mkdir(parents=True, exist_ok=True)
        member.write_bytes(payload)
    with tarfile.open(path, "w") as archive:
        for relative in files:
            archive.add(source / relative, arcname=relative)
    shutil.rmtree(source)
    return {
        "name": path.name,
        "tar_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "file_count": len(files),
    }


class _FakeStore:
    def __init__(self, root: Path, manifest: dict, fail_first_download: bool = False):
        self.root = root
        self.manifest = manifest
        self.fail_first_download = fail_first_download
        self.downloads = 0
        self.refreshes = 0

    def read_json(self, name):
        assert name == "archive_manifest.json"
        return self.manifest

    def iter_objects(self, component):
        assert component == "metadata"
        for shard in self.manifest["components"]["train"]["shards"]:
            yield SimpleNamespace(relative_path=shard["name"])

    def _refresh_source(self):
        self.refreshes += 1

    def download(self, obj, destination):
        self.downloads += 1
        if self.fail_first_download and self.downloads == 1:
            raise OSError("temporary credential failure")
        shutil.copy2(self.root / obj.relative_path, destination)


class _FakeRCTransDataset:
    def __init__(self, root, **_kwargs):
        root = Path(root)
        metadata_paths = sorted(root.rglob("*_sequence_meta.json"))
        if not metadata_paths:
            raise FileNotFoundError(f"No *_sequence_meta.json found below {root}")
        self.records = [
            SimpleNamespace(
                metadata_path=path,
                prefix=Path(str(path)[: -len("_sequence_meta.json")]),
            )
            for path in metadata_paths
        ]

    def set_epoch(self, _epoch):
        return None

    def __getitem__(self, index):
        record = self.records[index]
        metadata = json.loads(record.metadata_path.read_text())
        return {
            "sequence_id": record.prefix.name,
            "paired_background_group_id": metadata["paired_background_group_id"],
            "metadata": metadata,
            "tensors": {},
        }


def test_estimated_count_expands_renderer_shape_shard():
    manifest = {
        "configuration": {
            "backgrounds_per_shape": 8,
            "sequence_num_per_background": 4,
        },
        "resources": {"train": {"shapes": ["a", "b"]}},
        "shard": {"enabled": True, "count": 5, "splits": ["train"]},
    }

    assert streaming._estimated_sequence_count(manifest, "train") == 320


def test_cycles_across_tar_boundary_retries_and_cleans_cache(tmp_path, monkeypatch):
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    first = _tar(
        archive_root / "train-00000.tar",
        {"train/sample_payload.bin": b"payload"},
    )
    metadata = {
        "shape_path": "shape.ply",
        "background_path": "background.png",
        "paired_background_group_id": "pair-0",
    }
    second = _tar(
        archive_root / "train-00001.tar",
        {"train/sample_sequence_meta.json": json.dumps(metadata).encode()},
    )
    archive_manifest = {
        "complete": True,
        "components": {
            "train": {"shards": [first, second], "file_count": 2}
        },
    }
    store = _FakeStore(archive_root, archive_manifest, fail_first_download=True)
    monkeypatch.setattr(streaming, "RCTransPRISMDataset", _FakeRCTransDataset)
    monkeypatch.setattr(streaming.VesslShardCyclingDataset, "_store", lambda _self: store)

    metadata_root = tmp_path / "metadata"
    metadata_root.mkdir()
    dataset_manifest = metadata_root / "dataset_manifest.json"
    dataset_manifest.write_text(
        json.dumps(
            {
                "materialization": {"sequence_counts": {"train": 1}},
                "resources": {"train": {"shapes": [], "backgrounds": []}},
            }
        )
    )
    cache_root = tmp_path / "cache"
    dataset = streaming.VesslShardCyclingDataset(
        dataset_manifest=dataset_manifest,
        split="train",
        storage_name="storage",
        archive_volume="archive",
        cache_root=cache_root,
        clip_length=4,
        frame_stride=1,
        strict_contract=True,
        shuffle_buffer=1,
        download_retries=2,
    )

    samples = list(dataset)

    assert [sample["sequence_id"] for sample in samples] == ["sample"]
    assert store.downloads == 3
    assert store.refreshes == 3
    assert not (cache_root / "train").exists()


def test_paired_stream_selects_one_batch_from_complete_background_group(
    tmp_path, monkeypatch
):
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    files = {}
    for index in range(4):
        metadata = {
            "shape_path": "shape.ply",
            "background_path": f"background-{index}.png",
            "paired_background_group_id": "pair-0",
        }
        files[f"train/sample-bg{index}_sequence_meta.json"] = json.dumps(
            metadata
        ).encode()
    shard = _tar(archive_root / "train-00000.tar", files)
    archive_manifest = {
        "complete": True,
        "components": {
            "train": {"shards": [shard], "file_count": len(files)}
        },
    }
    store = _FakeStore(archive_root, archive_manifest)
    monkeypatch.setattr(streaming, "RCTransPRISMDataset", _FakeRCTransDataset)
    monkeypatch.setattr(streaming.VesslShardCyclingDataset, "_store", lambda _self: store)

    metadata_root = tmp_path / "metadata"
    metadata_root.mkdir()
    dataset_manifest = metadata_root / "dataset_manifest.json"
    dataset_manifest.write_text(
        json.dumps(
            {
                "materialization": {"sequence_counts": {"train": 4}},
                "configuration": {"backgrounds_per_shape": 4},
                "resources": {"train": {"shapes": [], "backgrounds": []}},
            }
        )
    )
    dataset = streaming.VesslShardCyclingDataset(
        dataset_manifest=dataset_manifest,
        split="train",
        storage_name="storage",
        archive_volume="archive",
        cache_root=tmp_path / "cache",
        clip_length=4,
        frame_stride=1,
        strict_contract=True,
        paired_backgrounds=True,
        pair_size=2,
        shuffle_buffer=2,
    )

    samples = list(dataset)

    assert len(dataset) == 2
    assert len(samples) == 2
    assert len({sample["metadata"]["background_path"] for sample in samples}) == 2


def test_streaming_extractor_rejects_path_traversal(tmp_path):
    archive_path = tmp_path / "unsafe.tar"
    with tarfile.open(archive_path, "w") as archive:
        info = tarfile.TarInfo("../escape")
        info.size = 0
        archive.addfile(info)

    with pytest.raises(RuntimeError, match="Path traversal"):
        streaming._safe_extract(archive_path, tmp_path / "output")


def test_discards_only_sequence_with_incomplete_frame_count(tmp_path, capsys):
    split_root = tmp_path / "train"
    split_root.mkdir()
    good = split_root / "good"
    bad = split_root / "bad"
    Path(str(good) + "_sequence_meta.json").write_text(
        json.dumps({"frame_count": 1})
    )
    Path(str(good) + "_background.exr").touch()
    good_frame = Path(str(good) + "_frame0000")
    for suffix in streaming.RCTransPRISMDataset.REQUIRED_SUFFIXES:
        Path(str(good_frame) + suffix).touch()
    Path(str(bad) + "_sequence_meta.json").write_text(
        json.dumps({"frame_count": 2})
    )
    Path(str(bad) + "_frame0000_I.exr").touch()
    Path(str(bad) + "_background.exr").touch()

    discarded = streaming._discard_structurally_invalid_sequences(split_root)

    assert discarded == 1
    assert Path(str(good) + "_sequence_meta.json").is_file()
    assert Path(str(good) + "_frame0000_I.exr").is_file()
    assert not Path(str(bad) + "_sequence_meta.json").exists()
    assert not Path(str(bad) + "_frame0000_I.exr").exists()
    assert not Path(str(bad) + "_background.exr").exists()
    assert "PRISM_STREAM_SEQUENCE_SKIPPED" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("missing_suffix", "reason"),
    [
        ("_background.exr", "missing_background"),
        ("_alpha.npy", "missing_frame_outputs"),
    ],
)
def test_discards_sequence_with_missing_required_output(
    tmp_path, capsys, missing_suffix, reason
):
    split_root = tmp_path / "validation"
    split_root.mkdir()
    prefix = split_root / "broken"
    Path(str(prefix) + "_sequence_meta.json").write_text(
        json.dumps({"frame_count": 1})
    )
    Path(str(prefix) + "_background.exr").touch()
    frame = Path(str(prefix) + "_frame0000")
    for suffix in streaming.RCTransPRISMDataset.REQUIRED_SUFFIXES:
        Path(str(frame) + suffix).touch()
    target = (
        Path(str(prefix) + missing_suffix)
        if missing_suffix == "_background.exr"
        else Path(str(frame) + missing_suffix)
    )
    target.unlink()

    discarded = streaming._discard_structurally_invalid_sequences(split_root)

    assert discarded == 1
    assert not Path(str(prefix) + "_sequence_meta.json").exists()
    output = capsys.readouterr().out
    assert f"reason={reason}" in output
    assert missing_suffix in output or target.name in output


def test_paired_stream_keeps_pair_from_partial_full_split(tmp_path, monkeypatch, capsys):
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    files = {}
    for index in range(3):
        metadata = {
            "shape_path": "shape.ply",
            "background_path": f"background-{index}.png",
            "paired_background_group_id": "partial-pair",
        }
        files[f"train/sample-bg{index}_sequence_meta.json"] = json.dumps(
            metadata
        ).encode()
    shard = _tar(archive_root / "train-00000.tar", files)
    archive_manifest = {
        "complete": True,
        "components": {"train": {"shards": [shard], "file_count": len(files)}},
    }
    store = _FakeStore(archive_root, archive_manifest)
    monkeypatch.setattr(streaming, "RCTransPRISMDataset", _FakeRCTransDataset)
    monkeypatch.setattr(streaming.VesslShardCyclingDataset, "_store", lambda _self: store)

    metadata_root = tmp_path / "metadata"
    metadata_root.mkdir()
    dataset_manifest = metadata_root / "dataset_manifest.json"
    dataset_manifest.write_text(
        json.dumps(
            {
                "materialization": {"sequence_counts": {"train": 4}},
                "configuration": {"backgrounds_per_shape": 4},
                "resources": {"train": {"shapes": [], "backgrounds": []}},
            }
        )
    )
    dataset = streaming.VesslShardCyclingDataset(
        dataset_manifest=dataset_manifest,
        split="train",
        storage_name="storage",
        archive_volume="archive",
        cache_root=tmp_path / "cache",
        clip_length=4,
        frame_stride=1,
        strict_contract=True,
        paired_backgrounds=True,
        pair_size=2,
        shuffle_buffer=2,
    )

    samples = list(dataset)

    assert len(samples) == 2
    assert "PRISM_STREAM_PAIRED_GROUP_PARTIAL" in capsys.readouterr().out
