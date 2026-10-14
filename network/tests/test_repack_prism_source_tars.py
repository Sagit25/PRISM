import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from repack_prism_source_tars import Source, parse_source, repack, sha256_file


def _source(root: Path, split: str, index: int) -> Source:
    stem = f"source_{split}_{index}"
    folder = root / stem
    folder.mkdir()
    manifest = {
        "configuration": {"Camera": {"film": {"width": 64, "height": 64}}},
        "shard": {"enabled": split == "train", "count": 5 if split == "train" else 1},
    }
    (folder / f"{stem}.dataset_manifest.json").write_text(json.dumps(manifest))
    (folder / f"{stem}.generation_complete").write_text(f"split={split}\n")
    archive_path = folder / f"{stem}.tar"
    metadata = json.dumps(
        {
            "generator_version": "v17_pose_aligned_trace",
            "shape_path": f"shape_{split}_{index}",
            "background_path": f"background_{split}_{index}",
        }
    ).encode()
    with tarfile.open(archive_path, "w") as archive:
        for file_index in range(100):
            suffix = "sequence_meta.json" if file_index == 0 else f"frame{file_index:04d}.exr"
            data = metadata if file_index == 0 else b"tiny frame"
            name = f"{stem}/{split}/{stem}_seq0000_{suffix}"
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    (folder / f"{stem}.tar.sha256").write_text(
        f"{sha256_file(archive_path)}  {archive_path.name}\n"
    )
    return Source(split, stem, folder, 1)


def test_repack_builds_complete_split_archive_without_changing_source(tmp_path: Path) -> None:
    sources = [
        *[_source(tmp_path, "train", index) for index in range(5)],
        _source(tmp_path, "validation", 0),
        _source(tmp_path, "test", 0),
    ]
    original_hashes = {
        source.stem: sha256_file(source.directory / f"{source.stem}.tar")
        for source in sources
    }
    output = tmp_path / "packed"

    manifest = repack(sources, output)

    assert manifest["complete"] is True
    assert manifest["totals"]["sequence_count"] == 7
    assert manifest["totals"]["shard_count"] == 8
    assert manifest["totals"]["file_count"] == 701
    combined = json.loads((output / "dataset_manifest.json").read_text())
    assert combined["materialization"]["sequence_counts"] == {
        "train": 5,
        "validation": 1,
        "test": 1,
    }
    for source in sources:
        assert sha256_file(source.directory / f"{source.stem}.tar") == original_hashes[
            source.stem
        ]
    for component in ("metadata", "train", "validation", "test"):
        for shard in manifest["components"][component]["shards"]:
            output_tar = output / shard["name"]
            assert sha256_file(output_tar) == shard["sha256"]
            with tarfile.open(output_tar, "r:") as archive:
                assert all(
                    member.name == "dataset_manifest.json"
                    or member.name.startswith(component + "/")
                    for member in archive
                )


def test_repack_rejects_source_checksum_mismatch(tmp_path: Path) -> None:
    sources = [
        *[_source(tmp_path, "train", index) for index in range(5)],
        _source(tmp_path, "validation", 0),
        _source(tmp_path, "test", 0),
    ]
    checksum = sources[0].directory / f"{sources[0].stem}.tar.sha256"
    checksum.write_text("0" * 64 + "  invalid.tar\n")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        repack(sources, tmp_path / "packed")


def test_parse_source_validates_format() -> None:
    with pytest.raises(Exception, match="--source must be"):
        parse_source("train:incomplete")
