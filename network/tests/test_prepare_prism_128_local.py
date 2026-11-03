"""Exercise the native 128px seven-run merge without rendering GPU images."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_prism_128_local.py"
FREEZER = Path(__file__).resolve().parents[2] / "RCDatasetCreation" / "tools" / "freeze_prism_manifest.py"
SPEC = importlib.util.spec_from_file_location("prepare_prism_128_local", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _make_sources(root: Path) -> None:
    resources = {
        "train": {"shapes": [f"shape_{i}" for i in range(5)], "backgrounds": ["train_bg"]},
        "validation": {"shapes": ["validation_shape"], "backgrounds": ["validation_bg"]},
        "test": {"shapes": ["test_shape"], "backgrounds": ["test_bg"]},
    }
    for split, project, index in MODULE.projects(root):
        manifest = {
            "generator_version": "v17_pose_aligned_trace",
            "dataset_seed": 42,
            "validation_seed": 43,
            "split_kind": "main",
            "output_profile": "training",
            "run_splits": [split],
            "shard": {
                "enabled": split == "train", "count": 5 if split == "train" else 1,
                "index": int(index) if split == "train" else 0,
            },
            "configuration": {
                "frames_per_sequence": 8,
                "Camera": {"film": {"width": 128, "height": 128}},
                "project_name": project.name,
            },
            "resources": {
                **resources,
                "train": {
                    **resources["train"],
                    "shapes": [f"shape_{index}"] if split == "train" else resources["train"]["shapes"],
                },
            },
            "counts": {name: {"shapes": len(value["shapes"])} for name, value in resources.items()},
        }
        folder = project / split
        folder.mkdir(parents=True)
        (folder / f"{project.name}_sequence_meta.json").write_text("{}")
        (project / "dataset_manifest.json").write_text(json.dumps(manifest))
        subprocess.run([sys.executable, str(FREEZER), str(project)], check=True, capture_output=True)
        (project / ".generation_complete").touch()


def test_merges_all_frozen_sources_and_preserves_source_files(tmp_path: Path) -> None:
    _make_sources(tmp_path)
    selected = MODULE.wait_for_projects(tmp_path, poll_seconds=0, timeout_seconds=0)
    manifest, sources, provenance = MODULE.inspect_sources(selected, verify_content=True)
    assert manifest["resources"]["train"]["shapes"] == [f"shape_{i}" for i in range(5)]
    assert len(provenance) == 7
    merged = tmp_path / "merged"
    MODULE.materialize(merged, manifest, sources, provenance)
    subprocess.run([sys.executable, str(FREEZER), str(merged), "--verify"], check=True, capture_output=True)
    assert len(list((merged / "train").glob("*_sequence_meta.json"))) == 5
    assert len(list((merged / "validation").glob("*_sequence_meta.json"))) == 1
    assert len(list((merged / "test").glob("*_sequence_meta.json"))) == 1
    MODULE.materialize(merged, manifest, sources, provenance)


def test_rejects_overlapping_train_shard_shapes(tmp_path: Path) -> None:
    _make_sources(tmp_path)
    selected = MODULE.projects(tmp_path)
    second = selected[1][1]
    path = second / "dataset_manifest.json"
    payload = json.loads(path.read_text())
    payload["resources"]["train"]["shapes"] = ["shape_0"]
    path.write_text(json.dumps(payload))
    subprocess.run([sys.executable, str(FREEZER), str(second)], check=True, capture_output=True)
    with pytest.raises(ValueError, match="overlapping train shape shard"):
        MODULE.inspect_sources(selected, verify_content=True)
