import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "tools" / "freeze_prism_manifest.py"


def write_dataset(root: Path, run_splits: list[str]) -> None:
    resources = {
        split: {"shapes": [f"{split}.ply"], "backgrounds": [f"{split}.png"]}
        for split in ("train", "validation", "test")
    }
    (root / "dataset_manifest.json").write_text(
        json.dumps({"run_splits": run_splits, "resources": resources}),
        encoding="utf-8",
    )
    for split in run_splits:
        split_dir = root / split
        split_dir.mkdir()
        (split_dir / "sample.bin").write_bytes(split.encode())


def run_freezer(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_freezes_and_verifies_single_split_shard(tmp_path: Path) -> None:
    write_dataset(tmp_path, ["train"])

    frozen = run_freezer(tmp_path)
    assert frozen.returncode == 0, frozen.stderr
    artifact = json.loads((tmp_path / "artifact_manifest.json").read_text())
    assert artifact["splits"] == ["train"]
    assert artifact["file_count"] == 1

    verified = run_freezer(tmp_path, "--verify")
    assert verified.returncode == 0, verified.stderr
    assert "VERIFIED files=1" in verified.stdout


def test_rejects_missing_selected_split(tmp_path: Path) -> None:
    write_dataset(tmp_path, ["validation"])
    (tmp_path / "validation").rename(tmp_path / "missing-validation")

    result = run_freezer(tmp_path)
    assert result.returncode != 0
    assert "missing split directories: ['validation']" in result.stderr


def test_rejects_unknown_run_split(tmp_path: Path) -> None:
    write_dataset(tmp_path, ["train"])
    payload = json.loads((tmp_path / "dataset_manifest.json").read_text())
    payload["run_splits"] = ["train", "diagnostic"]
    (tmp_path / "dataset_manifest.json").write_text(json.dumps(payload))

    result = run_freezer(tmp_path)
    assert result.returncode != 0
    assert "unknown run_splits" in result.stderr
