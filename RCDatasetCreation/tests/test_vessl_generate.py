import importlib.util
from pathlib import Path
import shutil

import pytest


SCRIPT = Path(__file__).parents[1] / "tools" / "vessl_generate.py"
SPEC = importlib.util.spec_from_file_location("vessl_generate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_train_arguments_and_project_name() -> None:
    args = MODULE.parse_args(["train", "2", "5", "--run-version", "v3"])

    assert args.split == "train"
    assert args.shard_index == 2
    assert args.shard_count == 5
    assert MODULE.project_name("train", "v3", 2, 5) == (
        "prism_main_v3_train_s02of05"
    )


@pytest.mark.parametrize("split", ["validation", "test"])
def test_unsharded_project_names(split: str) -> None:
    args = MODULE.parse_args([split, "--run-version", "v3"])

    assert args.shard_index is None
    assert args.shard_count is None
    assert MODULE.project_name(split, "v3") == f"prism_main_v3_{split}"


@pytest.mark.parametrize(
    "arguments",
    [
        ["train"],
        ["train", "5", "5"],
        ["train", "0", "0"],
        ["validation", "0", "5"],
    ],
)
def test_rejects_invalid_shard_arguments(arguments: list[str]) -> None:
    with pytest.raises(SystemExit):
        MODULE.parse_args(arguments)


def test_render_command_is_argument_safe(tmp_path: Path) -> None:
    args = MODULE.parse_args(
        ["train", "0", "5", "--output-root", str(tmp_path)]
    )
    command = MODULE.render_command(args, "prism_main_v3_train_s00of05")

    assert command[-4:] == ["--shard-count", "5", "--shard-index", "0"]
    assert "--splits" in command
    assert command[command.index("--splits") + 1] == "train"


def test_failure_marker_preserves_traceback(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(MODULE, "install_requirements", lambda: None)

    def fail_to_prepare_assets(_asset_tar: Path) -> Path:
        raise RuntimeError("synthetic asset failure")

    monkeypatch.setattr(MODULE, "prepare_assets", fail_to_prepare_assets)

    with pytest.raises(RuntimeError, match="synthetic asset failure"):
        MODULE.main(
            [
                "validation",
                "--output-root",
                str(tmp_path),
                "--run-version",
                "marker-test",
            ]
        )

    dataset_root = tmp_path / "prism_main_marker-test_validation"
    marker = dataset_root / ".generation_failed"
    assert marker.is_file()
    assert "RuntimeError: synthetic asset failure" in marker.read_text()
    assert not (dataset_root / ".generation_complete").exists()


def test_execute_validates_assets_before_render(tmp_path: Path, monkeypatch) -> None:
    args = MODULE.parse_args(
        ["validation", "--output-root", str(tmp_path), "--run-version", "order"]
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(MODULE, "install_requirements", lambda: None)
    monkeypatch.setattr(MODULE, "prepare_assets", lambda _path: tmp_path)
    monkeypatch.setattr(MODULE, "run", lambda command, **_kwargs: calls.append(command))

    MODULE.execute(args)

    assert calls[0][-2:] == ["tools/prepare_prism_assets.py", "validate"]
    assert calls[1][1] == "render_dataset.py"


def test_restore_checkpoint_recovers_existing_frames(
    tmp_path: Path, monkeypatch
) -> None:
    output_root = tmp_path / "scratch"
    checkpoint_root = tmp_path / "persistent"
    project = "prism_main_v3_validation"
    saved = checkpoint_root / project / "validation" / "checkpoint.txt"
    saved.parent.mkdir(parents=True)
    saved.write_text("frame-0001\n", encoding="utf-8")

    def local_sync(source: Path, destination: Path) -> None:
        shutil.copytree(source, destination, dirs_exist_ok=True)

    monkeypatch.setattr(MODULE, "sync_tree", local_sync)
    MODULE.restore_checkpoint(checkpoint_root, output_root, project)

    restored = output_root / project / "validation" / "checkpoint.txt"
    assert restored.read_text(encoding="utf-8") == "frame-0001\n"


def test_periodic_checkpoint_always_performs_initial_and_final_sync(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_root = tmp_path / "scratch" / "project"
    checkpoint_root = tmp_path / "persistent"
    dataset_root.mkdir(parents=True)
    calls: list[tuple[Path, Path]] = []

    monkeypatch.setattr(
        MODULE,
        "sync_tree",
        lambda source, destination: calls.append((source, destination)),
    )
    with MODULE.periodic_checkpoint(
        dataset_root, checkpoint_root, None, 60.0
    ):
        (dataset_root / "frame.txt").write_text("done", encoding="utf-8")

    assert calls == [
        (dataset_root, checkpoint_root.resolve() / "project"),
        (dataset_root, checkpoint_root.resolve() / "project"),
    ]


def test_incremental_uri_checkpoint_stages_only_changed_files(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_root = tmp_path / "project"
    dataset_root.mkdir()
    first = dataset_root / "first.txt"
    first.write_text("one", encoding="utf-8")
    uploaded: dict[str, tuple[int, int]] = {}
    staged: list[list[str]] = []

    def fake_copy(source: str, destination: str, **_kwargs) -> bool:
        source_root = Path(source)
        staged.append(
            sorted(
                str(path.relative_to(source_root))
                for path in source_root.rglob("*")
                if path.is_file()
            )
        )
        assert destination == "volume://vessl-storage/output"
        return True

    monkeypatch.setattr(MODULE, "_vessl_copy", fake_copy)
    monkeypatch.setattr(MODULE.time, "time_ns", lambda: 10**20)
    MODULE.sync_tree_to_uri(
        dataset_root,
        "volume://vessl-storage/output",
        uploaded,
    )
    second = dataset_root / "second.txt"
    second.write_text("two", encoding="utf-8")
    MODULE.sync_tree_to_uri(
        dataset_root,
        "volume://vessl-storage/output",
        uploaded,
    )

    assert staged == [["first.txt"], ["second.txt"]]
