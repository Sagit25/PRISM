import importlib.util
from pathlib import Path

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
