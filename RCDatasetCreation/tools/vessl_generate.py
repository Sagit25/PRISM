#!/usr/bin/env python3
"""Run one reproducible PRISM dataset-generation job on VESSL.

The VESSL command only needs to fetch an exact Git commit and invoke this file.
Keeping the generation workflow here prevents the seven cloud runs from
silently drifting apart or depending on Bash-specific inline syntax.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile


DEFAULT_ASSET_TAR = Path("/input/assets/prism-research-assets-v1.tar")
DEFAULT_OUTPUT_ROOT = Path("/root/workspace/persistent_export")
DEFAULT_RUN_VERSION = "v3"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare assets, render, validate, and freeze one PRISM split"
    )
    parser.add_argument("split", choices=("train", "validation", "test"))
    parser.add_argument("shard_index", type=int, nargs="?")
    parser.add_argument("shard_count", type=int, nargs="?")
    parser.add_argument(
        "--asset-tar",
        type=Path,
        default=Path(os.environ.get("PRISM_ASSET_TAR", DEFAULT_ASSET_TAR)),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.environ.get("PRISM_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)),
    )
    parser.add_argument(
        "--run-version",
        default=os.environ.get("PRISM_RUN_VERSION", DEFAULT_RUN_VERSION),
    )
    args = parser.parse_args(argv)

    if args.split == "train":
        if args.shard_index is None or args.shard_count is None:
            parser.error("train requires SHARD_INDEX and SHARD_COUNT")
        if args.shard_count < 1:
            parser.error("SHARD_COUNT must be at least 1")
        if not 0 <= args.shard_index < args.shard_count:
            parser.error("SHARD_INDEX must be in [0, SHARD_COUNT)")
    elif args.shard_index is not None or args.shard_count is not None:
        parser.error("validation and test do not accept shard arguments")
    return args


def project_name(
    split: str,
    run_version: str,
    shard_index: int | None = None,
    shard_count: int | None = None,
) -> str:
    if split == "train":
        if shard_index is None or shard_count is None:
            raise ValueError("train project names require shard metadata")
        return f"prism_main_{run_version}_train_s{shard_index:02d}of{shard_count:02d}"
    return f"prism_main_{run_version}_{split}"


def run(command: list[str], *, cwd: Path = REPOSITORY_ROOT) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _safe_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    for member in members:
        member_path = Path(member.name)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise RuntimeError(f"unsafe path in asset archive: {member.name}")
        if member.issym() or member.islnk():
            raise RuntimeError(f"links are not allowed in asset archive: {member.name}")
    return members


def _find_asset_root(extracted_root: Path) -> Path:
    direct = extracted_root / "dataset_resources_research"
    candidates = [direct, extracted_root]
    candidates.extend(path for path in extracted_root.iterdir() if path.is_dir())
    for candidate in candidates:
        if (candidate / "background").is_dir() and (candidate / "shape").is_dir():
            return candidate
    raise RuntimeError(
        "asset archive must contain background/ and shape/ directories, "
        "optionally below dataset_resources_research/"
    )


def install_requirements() -> None:
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def prepare_assets(asset_tar: Path) -> Path:
    asset_tar = asset_tar.resolve()
    if not asset_tar.is_file():
        raise FileNotFoundError(f"VESSL asset archive is missing: {asset_tar}")

    destination = REPOSITORY_ROOT / "dataset_resources_research"
    with tempfile.TemporaryDirectory(
        prefix="prism-assets-", dir=str(REPOSITORY_ROOT.parent)
    ) as temporary:
        extracted_root = Path(temporary)
        print(f"Extracting {asset_tar} into a temporary directory", flush=True)
        with tarfile.open(asset_tar, mode="r:*") as archive:
            archive.extractall(extracted_root, members=_safe_members(archive))
        source = _find_asset_root(extracted_root)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(source), str(destination))

    if not (destination / "background").is_dir() or not (
        destination / "shape"
    ).is_dir():
        raise RuntimeError(f"asset extraction did not create a valid dataset: {destination}")
    return destination


def render_command(args: argparse.Namespace, name: str) -> list[str]:
    command = [
        sys.executable,
        "render_dataset.py",
        "--conf",
        "configs/dataset_prism_main.yaml",
        "--device",
        "gpu",
        "--output-folder",
        str(args.output_root),
        "--project_name",
        name,
        "--splits",
        args.split,
    ]
    if args.split == "train":
        command.extend(
            [
                "--shard-count",
                str(args.shard_count),
                "--shard-index",
                str(args.shard_index),
            ]
        )
    return command


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    name = project_name(
        args.split,
        args.run_version,
        args.shard_index,
        args.shard_count,
    )
    dataset_root = args.output_root.resolve() / name
    commit = os.environ.get("PRISM_GIT_COMMIT", "unspecified")

    print(
        f"PRISM_GENERATION_START project={name} split={args.split} commit={commit}",
        flush=True,
    )
    install_requirements()
    prepare_assets(args.asset_tar)
    args.output_root.mkdir(parents=True, exist_ok=True)
    run(render_command(args, name))
    run(
        [
            sys.executable,
            "tools/validate_prism_contract.py",
            str(dataset_root / args.split),
        ]
    )
    run([sys.executable, "tools/freeze_prism_manifest.py", str(dataset_root)])
    run(
        [
            sys.executable,
            "tools/freeze_prism_manifest.py",
            str(dataset_root),
            "--verify",
        ]
    )
    (dataset_root / ".generation_complete").write_text(
        f"commit={commit}\nsplit={args.split}\n", encoding="utf-8"
    )
    run(["sync"])
    print(
        f"PRISM_GENERATION_COMPLETE project={name} output={dataset_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
