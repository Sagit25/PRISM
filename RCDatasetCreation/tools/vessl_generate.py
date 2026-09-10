#!/usr/bin/env python3
"""Run one reproducible PRISM dataset-generation job on VESSL.

The VESSL command only needs to fetch an exact Git commit and invoke this file.
Keeping the generation workflow here prevents the seven cloud runs from
silently drifting apart or depending on Bash-specific inline syntax.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import traceback


DEFAULT_ASSET_TAR = Path("/input/assets/prism-research-assets-v1.tar")
DEFAULT_OUTPUT_ROOT = Path("/root/workspace/persistent_export")
DEFAULT_SYNC_INTERVAL_SECONDS = 300.0
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
        "--checkpoint-root",
        type=Path,
        default=(
            Path(os.environ["PRISM_CHECKPOINT_ROOT"])
            if os.environ.get("PRISM_CHECKPOINT_ROOT")
            else None
        ),
        help=(
            "Optional writable VESSL Storage mount used for periodic, resumable "
            "checkpoints. The renderer continues to use --output-root as local "
            "scratch for performance."
        ),
    )
    parser.add_argument(
        "--checkpoint-uri",
        default=os.environ.get("PRISM_CHECKPOINT_URI"),
        help=(
            "Optional VESSL Storage URI (for example "
            "volume://vessl-storage/prism-main-output) used for in-run "
            "incremental uploads and resume."
        ),
    )
    parser.add_argument(
        "--sync-interval-seconds",
        type=float,
        default=float(
            os.environ.get(
                "PRISM_SYNC_INTERVAL_SECONDS", DEFAULT_SYNC_INTERVAL_SECONDS
            )
        ),
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
    if args.sync_interval_seconds <= 0:
        parser.error("--sync-interval-seconds must be greater than zero")
    if args.checkpoint_root is not None and args.checkpoint_uri:
        parser.error("use only one of --checkpoint-root and --checkpoint-uri")
    if args.checkpoint_uri and not args.checkpoint_uri.startswith("volume://"):
        parser.error("--checkpoint-uri must start with volume://")
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


def sync_tree(source: Path, destination: Path) -> None:
    """Incrementally copy one dataset tree without exposing partial files."""
    source.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True, exist_ok=True)
    run(
        [
            "rsync",
            "-a",
            "--partial-dir=.rsync-partial",
            f"{source}/",
            f"{destination}/",
        ]
    )
    run(["sync"])


def restore_checkpoint(checkpoint_root: Path, output_root: Path, name: str) -> None:
    source = checkpoint_root.resolve() / name
    destination = output_root.resolve() / name
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    probe = checkpoint_root / ".prism_write_probe"
    probe.write_text("writable\n", encoding="utf-8")
    probe.unlink()
    if source.is_dir():
        print(f"Restoring checkpoint {source} -> {destination}", flush=True)
        sync_tree(source, destination)


def _vessl_copy(source: str, destination: str, *, allow_missing: bool = False) -> bool:
    command = ["vessl", "storage", "copy-file", source, destination]
    for attempt in range(1, 4):
        print("+", " ".join(command), flush=True)
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if completed.stdout:
            print(completed.stdout, end="", flush=True)
        if completed.returncode == 0:
            return True
        if attempt < 3:
            time.sleep(float(attempt))
    if allow_missing:
        return False
    raise subprocess.CalledProcessError(completed.returncode, command)


def _uri_join(root: str, *parts: str) -> str:
    return "/".join([root.rstrip("/"), *(part.strip("/") for part in parts)])


def _validate_checkpoint_uri(checkpoint_uri: str, name: str) -> None:
    with tempfile.TemporaryDirectory(prefix="prism-vessl-probe-") as temporary:
        probe = Path(temporary) / f"{name}.txt"
        probe.write_text("PRISM checkpoint transport probe\n", encoding="utf-8")
        _vessl_copy(
            str(probe),
            _uri_join(checkpoint_uri, ".probes", probe.name),
        )


def restore_checkpoint_uri(
    checkpoint_uri: str, output_root: Path, name: str
) -> None:
    _validate_checkpoint_uri(checkpoint_uri, name)
    destination = output_root.resolve() / name
    with tempfile.TemporaryDirectory(prefix="prism-vessl-restore-") as temporary:
        downloaded = Path(temporary) / "downloaded"
        downloaded.mkdir()
        if not _vessl_copy(
            _uri_join(checkpoint_uri, name),
            str(downloaded),
            allow_missing=True,
        ):
            print(f"No existing remote checkpoint for {name}", flush=True)
            return

        candidates = [downloaded / name]
        candidates.extend(path for path in downloaded.rglob(name) if path.is_dir())
        source = next((path for path in candidates if path.is_dir()), downloaded)
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination, dirs_exist_ok=True)
        print(
            f"Restored VESSL checkpoint {_uri_join(checkpoint_uri, name)} "
            f"-> {destination}",
            flush=True,
        )


def _file_snapshot(root: Path) -> dict[str, tuple[int, int]]:
    snapshot: dict[str, tuple[int, int]] = {}
    if not root.is_dir():
        return snapshot
    for path in root.rglob("*"):
        if not path.is_file() or ".rsync-partial" in path.parts:
            continue
        stat = path.stat()
        snapshot[str(path.relative_to(root))] = (stat.st_size, stat.st_mtime_ns)
    return snapshot


def sync_tree_to_uri(
    source: Path,
    checkpoint_uri: str,
    uploaded: dict[str, tuple[int, int]],
    *,
    include_unstable: bool = False,
) -> None:
    """Upload only new or changed files while preserving the project tree."""
    current = _file_snapshot(source)
    now_ns = time.time_ns()
    stable_age_ns = int(5.0 * 1_000_000_000)
    changed = [
        relative
        for relative, signature in current.items()
        if uploaded.get(relative) != signature
        and (
            include_unstable
            or now_ns - signature[1] >= stable_age_ns
            or relative.startswith(".")
        )
    ]
    if not changed:
        print("PRISM_CHECKPOINT_SYNC no changed stable files", flush=True)
        return

    with tempfile.TemporaryDirectory(prefix="prism-vessl-stage-") as temporary:
        staged_project = Path(temporary) / source.name
        for relative in changed:
            source_file = source / relative
            destination_file = staged_project / relative
            destination_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, destination_file)
        _vessl_copy(str(staged_project), checkpoint_uri)
    for relative in changed:
        uploaded[relative] = current[relative]
    print(f"PRISM_CHECKPOINT_UPLOADED files={len(changed)}", flush=True)


@contextmanager
def periodic_checkpoint(
    dataset_root: Path,
    checkpoint_root: Path | None,
    checkpoint_uri: str | None,
    interval_seconds: float,
):
    """Periodically mirror completed output into a persistent VESSL volume."""
    if checkpoint_root is None and checkpoint_uri is None:
        yield lambda _reason, _force=False: None
        return

    destination = (
        checkpoint_root.resolve() / dataset_root.name
        if checkpoint_root is not None
        else checkpoint_uri
    )
    stop = threading.Event()
    sync_lock = threading.Lock()
    uploaded = _file_snapshot(dataset_root) if checkpoint_uri else {}

    def checkpoint(reason: str, force: bool = False) -> None:
        with sync_lock:
            print(
                f"PRISM_CHECKPOINT_SYNC reason={reason} "
                f"source={dataset_root} destination={destination}",
                flush=True,
            )
            if checkpoint_root is not None:
                assert isinstance(destination, Path)
                sync_tree(dataset_root, destination)
            else:
                assert isinstance(destination, str)
                sync_tree_to_uri(
                    dataset_root,
                    destination,
                    uploaded,
                    include_unstable=force,
                )

    def worker() -> None:
        while not stop.wait(interval_seconds):
            try:
                checkpoint("periodic")
            except Exception:
                print("PRISM_CHECKPOINT_SYNC_WARNING", file=sys.stderr, flush=True)
                traceback.print_exc()

    checkpoint("initial")
    thread = threading.Thread(
        target=worker,
        name="prism-checkpoint-sync",
        daemon=True,
    )
    thread.start()
    try:
        yield checkpoint
    finally:
        stop.set()
        thread.join(timeout=max(1.0, min(interval_seconds, 30.0)))
        checkpoint("final")


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


def execute(args: argparse.Namespace) -> Path:
    name = project_name(
        args.split,
        args.run_version,
        args.shard_index,
        args.shard_count,
    )
    dataset_root = args.output_root.resolve() / name
    commit = os.environ.get("PRISM_GIT_COMMIT", "unspecified")

    if args.checkpoint_root is not None:
        restore_checkpoint(args.checkpoint_root, args.output_root, name)
    elif args.checkpoint_uri:
        restore_checkpoint_uri(args.checkpoint_uri, args.output_root, name)

    dataset_root.mkdir(parents=True, exist_ok=True)
    for marker_name in (".generation_complete", ".generation_failed"):
        marker = dataset_root / marker_name
        if marker.exists():
            marker.unlink()

    print(
        f"PRISM_GENERATION_START project={name} split={args.split} commit={commit}",
        flush=True,
    )
    with periodic_checkpoint(
        dataset_root,
        args.checkpoint_root,
        args.checkpoint_uri,
        args.sync_interval_seconds,
    ) as checkpoint:
        install_requirements()
        prepare_assets(args.asset_tar)
        run([sys.executable, "tools/prepare_prism_assets.py", "validate"])
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
        checkpoint("complete", True)
    print(
        f"PRISM_GENERATION_COMPLETE project={name} output={dataset_root}",
        flush=True,
    )
    return dataset_root


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    name = project_name(
        args.split,
        args.run_version,
        args.shard_index,
        args.shard_count,
    )
    dataset_root = args.output_root.resolve() / name
    try:
        execute(args)
    except BaseException:
        # VESSL's export phase is invoked by the small POSIX wrapper after this
        # process exits.  Persist a machine-readable failure marker next to any
        # completed frames so a replacement run can import and resume them.
        dataset_root.mkdir(parents=True, exist_ok=True)
        failure = traceback.format_exc()
        (dataset_root / ".generation_failed").write_text(
            failure, encoding="utf-8"
        )
        if args.checkpoint_root is not None:
            try:
                sync_tree(
                    dataset_root,
                    args.checkpoint_root.resolve() / name,
                )
            except Exception:
                print(
                    "PRISM_CHECKPOINT_FINAL_SYNC_FAILED",
                    file=sys.stderr,
                    flush=True,
                )
                traceback.print_exc()
        elif args.checkpoint_uri:
            try:
                _vessl_copy(
                    str(dataset_root / ".generation_failed"),
                    _uri_join(
                        args.checkpoint_uri,
                        name,
                        ".generation_failed",
                    ),
                )
            except Exception:
                print(
                    "PRISM_CHECKPOINT_FAILURE_MARKER_UPLOAD_FAILED",
                    file=sys.stderr,
                    flush=True,
                )
                traceback.print_exc()
        print(
            f"PRISM_GENERATION_FAILED project={name} output={dataset_root}",
            file=sys.stderr,
            flush=True,
        )
        raise


if __name__ == "__main__":
    def _terminate_gracefully(signum: int, _frame: object) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _terminate_gracefully)
    signal.signal(signal.SIGINT, _terminate_gracefully)
    main()
