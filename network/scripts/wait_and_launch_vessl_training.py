#!/usr/bin/env python3
"""Wait for a PRISM archive volume and launch the dependent GPU training run."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time
from datetime import datetime, timezone
from typing import Any

import yaml

from repack_vessl_dataset import MANIFEST_NAME, VesslObjectStore, atomic_write_json


LAUNCH_MARKER = "training_launch.json"


def training_command(args: argparse.Namespace) -> str:
    output_subdir = args.output_subdir or args.run_name
    lines = [
        "set -Eeuo pipefail",
        "export DEBIAN_FRONTEND=noninteractive",
        "export OPENCV_IO_ENABLE_OPENEXR=1",
        "apt-get update",
        "apt-get install -y git curl libgl1 libglib2.0-0",
        "python -m pip install --quiet --upgrade pip vessl",
        "git init /root/workspace/PRISM",
        "cd /root/workspace/PRISM",
        "git remote add origin https://github.com/Sagit25/PRISM.git",
        f"git fetch --depth 1 origin {args.git_commit}",
        "git checkout --detach FETCH_HEAD",
        'python -m pip install -e "./network[data,sam2,experiment,evaluation,diffusion]"',
        (
            "python -c 'import torch, torchvision; "
            'assert torch.__version__.startswith("2.5.1"), torch.__version__; '
            'assert torchvision.__version__.startswith("0.20.1"), '
            "torchvision.__version__; assert torch.cuda.is_available(), "
            '"CUDA is unavailable"; print("PRISM_RUNTIME_OK", '
            "torch.__version__, torchvision.__version__, "
            'torch.version.cuda, torch.cuda.get_device_name(0))\''
        ),
        "PYTHON_BIN=python network/scripts/install_official_sam2.sh",
        "export PRISM_DATA_ROOT=/root/workspace/prism-data",
        f"export PRISM_ARCHIVE_VOLUME={args.archive_volume}",
        f"export PRISM_ARCHIVE_STORAGE_NAME={args.storage_name}",
        "export PRISM_ARCHIVE_DOWNLOAD_ROOT=/root/workspace/prism-archive-downloads",
        "export PRISM_MATERIALIZED_ROOT=/root/workspace/prism-data",
        "export PRISM_DELETE_ARCHIVES_AFTER_EXTRACT=true",
        "export PRISM_AMP_DTYPE=bfloat16",
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        f"export PRISM_OUTPUT_ROOT=/output/{output_subdir}",
        (
            "export PRISM_CHECKPOINT_URI="
            f"volume://{args.storage_name}/{args.result_volume}/{output_subdir}"
        ),
        "export PRISM_CHECKPOINT_SYNC_SECONDS=300",
        "export PRISM_WANDB_MODE=online",
        "export PRISM_WANDB_PROJECT=PRISM",
        "export PRISM_WANDB_GROUP=main-v6-fresnel",
    ]
    if args.mock:
        lines.extend(
            [
                "export PRISM_ARCHIVE_MAX_SHARDS_PER_COMPONENT=1",
                "export PRISM_STAGE1A_EPOCHS=1",
                "export PRISM_STAGE1B_EPOCHS=1",
                "export PRISM_STAGE2_EPOCHS=1",
                "export PRISM_STAGE3_EPOCHS=1",
                "export PRISM_STAGE4_EPOCHS=1",
                "export PRISM_DIFFUSION_STEPS=2",
                f"export PRISM_EXPERIMENT_ID={output_subdir}",
                f"export PRISM_WANDB_GROUP={output_subdir}",
            ]
        )
    lines.append("network/scripts/train_prism_all_stages.sh")
    return "\n".join(lines)


def build_training_spec(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "name": args.run_name,
        "description": (
            "PRISM full-Fresnel Stage 1A-4 training with GLaMa FFC, W&B logging, "
            "persistent checkpoints, and frozen FLUX.1 Fill evaluation."
        ),
        "export": {"/output/": f"volume://{args.storage_name}"},
        "resources": {
            "cluster": args.cluster,
            "preset": args.preset,
            "node_names": args.node,
        },
        "image": args.image,
        "run": [
            {
                "command": training_command(args),
                "workdir": "/root/workspace",
            }
        ],
    }


def read_complete_manifest(store: VesslObjectStore) -> dict[str, Any] | None:
    manifest = store.read_json(MANIFEST_NAME)
    if manifest is None or not manifest.get("complete"):
        return None
    totals = manifest.get("totals", {})
    if not totals.get("shard_count") or not totals.get("file_count"):
        raise RuntimeError("Archive completion manifest has invalid totals")
    return manifest


def launch(args: argparse.Namespace) -> int:
    store = VesslObjectStore(
        storage_name=args.storage_name,
        source_volume_name=args.archive_volume,
        destination_volume_name=args.archive_volume,
    )
    marker_name = args.launch_marker or LAUNCH_MARKER
    existing = store.read_json(marker_name)
    if existing and existing.get("run_id"):
        print(
            f"PRISM_TRAINING_ALREADY_LAUNCHED run_id={existing['run_id']}", flush=True
        )
        return 0

    while True:
        manifest = read_complete_manifest(store)
        if manifest is not None:
            break
        print(
            f"PRISM_ARCHIVE_NOT_READY retry_in={args.poll_seconds}s "
            f"volume={args.archive_volume}",
            flush=True,
        )
        time.sleep(args.poll_seconds)

    spec = build_training_spec(args)
    yaml_body = yaml.safe_dump(spec, sort_keys=False)
    print(
        f"PRISM_ARCHIVE_READY shards={manifest['totals']['shard_count']} "
        f"files={manifest['totals']['file_count']}",
        flush=True,
    )
    import vessl

    response = vessl.create_run(
        yaml_file=None,
        yaml_body=yaml_body,
        yaml_file_name="prism-training.yaml",
        organization_name=args.organization,
        project_name=args.project,
    )
    run_id = getattr(response, "id", None)
    if run_id is None:
        raise RuntimeError("VESSL created a run but did not return its ID")

    marker = {
        "run_id": int(run_id),
        "run_name": args.run_name,
        "git_commit": args.git_commit,
        "archive_volume": args.archive_volume,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    marker_path = pathlib.Path(args.work_dir) / LAUNCH_MARKER
    atomic_write_json(marker_path, marker)
    store.upload(marker_path, marker_name)
    print(f"PRISM_TRAINING_LAUNCHED run_id={run_id}", flush=True)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--organization", default="snu-eng-dgx-heavy")
    parser.add_argument("--project", default="PRISM")
    parser.add_argument("--storage-name", default="vessl-storage")
    parser.add_argument("--archive-volume", required=True)
    parser.add_argument("--result-volume", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--run-name", default="prism-train-all-stages-flux-fill-v2")
    parser.add_argument(
        "--output-subdir",
        help="Result-volume subdirectory; defaults to the run name.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use one shard per split and one epoch per stage before the full run.",
    )
    parser.add_argument(
        "--launch-marker",
        help="Override the archive-volume launch marker for an independent run.",
    )
    parser.add_argument("--cluster", default="snu-eng-dgx")
    parser.add_argument("--preset", default="a100-1")
    parser.add_argument(
        "--node", action="append", default=["snuengdgx002", "snuengdgx003"]
    )
    parser.add_argument(
        "--image",
        default="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
        help=(
            "Container image. The default matches official SAM2's pinned "
            "PyTorch/torchvision ABI and remains compatible with CUDA 12.8 drivers."
        ),
    )
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--work-dir", default="/root/workspace/prism-launch")
    args = parser.parse_args(argv)
    if args.poll_seconds < 10:
        parser.error("--poll-seconds must be at least 10")
    output_subdir = args.output_subdir or args.run_name
    if re.fullmatch(r"[A-Za-z0-9._-]+", output_subdir) is None:
        parser.error("--output-subdir/run-name must be a safe single path component")
    return args


def main() -> int:
    return launch(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
