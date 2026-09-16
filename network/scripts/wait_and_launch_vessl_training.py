#!/usr/bin/env python3
"""Wait for a PRISM archive volume and launch the dependent GPU training run."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from datetime import datetime, timezone
from typing import Any

import yaml

from repack_vessl_dataset import MANIFEST_NAME, VesslObjectStore, atomic_write_json


LAUNCH_MARKER = "training_launch.json"


def training_command(args: argparse.Namespace) -> str:
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
        "PYTHON_BIN=python network/scripts/install_official_sam2.sh",
        "export PRISM_DATA_ROOT=/input",
        "export PRISM_ARCHIVE_ROOT=/input",
        "export PRISM_MATERIALIZED_ROOT=/input/prism-data",
        "export PRISM_DELETE_ARCHIVES_AFTER_EXTRACT=true",
        "export PRISM_OUTPUT_ROOT=/output/prism-training-v1",
        f"export PRISM_CHECKPOINT_URI=volume://{args.storage_name}/{args.result_volume}",
        "export PRISM_CHECKPOINT_SYNC_SECONDS=300",
        "export PRISM_WANDB_MODE=online",
        "export PRISM_WANDB_PROJECT=PRISM",
        "export PRISM_WANDB_GROUP=main-v6-fresnel",
        "network/scripts/train_prism_all_stages.sh",
    ]
    return "\n".join(lines)


def build_training_spec(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "name": args.run_name,
        "description": (
            "PRISM full-Fresnel Stage 1A-4 training with GLaMa FFC, W&B logging, "
            "persistent checkpoints, and frozen FLUX.1 Fill evaluation."
        ),
        "import": {
            "/input/": f"volume://{args.storage_name}/{args.archive_volume}"
        },
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
    existing = store.read_json(LAUNCH_MARKER)
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
    store.upload(marker_path, LAUNCH_MARKER)
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
    parser.add_argument("--cluster", default="snu-eng-dgx")
    parser.add_argument("--preset", default="a100-1")
    parser.add_argument(
        "--node", action="append", default=["snuengdgx002", "snuengdgx003"]
    )
    parser.add_argument("--image", default="quay.io/vessl-ai/torch:2.3.1-cuda12.1-r5")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--work-dir", default="/root/workspace/prism-launch")
    args = parser.parse_args(argv)
    if args.poll_seconds < 10:
        parser.error("--poll-seconds must be at least 10")
    return args


def main() -> int:
    return launch(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
