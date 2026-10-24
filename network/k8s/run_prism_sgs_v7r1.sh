#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
archive_root="/data/datasets/prism-full-64-singleflow-pack-v2-369367271785-output"
run_name="${PRISM_SGS_RUN_NAME:-prism-sgs-full-64-v7r1-aim500}"
output_root="/data/checkpoints/$run_name"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is required for online SGS training" >&2
  exit 2
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required for gated FLUX.1-Fill-dev" >&2
  exit 2
fi
for required in \
  "$archive_root/archive_manifest.json" \
  "$archive_root/dataset_manifest.json" \
  /data/models/MEMatte_AIM500.pth; do
  if [[ ! -s "$required" ]]; then
    echo "Required SGS volume file is missing: $required" >&2
    exit 2
  fi
done

cd "$repo_root"
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y git curl build-essential ninja-build libgl1 libglib2.0-0
actual_commit="$(git rev-parse HEAD)"
if [[ -n "${PRISM_SGS_EXPECTED_COMMIT:-}" \
  && "$actual_commit" != "$PRISM_SGS_EXPECTED_COMMIT" ]]; then
  echo "Wrong PRISM commit: expected $PRISM_SGS_EXPECTED_COMMIT, got $actual_commit" >&2
  exit 2
fi
echo "PRISM_SGS_COMMIT $actual_commit"
echo "PRISM_SGS_ARCHIVE $archive_root"
echo "PRISM_SGS_CHECKPOINTS $output_root"

python -m pip install --quiet --upgrade pip
python -m pip install -e './network[data,sam2,experiment,evaluation,diffusion]'

python - <<'PY'
import os
from huggingface_hub import HfApi
import torch
import torchvision

HfApi().model_info("black-forest-labs/FLUX.1-Fill-dev", token=os.environ["HF_TOKEN"])
print("PRISM_FLUX_ACCESS_OK", flush=True)
assert torch.__version__.startswith("2.5.1"), torch.__version__
assert torchvision.__version__.startswith("0.20.1"), torchvision.__version__
assert torch.cuda.is_available(), "CUDA is unavailable"
print("PRISM_RUNTIME_OK", torch.__version__, torchvision.__version__, torch.version.cuda, torch.cuda.get_device_name(0), flush=True)
PY

export PRISM_MEMATTE_CHECKPOINT=/data/models/MEMatte_AIM500.pth
PYTHON_BIN=python network/scripts/install_official_sam2.sh
PYTHON_BIN=python network/scripts/install_official_mematte.sh
if [[ "${PRISM_SGS_RUN_TESTS:-false}" == "true" ]]; then
  python -m pip install --quiet pytest
  PYTHONPATH="$repo_root/network" python -m pytest -q \
    network/tests/test_inverse_joint.py network/tests/test_matter.py \
    network/tests/test_pipeline.py \
    network/tests/test_dataset_contract.py \
    network/tests/test_temporal_geometry.py
  echo "PRISM_SGS_UNIT_TESTS_OK"
fi

export PRISM_DATA_ROOT=/data/datasets/prism-full-64-materialized-v7r1
export PRISM_ARCHIVE_ROOT="$archive_root"
export PRISM_MATERIALIZED_ROOT="$PRISM_DATA_ROOT"
export PRISM_MATERIALIZE_ALL=true
export PRISM_VERIFY_ARCHIVES=true
export PRISM_DELETE_ARCHIVES_AFTER_EXTRACT=false
unset PRISM_ARCHIVE_VOLUME PRISM_CHECKPOINT_URI PRISM_RESTORE_CHECKPOINT_URI
export PRISM_OUTPUT_ROOT="$output_root"
export PRISM_CACHE_ROOT="${PRISM_CACHE_ROOT:-/data/cache/prism-shared-models}"
export PRISM_SHARD_CACHE_ROOT="/data/cache/prism-shards-$run_name"
export PRISM_BACKGROUND_EXCLUSION_DILATION=1
export PRISM_DIFFUSION_MASK_DILATION=0
export PRISM_FLOW_REFINEMENT=true
export PRISM_EVIDENCE_FLUX_ONLY=true
export PRISM_RUNTIME_CONTRACT_CHECKS=true
export PRISM_STAGE2_PAIRED_BACKGROUNDS=true
export PRISM_KEEP_EPOCH_CHECKPOINTS=false
export PRISM_CHECKPOINT_INTERVAL_STEPS="${PRISM_CHECKPOINT_INTERVAL_STEPS:-100}"
export PRISM_AMP_DTYPE=bfloat16
export PRISM_IMAGE_PRETRAIN_CLIP_LENGTH=1
export PRISM_VIDEO_REFINE_CLIP_LENGTH=4
export PRISM_FINAL_EVAL_CLIP_LENGTH=8
export PRISM_MATTE_FRAME_CHUNK_SIZE=4
export PRISM_SAM2_TEMPORAL_CHUNK_SIZE=4
export PRISM_WANDB_MODE=online
export PRISM_WANDB_ENTITY=humangpt
export PRISM_WANDB_PROJECT=PRISM
export PRISM_WANDB_GROUP="$run_name"
export PRISM_EXPERIMENT_ID="$run_name"
export PYTHONUNBUFFERED=1

if [[ -n "${PRISM_SGS_STAGE1_DONOR_RUN:-}" ]]; then
  donor_run="$PRISM_SGS_STAGE1_DONOR_RUN"
  if [[ ! "$donor_run" =~ ^prism-kubectl-[a-z0-9-]+$ ]]; then
    echo "Invalid PRISM_SGS_STAGE1_DONOR_RUN: $donor_run" >&2
    exit 2
  fi
  for stage in 1a 1b; do
    donor_dir="/data/checkpoints/$donor_run/checkpoints/stage$stage"
    target_dir="$output_root/checkpoints/stage$stage"
    checkpoint="prism_stage${stage}_best.pt"
    if [[ ! -s "$donor_dir/$checkpoint" || ! -f "$donor_dir/.training_complete" ]]; then
      echo "Completed Stage 1 donor is missing: $donor_dir" >&2
      exit 2
    fi
    mkdir -p "$target_dir"
    if [[ ! -s "$target_dir/$checkpoint" ]]; then
      cp -p "$donor_dir/$checkpoint" "$target_dir/$checkpoint"
    fi
    touch "$target_dir/.training_complete"
  done
  export PRISM_STAGE1_TRANSFER=true
  echo "PRISM_STAGE1_TRANSFER_OK donor=$donor_run target=$run_name"
fi

echo "PRISM_SGS_WANDB_ONLINE entity=$PRISM_WANDB_ENTITY project=$PRISM_WANDB_PROJECT group=$PRISM_WANDB_GROUP"
exec network/scripts/train_prism_all_stages.sh
