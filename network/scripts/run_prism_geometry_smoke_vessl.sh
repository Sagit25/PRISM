#!/usr/bin/env bash
set -Eeuo pipefail

# One-run cloud smoke path: render and validate the compact geometry dataset,
# then train the hybrid T=1 -> T=4 curriculum directly from local scratch.
# A background rsync preserves the generated dataset while CUDA training starts.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
dataset_scratch="${PRISM_GEOMETRY_SMOKE_DATA_ROOT:-/root/workspace/prism-geometry-smoke-data}"
project_name="${PRISM_GEOMETRY_SMOKE_PROJECT_NAME:-prism_geometry_smoke}"
dataset_root="$dataset_scratch/$project_name"
output_root="${PRISM_OUTPUT_ROOT:-/output/prism-hybrid-geometry-smoke-v18}"
asset_tar="${PRISM_ASSET_TAR:-/input/assets/prism-research-assets-v1.tar}"
dataset_copy_pid=""

finish_dataset_copy() {
  if [[ -n "$dataset_copy_pid" ]]; then
    wait "$dataset_copy_pid"
  fi
}
trap finish_dataset_copy EXIT

export OPENCV_IO_ENABLE_OPENEXR=1
export PYTHONUNBUFFERED=1

cd "$repo_root/RCDatasetCreation"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt
PRISM_SMOKE_ASSET_TAR="$asset_tar" python -c \
  "import os; from pathlib import Path; from tools.vessl_generate import prepare_assets; prepare_assets(Path(os.environ['PRISM_SMOKE_ASSET_TAR']))"
python tools/prepare_prism_assets.py validate

python render_dataset.py \
  --conf configs/dataset_prism_geometry_smoke.yaml \
  --device gpu \
  --output-folder "$dataset_scratch" \
  --project_name "$project_name" \
  --splits train validation test

for split in train validation test; do
  python tools/validate_prism_contract.py "$dataset_root/$split" --require-pairs
done
python tools/freeze_prism_manifest.py "$dataset_root"
python tools/freeze_prism_manifest.py "$dataset_root" --verify
echo "PRISM_GEOMETRY_SMOKE_DATA_VALIDATED root=$dataset_root"

mkdir -p "$output_root/dataset"
rsync -a "$dataset_root/" "$output_root/dataset/" &
dataset_copy_pid="$!"

cd "$repo_root"
python -m pip install --quiet -e "./network[data,sam2,experiment,evaluation]"
PYTHON_BIN=python network/scripts/install_official_sam2.sh
PYTHON_BIN=python network/scripts/install_official_mematte.sh

export PRISM_DATA_ROOT="$dataset_root"
export PRISM_OUTPUT_ROOT="$output_root"
export PRISM_PREDICT_GEOMETRY=true
export PRISM_IMAGE_PRETRAIN_CLIP_LENGTH=1
export PRISM_VIDEO_REFINE_CLIP_LENGTH=4
export PRISM_FINAL_EVAL_CLIP_LENGTH=4
export PRISM_TEMPORAL_ABLATION_CLIP_LENGTHS=1,2,4
export PRISM_STAGE2_PAIRED_BACKGROUNDS=true
export PRISM_ARCHIVE_VOLUME=""
export PRISM_RUN_FINAL_EVALUATION=false
export PRISM_WANDB_PROJECT="${PRISM_WANDB_PROJECT:-PRISM}"
export PRISM_WANDB_ENTITY="${PRISM_WANDB_ENTITY:-humangpt}"
export PRISM_WANDB_MODE="${PRISM_WANDB_MODE:-online}"
export PRISM_EXPERIMENT_ID="${PRISM_EXPERIMENT_ID:-prism-hybrid-geometry-smoke-v18}"
export PRISM_WANDB_GROUP="${PRISM_WANDB_GROUP:-prism-hybrid-geometry-smoke-v18}"

network/scripts/train_prism_geometry_smoke.sh
finish_dataset_copy
dataset_copy_pid=""
echo "PRISM_HYBRID_GEOMETRY_SMOKE_COMPLETE output=$output_root"
