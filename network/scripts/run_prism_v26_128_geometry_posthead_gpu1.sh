#!/usr/bin/env bash
# Run only a new visible-surface geometry head after GPU1 v26 Stage 4 finishes.
# The donor weights and the original training directory are never modified.
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
donor_root="${PRISM_GEOMETRY_DONOR_ROOT:-/home/sukhun/prism-training/v26-128-full}"
donor_checkpoint="${PRISM_GEOMETRY_DONOR_CHECKPOINT:-$donor_root/checkpoints/stage4/prism_stage4_best.pt}"
output_root="${PRISM_OUTPUT_ROOT:-/home/sukhun/prism-training/v26-128-geometry-posthead}"
dataset_root="${PRISM_DATA_ROOT:-/home/sukhun/prism-datasets/128-20261103-merged}"
donor_repo="${PRISM_V26_REPO_ROOT:-/home/sukhun/prism-training-v26-128-20261103}"
python_bin="${PRISM_TRAIN_PYTHON:-$donor_repo/.venv/bin/python}"

if [[ ! -s "$donor_checkpoint" \
  || ! -f "$donor_root/checkpoints/stage4/.training_complete" ]]; then
  echo "Completed v26 Stage 4 donor checkpoint is required: $donor_checkpoint" >&2
  exit 2
fi
if [[ "$output_root" == "$donor_root" || "$output_root" == "$donor_root"/* ]]; then
  echo "Geometry output must be separate from the donor training directory" >&2
  exit 2
fi
if [[ ! -x "$python_bin" || ! -f "$dataset_root/dataset_manifest.json" ]]; then
  echo "GPU1 Python or verified 128px dataset is missing" >&2
  exit 2
fi
credential_file="${PRISM_CREDENTIAL_FILE:-/home/sukhun/.config/prism/credentials.json}"
if [[ ( -z "${HF_TOKEN:-}" || -z "${WANDB_API_KEY:-}" ) && -f "$credential_file" ]]; then
  while IFS= read -r -d '' credential_name && IFS= read -r -d '' credential_value; do
    export "$credential_name=$credential_value"
  done < <("$python_bin" -c '
import base64, json, os, stat, sys
path = sys.argv[1]
status = os.stat(path)
if status.st_uid != os.getuid() or stat.S_IMODE(status.st_mode) != 0o600:
    raise SystemExit("GPU1 credential file must be user-owned mode 600")
values = json.load(open(path, encoding="utf-8"))
for key in ("HF_TOKEN", "WANDB_API_KEY"):
    value = base64.b64decode(values[key], validate=True)
    if not value or b"\x00" in value:
        raise SystemExit("Invalid GPU1 credential entry")
    sys.stdout.buffer.write(key.encode() + b"\x00" + value + b"\x00")
' "$credential_file")
fi
if [[ -z "${HF_TOKEN:-}" || -z "${WANDB_API_KEY:-}" ]]; then
  echo "Use the existing GPU1 HF_TOKEN and WANDB_API_KEY environment" >&2
  exit 2
fi

# Reuse the already installed GPU1 dependencies without changing the donor
# environment's editable package or the running/finished v26 checkout.
mkdir -p "$output_root/python_path"
package_link="$output_root/python_path/refractive_mam2"
if [[ -L "$package_link" ]]; then
  [[ "$(readlink "$package_link")" == "$repo_root/network/src" ]] || {
    echo "Geometry package link points to another checkout" >&2
    exit 2
  }
elif [[ ! -e "$package_link" ]]; then
  ln -s "$repo_root/network/src" "$package_link"
else
  echo "Geometry package path already exists and is not the expected link" >&2
  exit 2
fi
export PYTHONPATH="$output_root/python_path${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$(dirname -- "$python_bin"):$PATH"
export CUDA_VISIBLE_DEVICES=1
export OPENCV_IO_ENABLE_OPENEXR=1
export PYTHONUNBUFFERED=1
export PRISM_DATA_ROOT="$dataset_root"
export PRISM_ARCHIVE_ROOT="$dataset_root"
export PRISM_MATERIALIZE_ALL=true
export PRISM_OUTPUT_ROOT="$output_root"
export PRISM_CACHE_ROOT="${PRISM_CACHE_ROOT:-/home/sukhun/prism-model-cache}"
export PRISM_GEOMETRY_POST_HEAD_ONLY=true
export PRISM_STAGE4_INITIAL_CHECKPOINT="$donor_checkpoint"
export PRISM_PREDICT_GEOMETRY=false
export PRISM_FIXED_MAM2_ALPHA=true
export PRISM_INDEPENDENT_PHYSICAL_ALPHA=false
export PRISM_MAM2_MATTER_BACKEND=external_mematte
export PRISM_MEMATTE_ROOT="${PRISM_MEMATTE_ROOT:-$donor_repo/network/third_party/MEMatte}"
export PRISM_MEMATTE_CHECKPOINT="${PRISM_MEMATTE_CHECKPOINT:-$donor_repo/network/checkpoints/MEMatte_AIM500.pth}"
export PRISM_SAM2_ROOT="${PRISM_SAM2_ROOT:-$donor_repo/network/third_party/sam2}"
export PRISM_SAM2_CHECKPOINT="${PRISM_SAM2_CHECKPOINT:-$donor_repo/network/checkpoints/sam2.1_hiera_large.pt}"
export PRISM_LOSS_PROFILE=balanced_detail_recovery
export PRISM_FOREGROUND_REFINEMENT=true
export PRISM_FOREGROUND_DETAIL_REFINEMENT=true
export PRISM_FLOW_REFINEMENT=true
export PRISM_FLOW_GUIDED_LATENT_BRIDGE=true
export PRISM_USE_MAM2_FOREGROUND_PRIOR=true
export PRISM_EVIDENCE_FLUX_ONLY=true
export PRISM_DIFFUSION_BACKEND=auto
export PRISM_DIFFUSION_MIN_WORKING_SIDE=1024
export PRISM_DIFFUSION_PIXEL_REFINEMENT=true
export PRISM_FLUX_SPATIAL_SINGLETON_EVIDENCE=true
export PRISM_REFINE_INVERSE_EVIDENCE=true
export PRISM_STAGE34_BATCH_SIZE="${PRISM_STAGE34_BATCH_SIZE:-2}"
export PRISM_FINAL_EVAL_BATCH_SIZE="${PRISM_FINAL_EVAL_BATCH_SIZE:-2}"
export PRISM_VIDEO_REFINE_CLIP_LENGTH=4
export PRISM_FINAL_EVAL_CLIP_LENGTH=8
export PRISM_MATTE_FRAME_CHUNK_SIZE=4
export PRISM_SAM2_TEMPORAL_CHUNK_SIZE=4
export PRISM_AMP_DTYPE=bfloat16
export PRISM_STAGE4_EPOCHS="${PRISM_GEOMETRY_EPOCHS:-12}"
export PRISM_STAGE4_LR="${PRISM_GEOMETRY_LR:-1e-4}"
export PRISM_STAGE4_SELECTION_METRIC=
export PRISM_WANDB_ENTITY=humangpt
export PRISM_WANDB_PROJECT=PRISM
export PRISM_WANDB_MODE=online
export PRISM_WANDB_GROUP="${PRISM_WANDB_GROUP:-prism-v26-128-geometry}"
export PRISM_EXPERIMENT_ID="${PRISM_EXPERIMENT_ID:-prism-v26-128-geometry-gpu1}"

"$python_bin" - <<'PY'
import refractive_mam2.geometry_posthead as geometry
print("PRISM_GEOMETRY_POST_HEAD_CODE_READY", geometry.__file__, flush=True)
PY
echo "PRISM_GEOMETRY_POST_HEAD_START donor=$donor_checkpoint output=$output_root"
bash "$script_dir/train_prism_all_stages.sh"
