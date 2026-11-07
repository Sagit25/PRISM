#!/usr/bin/env bash
# Run the v26 MAM2-style PRISM from scratch on the professor's GPU 1.
# The renderer's seven native 128px projects are verified before any GPU train.
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
generation_root="${PRISM_128_GENERATION_ROOT:-/home/sukhun/prism-datasets/128-20261103}"
dataset_root="${PRISM_128_MERGED_ROOT:-/home/sukhun/prism-datasets/128-20261103-merged}"
output_root="${PRISM_OUTPUT_ROOT:-/home/sukhun/prism-training/v26-128-full}"
python_bin="${PRISM_TRAIN_PYTHON:-$repo_root/.venv/bin/python}"

if [[ "$(git -C "$repo_root" status --porcelain --untracked-files=no)" != "" ]]; then
  echo "PRISM_GPU1_PREFLIGHT_FAIL: training checkout has modified tracked files" >&2
  exit 2
fi
actual_commit="$(git -C "$repo_root" rev-parse HEAD)"
if [[ -n "${PRISM_EXPECTED_COMMIT:-}" && "$actual_commit" != "$PRISM_EXPECTED_COMMIT" ]]; then
  echo "PRISM_GPU1_PREFLIGHT_FAIL: checkout SHA does not match the requested commit" >&2
  exit 2
fi
if [[ ! -x "$python_bin" ]]; then
  echo "PRISM_GPU1_PREFLIGHT_FAIL: training Python is missing: $python_bin" >&2
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
    raise SystemExit("PRISM_GPU1_PREFLIGHT_FAIL: credential file must be user-owned mode 600")
values = json.load(open(path, encoding="utf-8"))
for key in ("HF_TOKEN", "WANDB_API_KEY"):
    value = base64.b64decode(values[key], validate=True)
    if not value or b"\x00" in value:
        raise SystemExit("PRISM_GPU1_PREFLIGHT_FAIL: invalid credential entry")
    sys.stdout.buffer.write(key.encode() + b"\x00" + value + b"\x00")
' "$credential_file")
fi
if [[ -z "${HF_TOKEN:-}" || -z "${WANDB_API_KEY:-}" ]]; then
  echo "PRISM_GPU1_PREFLIGHT_FAIL: HF_TOKEN and WANDB_API_KEY must be set in the server environment" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "PRISM_GPU1_PREFLIGHT_FAIL: nvidia-smi is unavailable" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES=1
export OPENCV_IO_ENABLE_OPENEXR=1
export PYTHONUNBUFFERED=1
export PATH="$(dirname -- "$python_bin"):$PATH"
export PRISM_DATA_ROOT="$dataset_root"
export PRISM_ARCHIVE_ROOT="$dataset_root"
export PRISM_MATERIALIZE_ALL=true
export PRISM_OUTPUT_ROOT="$output_root"
export PRISM_CACHE_ROOT="${PRISM_CACHE_ROOT:-/home/sukhun/prism-model-cache}"
export PRISM_TRAINING_SCHEDULE=staged
export PRISM_FIXED_MAM2_ALPHA=true
export PRISM_INDEPENDENT_PHYSICAL_ALPHA=false
export PRISM_MAM2_MATTER_BACKEND=external_mematte
export PRISM_LOSS_PROFILE="${PRISM_LOSS_PROFILE:-balanced_detail_recovery}"
export PRISM_GRADIENT_CLIP=3
export PRISM_FOREGROUND_REFINEMENT=true
export PRISM_FOREGROUND_DETAIL_REFINEMENT=true
export PRISM_FLOW_GUIDED_LATENT_BRIDGE=true
export PRISM_USE_MAM2_FOREGROUND_PRIOR=true
export PRISM_OPTICAL_CURRICULUM=true
export PRISM_BACKGROUND_PIXEL_REFINER_LR_SCALE=10
export PRISM_FOREGROUND_REFINER_LR_SCALE=5
export PRISM_STAGE4_SELECTION_METRIC=joint_worst_region_psnr
export PRISM_STAGE2_LR=2e-5
export PRISM_STAGE3_LR=2e-5
export PRISM_STAGE4_LR=1e-5
export PRISM_SKIP_STAGE1_PRETRAINED=false
# Stage 1B trains MAM2 semantics/matting with the Stage-1 physical-alpha mode.
# Stage 2 enables fixed MAM2 alpha and a new PAM; transfer only MAM2 weights.
export PRISM_STAGE1_TRANSFER=true
export PRISM_EVIDENCE_FLUX_ONLY=true
export PRISM_FLOW_REFINEMENT=true
export PRISM_RUNTIME_CONTRACT_CHECKS=false
export PRISM_STAGE2_PAIRED_BACKGROUNDS=true
export PRISM_DIFFUSION_BACKEND=auto
export PRISM_DIFFUSION_MIN_WORKING_SIDE=1024
export PRISM_DIFFUSION_CPU_OFFLOAD="${PRISM_DIFFUSION_CPU_OFFLOAD:-auto}"
export PRISM_DIFFUSION_PIXEL_REFINEMENT=true
export PRISM_FLUX_SPATIAL_SINGLETON_EVIDENCE=true
export PRISM_REFINE_INVERSE_EVIDENCE=true
export PRISM_SAVE_REFINEMENT_TRACE=true
export PRISM_STAGE1_BATCH_SIZE="${PRISM_STAGE1_BATCH_SIZE:-2}"
export PRISM_STAGE2_BATCH_SIZE="${PRISM_STAGE2_BATCH_SIZE:-4}"
export PRISM_STAGE34_BATCH_SIZE="${PRISM_STAGE34_BATCH_SIZE:-4}"
export PRISM_FINAL_EVAL_BATCH_SIZE="${PRISM_FINAL_EVAL_BATCH_SIZE:-4}"
export PRISM_FAST_EARLY_STAGES="${PRISM_FAST_EARLY_STAGES:-true}"
export PRISM_RUN_FINAL_EVALUATION=true
export PRISM_WANDB_ENTITY=humangpt
export PRISM_WANDB_PROJECT=PRISM
export PRISM_WANDB_MODE=online
export PRISM_WANDB_GROUP="${PRISM_WANDB_GROUP:-prism-v26-128-full}"
export PRISM_EXPERIMENT_ID="${PRISM_EXPERIMENT_ID:-prism-v26-128-full-gpu1}"
export PRISM_STAGE1A_EPOCHS="${PRISM_STAGE1A_EPOCHS:-20}"
export PRISM_STAGE1B_EPOCHS="${PRISM_STAGE1B_EPOCHS:-20}"
export PRISM_STAGE2_EPOCHS="${PRISM_STAGE2_EPOCHS:-15}"
export PRISM_STAGE3_EPOCHS="${PRISM_STAGE3_EPOCHS:-15}"
export PRISM_STAGE4_EPOCHS="${PRISM_STAGE4_EPOCHS:-25}"
export PRISM_IMAGE_PRETRAIN_CLIP_LENGTH=1
export PRISM_VIDEO_REFINE_CLIP_LENGTH=4
export PRISM_FINAL_EVAL_CLIP_LENGTH=8
export PRISM_TEMPORAL_ABLATION_CLIP_LENGTHS=8
export PRISM_MATTE_FRAME_CHUNK_SIZE=4
export PRISM_SAM2_TEMPORAL_CHUNK_SIZE=4
export PRISM_PARTITION_TRAIN_EPOCHS=false
export PRISM_WORKERS="${PRISM_WORKERS:-8}"
export PRISM_EVAL_WORKERS="${PRISM_EVAL_WORKERS:-4}"
export PRISM_AMP_DTYPE=bfloat16
unset PRISM_MAX_TRAIN_BATCHES PRISM_MAX_EVAL_BATCHES PRISM_MAX_FINAL_EVAL_BATCHES
unset PRISM_STAGE2_INITIAL_CHECKPOINT
if [[ "${PRISM_PAM_ALPHA_STAGE4_ONLY:-false}" != "true" ]]; then
  unset PRISM_STAGE4_INITIAL_CHECKPOINT
fi
unset PRISM_STAGE2_MAM2_DONOR_CHECKPOINT PRISM_STAGE3_BACKGROUND_DONOR_CHECKPOINT
unset PRISM_STAGE4_BACKGROUND_DONOR_CHECKPOINT PRISM_RESTORE_CHECKPOINT_URI

echo "PRISM_GPU1_PREFLIGHT_START commit=$actual_commit gpu=1"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
"$python_bin" - <<'PY'
import os
import cv2
import diffusers
import peft
import wandb
import sam2
import detectron2
from packaging.version import Version
from huggingface_hub import hf_hub_download
import refractive_mam2

if Version(peft.__version__) < Version("0.17.0"):
    raise SystemExit(f"PRISM_GPU1_PREFLIGHT_FAIL: peft {peft.__version__} < 0.17.0")
hf_hub_download("black-forest-labs/FLUX.1-Fill-dev", "model_index.json", token=os.environ["HF_TOKEN"])
if not wandb.Api().viewer:
    raise SystemExit("PRISM_GPU1_PREFLIGHT_FAIL: W&B online authentication failed")
print("PRISM_GPU1_PACKAGE_FLUX_WANDB_OK", diffusers.__version__, flush=True)
PY

sam2_root="${PRISM_SAM2_ROOT:-$repo_root/network/third_party/sam2}"
sam2_checkpoint="${PRISM_SAM2_CHECKPOINT:-$repo_root/network/checkpoints/sam2.1_hiera_large.pt}"
mematte_root="${PRISM_MEMATTE_ROOT:-$repo_root/network/third_party/MEMatte}"
mematte_checkpoint="${PRISM_MEMATTE_CHECKPOINT:-$repo_root/network/checkpoints/MEMatte_AIM500.pth}"
for required in "$sam2_root/sam2" "$sam2_checkpoint" \
  "$mematte_root/configs/MEMatte_S_topk0.25_win_global_long.py" "$mematte_checkpoint"; do
  if [[ ! -e "$required" ]]; then
    echo "PRISM_GPU1_PREFLIGHT_FAIL: official model asset missing: $required" >&2
    exit 2
  fi
done
if [[ ! -s "$sam2_checkpoint" || ! -s "$mematte_checkpoint" ]]; then
  echo "PRISM_GPU1_PREFLIGHT_FAIL: model checkpoint is empty" >&2
  exit 2
fi
export PRISM_MEMATTE_ROOT="$mematte_root"
export PRISM_MEMATTE_CHECKPOINT="$mematte_checkpoint"
export PRISM_SAM2_ROOT="$sam2_root"
export PRISM_SAM2_CHECKPOINT="$sam2_checkpoint"
echo "PRISM_GPU1_OFFICIAL_ASSETS_OK"
"$python_bin" -m pytest -q \
  "$repo_root/network/tests/test_pipeline.py" \
  "$repo_root/network/tests/test_prepare_prism_128_local.py"
echo "PRISM_GPU1_CODE_TESTS_OK"

"$python_bin" "$script_dir/prepare_prism_128_local.py" \
  --generation-root "$generation_root" --dataset-root "$dataset_root" \
  --poll-seconds 60
"$python_bin" "$repo_root/RCDatasetCreation/tools/freeze_prism_manifest.py" \
  "$dataset_root" --verify
for split in train validation test; do
  "$python_bin" "$repo_root/RCDatasetCreation/tools/validate_prism_contract.py" \
    "$dataset_root/$split"
done
echo "PRISM_GPU1_DATA_CONTRACT_OK root=$dataset_root"

mkdir -p "$output_root"
test_sequences="$(find "$dataset_root/test" -maxdepth 1 -name '*_sequence_meta.json' -type f | wc -l | tr -d ' ')"
available_kib="$(df -Pk "$output_root" | awk 'NR == 2 {print $4}')"
# A lossless 8-frame, four-state trace is about 32 MiB per test sequence
# before compression; reserve that plus model cache and checkpoints.
required_kib="$((40 * 1024 * 1024 + test_sequences * 32 * 1024))"
if (( available_kib < required_kib )); then
  echo "PRISM_GPU1_PREFLIGHT_FAIL: insufficient free disk for full test trace (free=${available_kib}KiB required=${required_kib}KiB test_sequences=$test_sequences)" >&2
  exit 2
fi
echo "PRISM_GPU1_DISK_OK free_kib=$available_kib test_sequences=$test_sequences"
"$python_bin" - <<'PY'
import torch
import torchvision
from packaging.version import Version

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("PRISM_GPU1_PREFLIGHT_FAIL: CUDA_VISIBLE_DEVICES=1 did not isolate one GPU")
if Version(torch.__version__.split("+")[0]) < Version("2.7.1") or torch.version.cuda is None:
    raise SystemExit("PRISM_GPU1_PREFLIGHT_FAIL: Blackwell requires PyTorch >=2.7.1 with CUDA 12.8+")
if Version(torch.version.cuda) < Version("12.8"):
    raise SystemExit(f"PRISM_GPU1_PREFLIGHT_FAIL: CUDA wheel is {torch.version.cuda}, expected 12.8+")
if not hasattr(torchvision.ops, "nms"):
    raise SystemExit("PRISM_GPU1_PREFLIGHT_FAIL: torchvision CUDA/torch ABI is broken")
torchvision.ops.nms(
    torch.tensor([[0., 0., 1., 1.]], device="cuda"),
    torch.tensor([1.], device="cuda"), 0.5,
)
gpu = torch.cuda.get_device_properties(0)
if gpu.total_memory < 80 * 1024**3:
    raise SystemExit(f"PRISM_GPU1_PREFLIGHT_FAIL: expected 96GB-class GPU, got {gpu.total_memory}")
if torch.cuda.mem_get_info()[0] < 60 * 1024**3:
    raise SystemExit("PRISM_GPU1_PREFLIGHT_FAIL: GPU 1 has less than 60GiB free")
print("PRISM_GPU1_CUDA_ABI_OK", gpu.name, torch.__version__, torchvision.__version__, flush=True)
PY
exec > >(tee -a "$output_root/run.log") 2>&1
stages="1a,1b,2,3,4"
if [[ "${PRISM_PAM_ALPHA_STAGE4_ONLY:-false}" == "true" ]]; then
  stages="4-pam-alpha-finetune"
fi
echo "PRISM_GPU1_TRAIN_START commit=$actual_commit stages=$stages data=$dataset_root batches=$PRISM_STAGE1_BATCH_SIZE/$PRISM_STAGE2_BATCH_SIZE/$PRISM_STAGE34_BATCH_SIZE"
attempt=1
while :; do
  attempt_log="$output_root/train-attempt-$attempt.log"
  echo "PRISM_GPU1_TRAIN_ATTEMPT number=$attempt early_checkpointing=$PRISM_FAST_EARLY_STAGES batches=$PRISM_STAGE1_BATCH_SIZE/$PRISM_STAGE2_BATCH_SIZE/$PRISM_STAGE34_BATCH_SIZE"
  if bash "$script_dir/train_prism_all_stages.sh" 2>&1 | tee "$attempt_log"; then
    echo "PRISM_GPU1_TRAIN_COMPLETE attempts=$attempt"
    break
  fi
  if ! grep -Eiq 'CUDA out of memory|OutOfMemoryError' "$attempt_log" || (( attempt >= 4 )); then
    echo "PRISM_GPU1_TRAIN_FAILED attempt=$attempt; inspect $attempt_log" >&2
    exit 1
  fi
  failed_stage="$(sed -n 's/.*PRISM_STAGE_SCHEDULE stage=\([^ ]*\).*/\1/p' "$attempt_log" | tail -n 1)"
  if [[ "$PRISM_FAST_EARLY_STAGES" == "true" && ( "$failed_stage" == "1a" || "$failed_stage" == "1b" || "$failed_stage" == "2" ) ]]; then
    export PRISM_FAST_EARLY_STAGES=false
    echo "PRISM_GPU1_OOM_RETRY enabled activation checkpointing for early stages"
  elif (( PRISM_STAGE1_BATCH_SIZE > 1 || PRISM_STAGE2_BATCH_SIZE > 2 || PRISM_STAGE34_BATCH_SIZE > 2 || PRISM_FINAL_EVAL_BATCH_SIZE > 2 )); then
    export PRISM_FAST_EARLY_STAGES=false
    export PRISM_STAGE1_BATCH_SIZE=1
    export PRISM_STAGE2_BATCH_SIZE=2
    export PRISM_STAGE34_BATCH_SIZE=2
    export PRISM_FINAL_EVAL_BATCH_SIZE=2
    echo "PRISM_GPU1_OOM_RETRY reduced batches; completed checkpoints remain in place"
  elif [[ ( "$failed_stage" == "3" || "$failed_stage" == "4" ) && "$PRISM_DIFFUSION_CPU_OFFLOAD" == "auto" ]]; then
    export PRISM_DIFFUSION_CPU_OFFLOAD=on
    echo "PRISM_GPU1_OOM_RETRY enabled FLUX CPU offload after minimum paired batch"
  else
    echo "PRISM_GPU1_TRAIN_FAILED: no safer memory configuration remains" >&2
    exit 1
  fi
  attempt=$((attempt + 1))
done
