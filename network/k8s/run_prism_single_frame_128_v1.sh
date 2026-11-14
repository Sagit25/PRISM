#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
run_name="${PRISM_RUN_NAME:-prism-kubectl-128-ablation-single-frame-maskfix-v1}"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git build-essential ninja-build libgl1 libglib2.0-0
expected_commit="${PRISM_EXPECTED_COMMIT:?full code SHA is required}"
actual_commit="$(git -C "$repo_root" rev-parse HEAD)"
if [[ "$actual_commit" != "$expected_commit" ]]; then
  echo "PRISM_SINGLE_FRAME_PREFLIGHT_FAIL code SHA mismatch" >&2
  exit 2
fi

data_root=/data/datasets/prism-main-128-20261105/prism_main_128_20261105_merged
output_root="/data/checkpoints/$run_name"
for required in \
  "$data_root/dataset_manifest.json" \
  "$data_root/train" "$data_root/validation" "$data_root/test" \
  /data/models/MEMatte_AIM500.pth \
  /data/models/sam2.1_hiera_large.pt \
  /data/code/PRISM/network/third_party/MEMatte/configs/MEMatte_S_topk0.25_win_global_long.py \
  /data/code/PRISM/network/third_party/sam2/sam2; do
  if [[ ! -e "$required" ]]; then
    echo "PRISM_SINGLE_FRAME_PREFLIGHT_FAIL missing $required" >&2
    exit 2
  fi
done
if [[ -z "${WANDB_API_KEY:-}" || -z "${HF_TOKEN:-}" ]]; then
  echo "PRISM_SINGLE_FRAME_PREFLIGHT_FAIL W&B and HF credentials are required" >&2
  exit 2
fi
if [[ -e "$output_root/checkpoints/stage1a/.training_complete" ]]; then
  [[ -s "$output_root/checkpoints/stage1a/prism_stage1a_best.pt" ]] || {
    echo "PRISM_SINGLE_FRAME_PREFLIGHT_FAIL completed Stage 1A checkpoint is missing" >&2
    exit 2
  }
  echo "PRISM_SINGLE_FRAME_REUSE_STAGE1A checkpoint=$output_root/checkpoints/stage1a/prism_stage1a_best.pt"
fi

python -m pip install -q --no-deps -e "$repo_root/network"
python -m pip install -q \
  'numpy>=1.24,<2' 'opencv-python-headless>=4.8,<5' \
  'hydra-core>=1.3,<2' 'iopath>=0.1.10,<0.2' \
  'diffusers>=0.35,<0.38' 'transformers>=4.48,<5' \
  'accelerate>=1.0,<2' 'peft>=0.17,<1' \
  'safetensors>=0.4,<1' 'sentencepiece>=0.2,<1' \
  'wandb>=0.17,<1' 'lpips>=0.1.4,<0.2' \
  'timm==0.5.4' fairscale easydict scikit-image pytest
SAM2_BUILD_CUDA=0 python -m pip install -q --no-build-isolation --no-deps \
  -e /data/code/PRISM/network/third_party/sam2
python -m pip install -q --no-build-isolation \
  'git+https://github.com/facebookresearch/detectron2.git@1e3e13bbf607b54f62205c4c33922521822fb298'

export PRISM_MEMATTE_ROOT=/data/code/PRISM/network/third_party/MEMatte
export PRISM_MEMATTE_CHECKPOINT=/data/models/MEMatte_AIM500.pth
export PRISM_SAM2_ROOT=/data/code/PRISM/network/third_party/sam2
export PRISM_SAM2_CHECKPOINT=/data/models/sam2.1_hiera_large.pt
export HF_HOME=/eval/cache/prism-mam2-pam2-hf
export PRISM_DATA_ROOT="$data_root"
export PRISM_ARCHIVE_ROOT="$data_root"
export PRISM_OUTPUT_ROOT="$output_root"
export PRISM_CACHE_ROOT=/data/cache/prism-single-frame-128
export PRISM_ABLATION=single_frame
export PRISM_TRAINING_SCHEDULE=staged
export PRISM_STAGE1_TRANSFER=true
export PRISM_FIXED_MAM2_ALPHA=true
export PRISM_INDEPENDENT_PHYSICAL_ALPHA=false
export PRISM_MAM2_MATTER_BACKEND=external_mematte
export PRISM_LOSS_PROFILE=balanced_detail_recovery
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
export PRISM_EVIDENCE_FLUX_ONLY=true
export PRISM_FLOW_REFINEMENT=true
export PRISM_STAGE2_PAIRED_BACKGROUNDS=true
export PRISM_DIFFUSION_BACKEND=auto
export PRISM_DIFFUSION_MIN_WORKING_SIDE=1024
export PRISM_DIFFUSION_CPU_OFFLOAD=sequential
export PRISM_DIFFUSION_PIXEL_REFINEMENT=true
export PRISM_FLUX_SPATIAL_SINGLETON_EVIDENCE=true
export PRISM_REFINE_INVERSE_EVIDENCE=true
export PRISM_PAIRED_MICROBATCH_CHECKPOINTING=false
export PRISM_SAVE_REFINEMENT_TRACE=false
export PRISM_STAGE1_BATCH_SIZE=1
export PRISM_STAGE2_BATCH_SIZE=2
export PRISM_STAGE34_BATCH_SIZE=2
export PRISM_FINAL_EVAL_BATCH_SIZE=2
export PRISM_FAST_EARLY_STAGES=false
export PRISM_STAGE1A_EPOCHS=20
export PRISM_STAGE1B_EPOCHS=20
export PRISM_STAGE2_EPOCHS=15
export PRISM_STAGE3_EPOCHS=15
export PRISM_STAGE4_EPOCHS=25
export PRISM_IMAGE_PRETRAIN_CLIP_LENGTH=1
export PRISM_VIDEO_REFINE_CLIP_LENGTH=1
export PRISM_FINAL_EVAL_CLIP_LENGTH=1
export PRISM_TEMPORAL_ABLATION_CLIP_LENGTHS=1
export PRISM_FIXED_EVAL_FRAME_INDEX=4
export PRISM_MATTE_FRAME_CHUNK_SIZE=1
export PRISM_SAM2_TEMPORAL_CHUNK_SIZE=1
export PRISM_PARTITION_TRAIN_EPOCHS=false
export PRISM_WORKERS=4
export PRISM_EVAL_WORKERS=2
export PRISM_AMP_DTYPE=bfloat16
export PRISM_RUN_FINAL_EVALUATION=true
export PRISM_WANDB_MODE=online
export PRISM_WANDB_ENTITY=sukhuny-seoul-national-university
export PRISM_WANDB_PROJECT=PRISM
export PRISM_WANDB_GROUP=prism-128-independent-ablation
export PRISM_EXPERIMENT_ID="$run_name"
unset PRISM_MAX_TRAIN_BATCHES PRISM_MAX_EVAL_BATCHES PRISM_MAX_FINAL_EVAL_BATCHES
unset PRISM_STAGE2_INITIAL_CHECKPOINT PRISM_STAGE4_INITIAL_CHECKPOINT
unset PRISM_STAGE2_MAM2_DONOR_CHECKPOINT PRISM_STAGE3_BACKGROUND_DONOR_CHECKPOINT
unset PRISM_STAGE4_BACKGROUND_DONOR_CHECKPOINT PRISM_RESTORE_CHECKPOINT_URI

python - <<'PY'
import torch, cv2, diffusers, peft, wandb, sam2, detectron2
from huggingface_hub import hf_hub_download
import os
assert torch.cuda.is_available()
hf_hub_download('black-forest-labs/FLUX.1-Fill-dev', 'model_index.json', token=os.environ['HF_TOKEN'], local_files_only=True)
print('PRISM_SINGLE_FRAME_RUNTIME_READY', torch.__version__, diffusers.__version__, flush=True)
PY
python -m pytest -q \
  "$repo_root/network/tests/test_matter.py" \
  "$repo_root/network/tests/test_pipeline.py" \
  "$repo_root/network/tests/test_background.py" \
  -k 'single_frame_direct_evidence or image_only_flux'
echo "PRISM_SINGLE_FRAME_START commit=$actual_commit data=$data_root"
exec bash "$repo_root/network/scripts/train_prism_all_stages.sh"
