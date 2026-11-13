#!/usr/bin/env bash
# Train one independently removed 128px ablation on the assigned A100 GPU.
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
server_home="${PRISM_GPUENG_HOME:-/home/ruai-u6151}"
gpu_name="${PRISM_GPUENG_NAME:-gpueng10}"
case "$gpu_name" in
  gpueng9|gpueng10|gpueng11|gpueng14|gpueng16) ;;
  *) echo "PRISM_GPUENG_FAIL: unexpected GPU name=$gpu_name" >&2; exit 2 ;;
esac
dataset_root="${PRISM_128_MERGED_ROOT:-$server_home/prism-datasets/128-20261103-merged}"
work_root="${PRISM_ABLATION_QUEUE_ROOT:-$server_home/prism-ablation-scratch-128-20261111}"
variant="${PRISM_ABLATION:-single_frame}"
run_tag="${PRISM_ABLATION_RUN_TAG:-}"
if [[ -n "$run_tag" && ! "$run_tag" =~ ^[a-z0-9-]+$ ]]; then
  echo "PRISM_GPUENG_FAIL: invalid run tag=$run_tag" >&2
  exit 2
fi
case "$variant" in
  no_inverse|no_joint_refinement|scalar_transmission|identity_transmission|no_residual|identity_flow+identity_transmission|identity_flow+identity_transmission+no_residual|single_frame|no_pair|single_image_no_pair) ;;
  *) echo "PRISM_GPUENG_FAIL: unexpected independent variant=$variant" >&2; exit 2 ;;
esac

output_root="$work_root/${variant//+/_}${run_tag:+-$run_tag}"
mkdir -p "$output_root"
exec 9>"$output_root/.launcher.lock"
if ! flock -n 9; then
  echo "PRISM_GPUENG_FAIL: this variant already has a launcher" >&2
  exit 2
fi

while [[ ! -f "$dataset_root/.transfer-complete" || ! -f "$work_root/.assets-complete" ]]; do
  echo "PRISM_GPUENG_WAIT_FOR_DATA gpu=$gpu_name variant=$variant"
  sleep 60
done

python_bin="$server_home/prism-venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
  echo "PRISM_GPUENG_FAIL: Python environment is missing" >&2
  exit 2
fi
export OPENCV_IO_ENABLE_OPENEXR=1
stamp="$output_root/dataset_contract.stamp"
expected_stamp="$dataset_root $(sha256sum "$dataset_root/dataset_manifest.json" | cut -d' ' -f1)"
verified_stamp="${PRISM_VERIFIED_DATA_CONTRACT_STAMP:-}"
if [[ -n "$verified_stamp" && -f "$verified_stamp" && "$(cat "$verified_stamp")" == "$expected_stamp" ]]; then
  echo "PRISM_GPUENG_REUSE_VERIFIED_CONTRACT $verified_stamp"
else
  "$python_bin" "$repo_root/RCDatasetCreation/tools/freeze_prism_manifest.py" "$dataset_root" --verify
  for split in train validation test; do
    "$python_bin" "$repo_root/RCDatasetCreation/tools/validate_prism_contract.py" "$dataset_root/$split"
  done
fi
printf '%s\n' "$expected_stamp" > "$stamp"
echo "PRISM_GPUENG_DATA_CONTRACT_OK gpu=$gpu_name root=$dataset_root"

export WANDB_BASE_URL=https://api.forge.coreweave.com
export WANDB_DIR="$output_root/wandb"
export TMPDIR="${PRISM_GPUENG_SHORT_TMPDIR:-/tmp/prism-${gpu_name}-${variant//+/_}${run_tag:+-$run_tag}}"
if [[ "$TMPDIR" != /* || ${#TMPDIR} -gt 65 ]]; then
  echo "PRISM_GPUENG_FAIL: TMPDIR must be an absolute path at most 65 characters" >&2
  exit 2
fi
mkdir -p "$WANDB_DIR" "$TMPDIR"
chmod 700 "$TMPDIR"
export PRISM_CREDENTIAL_FILE="$server_home/.config/prism/credentials.json"
export PRISM_TRAIN_PYTHON="$python_bin"
export PRISM_128_MERGED_ROOT="$dataset_root"
export PRISM_128_GENERATION_ROOT="$dataset_root"
export PRISM_DATA_CONTRACT_STAMP="$stamp"
export PRISM_OUTPUT_ROOT="$output_root"
export PRISM_CACHE_ROOT="$server_home/prism-model-cache"
export PRISM_SAM2_ROOT="$repo_root/network/third_party/sam2"
export PRISM_SAM2_CHECKPOINT="$repo_root/network/checkpoints/sam2.1_hiera_large.pt"
export PRISM_MEMATTE_ROOT="$repo_root/network/third_party/MEMatte"
export PRISM_MEMATTE_CHECKPOINT="$repo_root/network/checkpoints/MEMatte_AIM500.pth"
export PRISM_GPU_INDEX=0
export PRISM_MIN_GPU_GIB=39
export PRISM_MIN_FREE_GPU_GIB=35
export PRISM_EXPECTED_COMMIT="${PRISM_EXPECTED_COMMIT:-$(git -C "$repo_root" rev-parse HEAD)}"
export PRISM_ABLATION="$variant"
export PRISM_EXPERIMENT_ID="prism-${gpu_name}-128-scratch-${variant//+/_}${run_tag:+-$run_tag}-20261111"
export PRISM_WANDB_GROUP=prism-128-scratch-ablation-20261111
export PRISM_WANDB_ENTITY=sukhuny-seoul-national-university
export PRISM_STAGE1A_EPOCHS=20
export PRISM_STAGE1B_EPOCHS=20
export PRISM_STAGE2_EPOCHS=15
export PRISM_STAGE3_EPOCHS=15
export PRISM_STAGE4_EPOCHS=25
export PRISM_IMAGE_PRETRAIN_CLIP_LENGTH=1
if [[ "$variant" == "single_image_no_pair" || "$variant" == "no_pair" ]]; then
  export PRISM_STAGE2_PAIRED_BACKGROUNDS=false
  export PRISM_JOINT_PAIRED_BACKGROUNDS=false
  export PRISM_PAIRED_MICROBATCH_CHECKPOINTING=false
else
  export PRISM_STAGE2_PAIRED_BACKGROUNDS=true
  export PRISM_JOINT_PAIRED_BACKGROUNDS=true
  export PRISM_PAIRED_MICROBATCH_CHECKPOINTING=true
fi
if [[ "$variant" == "single_frame" || "$variant" == "single_image_no_pair" ]]; then
  export PRISM_VIDEO_REFINE_CLIP_LENGTH=1
  export PRISM_FINAL_EVAL_CLIP_LENGTH=1
  export PRISM_TEMPORAL_ABLATION_CLIP_LENGTHS=1
  export PRISM_FIXED_EVAL_FRAME_INDEX=4
else
  export PRISM_VIDEO_REFINE_CLIP_LENGTH=4
  export PRISM_FINAL_EVAL_CLIP_LENGTH=8
  export PRISM_TEMPORAL_ABLATION_CLIP_LENGTHS=8
  unset PRISM_FIXED_EVAL_FRAME_INDEX
fi
export PRISM_STAGE1_BATCH_SIZE="${PRISM_STAGE1_BATCH_SIZE:-2}"
export PRISM_STAGE2_BATCH_SIZE="${PRISM_STAGE2_BATCH_SIZE:-4}"
export PRISM_STAGE34_BATCH_SIZE="${PRISM_STAGE34_BATCH_SIZE:-4}"
export PRISM_FINAL_EVAL_BATCH_SIZE="${PRISM_FINAL_EVAL_BATCH_SIZE:-4}"
export PRISM_FAST_EARLY_STAGES="${PRISM_FAST_EARLY_STAGES:-false}"
export PRISM_DIFFUSION_CPU_OFFLOAD="${PRISM_DIFFUSION_CPU_OFFLOAD:-on}"
export PRISM_RUN_FINAL_EVALUATION=true
unset PRISM_STAGE1_MANIFESTS PRISM_STAGE2_INITIAL_CHECKPOINT PRISM_STAGE4_INITIAL_CHECKPOINT
unset PRISM_STAGE2_MAM2_DONOR_CHECKPOINT PRISM_STAGE3_BACKGROUND_DONOR_CHECKPOINT
unset PRISM_STAGE4_BACKGROUND_DONOR_CHECKPOINT PRISM_RESTORE_CHECKPOINT_URI

echo "PRISM_GPUENG_ABLATION_START gpu=$gpu_name variant=$variant commit=$PRISM_EXPECTED_COMMIT output=$output_root"
bash "$script_dir/run_prism_v26_128_gpu1.sh"
echo "PRISM_GPUENG_ABLATION_COMPLETE gpu=$gpu_name variant=$variant output=$output_root"
