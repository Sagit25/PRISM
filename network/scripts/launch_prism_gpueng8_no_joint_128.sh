#!/usr/bin/env bash
# Independently retrain the 128px no-joint-loop ablation on gpueng8.
set -Eeuo pipefail

home_dir=/home/ruai-u6151
work_root="$home_dir/prism-ablation-scratch-128-20261111"
dataset_root="$home_dir/prism-datasets/128-20261103-merged"
repo_root="$work_root/code"
output_root="$work_root/no_joint_refinement"
python_bin="$home_dir/prism-venv/bin/python"
credential_file="$home_dir/.config/prism/credentials.json"
expected_commit=564df2de052e655ae033558d7de7b3920f1dbaef
expected_manifest=4bcaf6af8025337f63b930b08bf1b5147a6337d2cf215816f767abcd93e39a79

mkdir -p "$output_root/wandb" "$output_root/tmp"
exec 9>"$output_root/.launcher.lock"
flock -n 9 || { echo PRISM_GPUENG8_ALREADY_RUNNING; exit 2; }

for ((attempt=0; attempt<43200; attempt++)); do
  if [[ -f "$work_root/.gpueng8-data-ready" \
    && -f "$work_root/.gpueng8-runtime-ready" ]]; then
    break
  fi
  sleep 60
done
[[ -f "$work_root/.gpueng8-data-ready" && -f "$work_root/.gpueng8-runtime-ready" ]] || {
  echo PRISM_GPUENG8_INPUTS_UNAVAILABLE >&2; exit 3;
}
[[ "$(git -C "$repo_root" rev-parse HEAD)" == "$expected_commit" ]] || {
  echo PRISM_GPUENG8_CODE_SHA_MISMATCH >&2; exit 3;
}
[[ "$(sha256sum "$dataset_root/dataset_manifest.json" | cut -d' ' -f1)" == "$expected_manifest" ]] || {
  echo PRISM_GPUENG8_MANIFEST_SHA_MISMATCH >&2; exit 3;
}
[[ -x "$python_bin" && -f "$credential_file" ]] || {
  echo PRISM_GPUENG8_RUNTIME_OR_CREDENTIALS_MISSING >&2; exit 3;
}
[[ "$(stat -c %a "$credential_file")" == 600 ]] || {
  echo PRISM_GPUENG8_CREDENTIAL_PERMISSIONS_INVALID >&2; exit 3;
}

export PYTHONPATH="$repo_root/network/third_party/sam2:$repo_root/network/third_party/MEMatte${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_BASE_URL=https://api.forge.coreweave.com
export WANDB_DIR="$output_root/wandb"
export TMPDIR="$output_root/tmp"
export PRISM_CREDENTIAL_FILE="$credential_file"
export PRISM_TRAIN_PYTHON="$python_bin"
export PRISM_128_MERGED_ROOT="$dataset_root"
export PRISM_128_GENERATION_ROOT="$dataset_root"
export PRISM_DATA_CONTRACT_STAMP="$output_root/dataset_contract.stamp"
export PRISM_OUTPUT_ROOT="$output_root"
export PRISM_CACHE_ROOT="$home_dir/prism-model-cache"
export PRISM_SAM2_ROOT="$repo_root/network/third_party/sam2"
export PRISM_SAM2_CHECKPOINT="$repo_root/network/checkpoints/sam2.1_hiera_large.pt"
export PRISM_MEMATTE_ROOT="$repo_root/network/third_party/MEMatte"
export PRISM_MEMATTE_CHECKPOINT="$repo_root/network/checkpoints/MEMatte_AIM500.pth"
export PRISM_GPU_INDEX=0
export PRISM_MIN_GPU_GIB=39
export PRISM_MIN_FREE_GPU_GIB=35
export PRISM_EXPECTED_COMMIT="$expected_commit"
export PRISM_ABLATION=no_joint_refinement
export PRISM_EXPERIMENT_ID=prism-gpueng8-128-scratch-no_joint_refinement-20261112
export PRISM_WANDB_GROUP=prism-128-scratch-ablation-20261111
export PRISM_WANDB_ENTITY=sukhuny-seoul-national-university
export PRISM_STAGE1A_EPOCHS=20
export PRISM_STAGE1B_EPOCHS=20
export PRISM_STAGE2_EPOCHS=15
export PRISM_STAGE3_EPOCHS=15
export PRISM_STAGE4_EPOCHS=25
export PRISM_IMAGE_PRETRAIN_CLIP_LENGTH=1
export PRISM_VIDEO_REFINE_CLIP_LENGTH=4
export PRISM_FINAL_EVAL_CLIP_LENGTH=8
export PRISM_TEMPORAL_ABLATION_CLIP_LENGTHS=8
export PRISM_STAGE2_PAIRED_BACKGROUNDS=true
export PRISM_JOINT_PAIRED_BACKGROUNDS=true
export PRISM_PAIRED_MICROBATCH_CHECKPOINTING=true
export PRISM_STAGE1_BATCH_SIZE=2
export PRISM_STAGE2_BATCH_SIZE=4
export PRISM_STAGE34_BATCH_SIZE=4
export PRISM_FINAL_EVAL_BATCH_SIZE=4
export PRISM_FAST_EARLY_STAGES=false
export PRISM_DIFFUSION_CPU_OFFLOAD=on
export PRISM_RUN_FINAL_EVALUATION=true
unset PRISM_STAGE1_MANIFESTS PRISM_STAGE2_INITIAL_CHECKPOINT PRISM_STAGE4_INITIAL_CHECKPOINT
unset PRISM_STAGE2_MAM2_DONOR_CHECKPOINT PRISM_STAGE3_BACKGROUND_DONOR_CHECKPOINT
unset PRISM_STAGE4_BACKGROUND_DONOR_CHECKPOINT PRISM_RESTORE_CHECKPOINT_URI

echo "PRISM_GPUENG8_ABLATION_START variant=no_joint_refinement commit=$expected_commit"
bash "$repo_root/network/scripts/run_prism_v26_128_gpu1.sh"
echo "PRISM_GPUENG8_ABLATION_COMPLETE output=$output_root"
