#!/usr/bin/env bash
set -Eeuo pipefail
home_dir=/home/ruai-u6151
root="$home_dir/prism-ablation-scratch-128-20261111"
repo="$root/code-identity-transmission-cpuprompt-v7"
output="$root/identity_transmission-cpuprompt-v7"
dataset="$home_dir/prism-datasets/128-20261103-merged"
commit=__COMMIT_SHA__
mkdir -p "$output/wandb" /tmp/prism-gpueng10-u6151
mkdir -p "$output/checkpoints"
if [[ ! -f "$output/dataset_contract.stamp" ]]; then
  cp "$root/identity_transmission-promptwarm-v5/dataset_contract.stamp" "$output/dataset_contract.stamp"
fi
for stage in stage1a stage1b stage2 stage3; do
  if [[ ! -e "$output/checkpoints/$stage" ]]; then
    ln -s "$root/identity_transmission-promptwarm-v5/checkpoints/$stage" \
      "$output/checkpoints/$stage"
  fi
done
chmod 700 /tmp/prism-gpueng10-u6151
exec 9>"$output/.launcher.lock"
flock -n 9 || { echo PRISM_GPUENG10_ALREADY_RUNNING; exit 2; }
test "$(git -C "$repo" rev-parse HEAD)" = "$commit"
test -s "$output/checkpoints/stage3/prism_stage3_best.pt"
test -x "$home_dir/prism-venv/bin/python"
test -f "$home_dir/.config/prism/credentials.json"
"$home_dir/prism-venv/bin/pip" install --no-deps --no-build-isolation -e "$repo/network" > "$output/editable_install.log" 2>&1
source_file="$("$home_dir/prism-venv/bin/python" -c 'import refractive_mam2.train as train; print(train.__file__)')"
test "$source_file" = "$repo/network/src/train.py"
echo "PRISM_GPUENG10_EDITABLE_SOURCE_OK $source_file"
export PRISM_CREDENTIAL_FILE="$home_dir/.config/prism/credentials.json"
export PRISM_TRAIN_PYTHON="$home_dir/prism-venv/bin/python"
export PRISM_128_MERGED_ROOT="$dataset"
export PRISM_128_GENERATION_ROOT="$dataset"
export PRISM_DATA_CONTRACT_STAMP="$output/dataset_contract.stamp"
export PRISM_OUTPUT_ROOT="$output"
export PRISM_CACHE_ROOT="$home_dir/prism-model-cache"
export PRISM_FLUX_PROMPT_CACHE="$root/flux_prompt_cpu_v6.pt"
if [[ ! -s "$PRISM_FLUX_PROMPT_CACHE" ]]; then
  model="$PRISM_CACHE_ROOT/huggingface/hub/models--black-forest-labs--FLUX.1-Fill-dev/snapshots/358293da0354175698b67ec8299acf928313a78a"
  CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 "$PRISM_TRAIN_PYTHON" \
    "$repo/network/scripts/precompute_flux_prompt_cpu.py" \
    --model "$model" --output "$PRISM_FLUX_PROMPT_CACHE"
fi
test -s "$PRISM_FLUX_PROMPT_CACHE"
export PRISM_SAM2_ROOT="$repo/network/third_party/sam2"
export PRISM_SAM2_CHECKPOINT="$repo/network/checkpoints/sam2.1_hiera_large.pt"
export PRISM_MEMATTE_ROOT="$repo/network/third_party/MEMatte"
export PRISM_MEMATTE_CHECKPOINT="$repo/network/checkpoints/MEMatte_AIM500.pth"
export PYTHONPATH="$repo/network/third_party/sam2:$repo/network/third_party/MEMatte${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_BASE_URL=https://api.forge.coreweave.com
export WANDB_DIR="$output/wandb"
export TMPDIR=/tmp/prism-gpueng10-u6151
export PRISM_GPU_INDEX=0
export PRISM_MIN_GPU_GIB=39
export PRISM_MIN_FREE_GPU_GIB=35
export PRISM_EXPECTED_COMMIT="$commit"
export PRISM_ABLATION=identity_transmission
export PRISM_EXPERIMENT_ID=prism-gpueng10-128-scratch-identity_transmission-cpuprompt-v7-20261114
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
export PRISM_PAIRED_MICROBATCH_CHECKPOINTING=false
export PRISM_STAGE1_BATCH_SIZE=2
export PRISM_STAGE2_BATCH_SIZE=4
export PRISM_STAGE34_BATCH_SIZE=2
export PRISM_FINAL_EVAL_BATCH_SIZE=2
export PRISM_FAST_EARLY_STAGES=false
export PRISM_DIFFUSION_CPU_OFFLOAD=sequential
export PRISM_RUN_FINAL_EVALUATION=true
unset PRISM_STAGE1_MANIFESTS PRISM_STAGE2_INITIAL_CHECKPOINT PRISM_STAGE4_INITIAL_CHECKPOINT
unset PRISM_STAGE2_MAM2_DONOR_CHECKPOINT PRISM_STAGE3_BACKGROUND_DONOR_CHECKPOINT
unset PRISM_STAGE4_BACKGROUND_DONOR_CHECKPOINT PRISM_RESTORE_CHECKPOINT_URI
echo "PRISM_GPUENG10_RECOVERY_START commit=$commit from_stage3=$output/checkpoints/stage3/prism_stage3_best.pt"
bash "$repo/network/scripts/run_prism_v26_128_gpu1.sh"
echo "PRISM_GPUENG10_RECOVERY_COMPLETE output=$output"
