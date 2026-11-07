#!/usr/bin/env bash
# Warm-start a bounded PAM alpha correction from the separate v26 Stage 4
# model. The donor checkpoint and all v26 training files remain untouched.
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
donor_checkout="${PRISM_V26_CHECKOUT:-/home/sukhun/prism-training-v26-128-20261103}"
export PRISM_TRAIN_PYTHON="${PRISM_TRAIN_PYTHON:-$donor_checkout/.venv/bin/python}"
export PRISM_SAM2_ROOT="${PRISM_SAM2_ROOT:-$donor_checkout/network/third_party/sam2}"
export PRISM_SAM2_CHECKPOINT="${PRISM_SAM2_CHECKPOINT:-$donor_checkout/network/checkpoints/sam2.1_hiera_large.pt}"
export PRISM_MEMATTE_ROOT="${PRISM_MEMATTE_ROOT:-$donor_checkout/network/third_party/MEMatte}"
export PRISM_MEMATTE_CHECKPOINT="${PRISM_MEMATTE_CHECKPOINT:-$donor_checkout/network/checkpoints/MEMatte_AIM500.pth}"
# Reuse installed CUDA packages without changing the v26 environment's
# editable package registration while that process is still running.
mkdir -p "$repo_root/.runtime"
if [[ ! -e "$repo_root/.runtime/refractive_mam2" ]]; then
  ln -s "$repo_root/network/src" "$repo_root/.runtime/refractive_mam2"
fi
export PYTHONPATH="$repo_root/.runtime${PYTHONPATH:+:$PYTHONPATH}"
export PRISM_OUTPUT_ROOT="${PRISM_OUTPUT_ROOT:-/home/sukhun/prism-training/v27-128-pam-alpha}"
export PRISM_STAGE4_INITIAL_CHECKPOINT="${PRISM_STAGE4_INITIAL_CHECKPOINT:-/home/sukhun/prism-training/v26-128-full/checkpoints/stage4/prism_stage4_best.pt}"
export PRISM_PAM_ALPHA_STAGE4_ONLY=true
export PRISM_PAM_ALPHA_RESIDUAL=true
export PRISM_PAM_ALPHA_MAX_DELTA="${PRISM_PAM_ALPHA_MAX_DELTA:-0.25}"
export PRISM_LOSS_PROFILE=pam_alpha_refinement
export PRISM_STAGE4_EPOCHS="${PRISM_STAGE4_EPOCHS:-10}"
stage5_epochs="${PRISM_STAGE5_EPOCHS:-8}"
export PRISM_WANDB_GROUP="${PRISM_WANDB_GROUP:-prism-v27-pam-alpha-128}"
export PRISM_EXPERIMENT_ID="${PRISM_EXPERIMENT_ID:-prism-gpu1-v27-pam-alpha-128}"

if [[ ! -s "$PRISM_STAGE4_INITIAL_CHECKPOINT" ]]; then
  echo "PRISM_V27_DONOR_MISSING checkpoint=$PRISM_STAGE4_INITIAL_CHECKPOINT" >&2
  exit 2
fi
"$PRISM_TRAIN_PYTHON" "$script_dir/verify_prism_stage_checkpoint.py" \
  "$PRISM_STAGE4_INITIAL_CHECKPOINT" \
  --expect arguments stage 4 \
  --expect matter fixed_mam2_alpha true \
  --expect matter use_mam2_foreground_prior true \
  --expect background evidence_flux_only true
if [[ "$PRISM_OUTPUT_ROOT" == /home/sukhun/prism-training/v26-128-full* ]]; then
  echo "PRISM_V27_OUTPUT_CONFLICT: v26 output must remain intact" >&2
  exit 2
fi
stage4_output="$PRISM_OUTPUT_ROOT"
echo "PRISM_V27_STAGE4_WARMSTART checkpoint=$PRISM_STAGE4_INITIAL_CHECKPOINT optimizer=fresh"
bash "$script_dir/run_prism_v26_128_gpu1.sh"

stage4_best="$stage4_output/checkpoints/stage4/prism_stage4_best.pt"
if [[ ! -s "$stage4_best" \
  || ! -f "$stage4_output/checkpoints/stage4/.training_complete" ]]; then
  echo "PRISM_V27_STAGE5_BLOCKED: Stage 4 model is not complete" >&2
  exit 3
fi
"$PRISM_TRAIN_PYTHON" "$script_dir/verify_prism_stage_checkpoint.py" \
  "$stage4_best" \
  --expect arguments stage 4 \
  --expect matter fixed_mam2_alpha true \
  --expect matter pam_alpha_residual true \
  --expect background evidence_flux_only true
unset PRISM_PAM_ALPHA_STAGE4_ONLY
export PRISM_GEOMETRY_POST_HEAD_ONLY=true
export PRISM_STAGE4_INITIAL_CHECKPOINT="$stage4_best"
export PRISM_OUTPUT_ROOT="$stage4_output/stage5-geometry"
export PRISM_EXPERIMENT_ID="${PRISM_EXPERIMENT_ID}-geometry"
export PRISM_STAGE4_EPOCHS="$stage5_epochs"
export PRISM_SAVE_REFINEMENT_TRACE=false
echo "PRISM_V27_STAGE5_PREPARED donor=$stage4_best epochs=$stage5_epochs"
bash "$script_dir/run_prism_v26_128_gpu1.sh"
echo "PRISM_V27_STAGE45_COMPLETE stage4=$stage4_best stage5=$PRISM_OUTPUT_ROOT/prism_stage5_best.pt"
