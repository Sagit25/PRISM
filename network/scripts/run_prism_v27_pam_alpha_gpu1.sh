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
export PRISM_WANDB_GROUP="${PRISM_WANDB_GROUP:-prism-v27-pam-alpha-128}"
export PRISM_EXPERIMENT_ID="${PRISM_EXPERIMENT_ID:-prism-gpu1-v27-pam-alpha-128}"

if [[ ! -s "$PRISM_STAGE4_INITIAL_CHECKPOINT" ]]; then
  echo "PRISM_V27_DONOR_MISSING checkpoint=$PRISM_STAGE4_INITIAL_CHECKPOINT" >&2
  exit 2
fi
if [[ "$PRISM_OUTPUT_ROOT" == /home/sukhun/prism-training/v26-128-full* ]]; then
  echo "PRISM_V27_OUTPUT_CONFLICT: v26 output must remain intact" >&2
  exit 2
fi
exec bash "$script_dir/run_prism_v26_128_gpu1.sh"
