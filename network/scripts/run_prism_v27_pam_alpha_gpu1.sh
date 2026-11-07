#!/usr/bin/env bash
# Warm-start a bounded PAM alpha correction from the separate v26 Stage 4
# model. The donor checkpoint and all v26 training files remain untouched.
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
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
