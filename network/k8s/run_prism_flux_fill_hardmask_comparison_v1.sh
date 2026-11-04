#!/usr/bin/env bash
set -Eeuo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
run_name="${PRISM_BASELINE_RUN_NAME:?PRISM_BASELINE_RUN_NAME is required}"
expected_commit="${PRISM_BASELINE_EXPECTED_COMMIT:?PRISM_BASELINE_EXPECTED_COMMIT is required}"
test_limit="${PRISM_BASELINE_MAX_SEQUENCES:-1}"
data_root=/data/datasets/prism-full-64-materialized-v7r1
output_root="/data/comparisons/prism-test64-v1/$run_name/flux_fill_hardmask"
export PYTHONUNBUFFERED=1 HF_HOME=/data/cache/prism-shared-models/huggingface
export TMPDIR=/data/cache/prism-baseline-tmp PIP_NO_CACHE_DIR=1
mkdir -p "$output_root" "$HF_HOME" "$TMPDIR"
test "${repo_root##*/}" = "PRISM-flux-hardmask-v1-$expected_commit"
test -d "$data_root/test"
test -n "${WANDB_API_KEY:-}"
test -n "${HF_TOKEN:-}"
echo "PRISM_FLUX_HARDMASK_CODE_READY $expected_commit"
python -m pip install --quiet -e './network[data,experiment,evaluation,diffusion]'
python network/scripts/evaluate_flux_fill_hardmask_background.py \
  --data-root "$data_root" --output-root "$output_root" \
  --frame-index 4 --working-side "${PRISM_BASELINE_WORKING_SIDE:-256}" \
  --steps "${PRISM_BASELINE_STEPS:-8}" --max-sequences "$test_limit" \
  --run-name "$run_name-flux-hardmask"
python network/scripts/evaluate_comparison_predictions.py \
  --data-root "$data_root" --prediction-root "$output_root/predictions" \
  --output-root "$output_root" --frame-index 4 --max-sequences "$test_limit" \
  --model-name flux_fill_hardmask --prompt-policy 'oracle frame-4 object hard mask; one RGB image; no flow or temporal evidence'
echo "PRISM_FLUX_HARDMASK_COMPARISON_COMPLETE $output_root"
