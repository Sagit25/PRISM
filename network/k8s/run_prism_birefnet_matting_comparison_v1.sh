#!/usr/bin/env bash
set -Eeuo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
run_name="${PRISM_BASELINE_RUN_NAME:?PRISM_BASELINE_RUN_NAME is required}"
expected_commit="${PRISM_BASELINE_EXPECTED_COMMIT:?PRISM_BASELINE_EXPECTED_COMMIT is required}"
test_limit="${PRISM_BASELINE_MAX_SEQUENCES:-1}"
data_root=/data/datasets/prism-full-64-materialized-v7r1
output_root="/data/comparisons/prism-test64-v1/$run_name/birefnet_matting"
export PYTHONUNBUFFERED=1 HF_HOME=/data/cache/prism-baseline-hf
export TMPDIR=/data/cache/prism-baseline-tmp PIP_NO_CACHE_DIR=1
mkdir -p "$output_root" "$HF_HOME" "$TMPDIR"
test "${repo_root##*/}" = "PRISM-birefnet-matting-v1-$expected_commit"
test -d "$data_root/test"
test -n "${WANDB_API_KEY:-}"
echo "PRISM_BIREFNET_CODE_READY $expected_commit"
python -m pip install --quiet -e './network[data,experiment,evaluation]'
python -m pip install --quiet 'transformers==4.46.3' 'timm>=1.0,<2' accelerate
python network/scripts/evaluate_official_birefnet_matting.py \
  --data-root "$data_root" --output-root "$output_root" \
  --frame-index 4 --working-side 1024 --max-sequences "$test_limit" \
  --run-name "$run_name-birefnet-matting"
python network/scripts/evaluate_comparison_predictions.py \
  --data-root "$data_root" --prediction-root "$output_root/predictions" \
  --output-root "$output_root" --frame-index 4 --max-sequences "$test_limit" \
  --model-name birefnet_matting_official --prompt-policy 'none; one RGB image, frame 4'
echo "PRISM_BIREFNET_COMPARISON_COMPLETE $output_root"
