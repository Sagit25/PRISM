#!/usr/bin/env bash
set -Eeuo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
run_name="${PRISM_BASELINE_RUN_NAME:?PRISM_BASELINE_RUN_NAME is required}"
expected_commit="${PRISM_BASELINE_EXPECTED_COMMIT:?PRISM_BASELINE_EXPECTED_COMMIT is required}"
test_limit="${PRISM_BASELINE_MAX_SEQUENCES:-1}"
data_root=/data/datasets/prism-full-64-materialized-v7r1
output_root="/data/comparisons/prism-test64-v1/$run_name/lama_port"
export PYTHONUNBUFFERED=1 HF_HOME=/data/cache/prism-baseline-hf
export TORCH_HOME=/data/cache/prism-baseline-torch
export TMPDIR=/data/cache/prism-baseline-tmp PIP_NO_CACHE_DIR=1
mkdir -p "$output_root" "$TORCH_HOME/hub/checkpoints" "$TMPDIR"
test "$(git rev-parse HEAD)" = "$expected_commit"
test -d "$data_root/test"
test -n "${WANDB_API_KEY:-}"
echo "PRISM_LAMA_PORT_CODE_READY $expected_commit"
python -m pip install --quiet -e './network[data,experiment,evaluation]'
python -m pip install --quiet 'simple-lama-inpainting==0.1.2'
python network/scripts/evaluate_lama_port_background.py \
  --data-root "$data_root" --output-root "$output_root" \
  --frame-index 4 --working-side 256 --max-sequences "$test_limit" \
  --run-name "$run_name-lama-port"
python network/scripts/evaluate_comparison_predictions.py \
  --data-root "$data_root" --prediction-root "$output_root/predictions" \
  --output-root "$output_root" --frame-index 4 --max-sequences "$test_limit" \
  --model-name lama_torchscript_port --prompt-policy 'oracle frame-4 object mask; one RGB image'
echo "PRISM_LAMA_PORT_COMPARISON_COMPLETE $output_root"
