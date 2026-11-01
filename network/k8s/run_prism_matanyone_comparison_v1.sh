#!/usr/bin/env bash
set -Eeuo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
run_name="${PRISM_BASELINE_RUN_NAME:?PRISM_BASELINE_RUN_NAME is required}"
expected_commit="${PRISM_BASELINE_EXPECTED_COMMIT:?PRISM_BASELINE_EXPECTED_COMMIT is required}"
test_limit="${PRISM_BASELINE_MAX_SEQUENCES:-16}"
data_root=/data/datasets/prism-full-64-materialized-v7r1
output_root="/data/comparisons/prism-test64-v1/$run_name/matanyone"
source_root=/data/code/MatAnyone-official-baseline
export PYTHONUNBUFFERED=1 OPENCV_IO_ENABLE_OPENEXR=1
export HF_HOME=/data/cache/prism-baseline-hf
export TMPDIR=/data/cache/prism-baseline-tmp
export PIP_NO_CACHE_DIR=1
mkdir -p "$output_root" /data/code "$HF_HOME" "$TMPDIR"
apt-get update -qq
apt-get install -y -qq git libgl1 libglib2.0-0
test "$(git rev-parse HEAD)" = "$expected_commit"
test -d "$data_root/test"
test -n "${WANDB_API_KEY:-}"
echo "PRISM_MATANYONE_COMPARISON_CODE_READY $expected_commit"
python -m pip install --quiet -e './network[data,sam2,experiment,evaluation]'
python -m pip install --quiet imageio imageio-ffmpeg huggingface-hub scikit-image
sam2_root=/data/code/SAM2-official-baseline
if [ ! -d "$sam2_root/.git" ]; then
  git clone --no-checkout https://github.com/facebookresearch/sam2.git "$sam2_root"
fi
git -C "$sam2_root" fetch origin 2b90b9f5ceec907a1c18123530e92e794ad901a4
git -C "$sam2_root" checkout --detach 2b90b9f5ceec907a1c18123530e92e794ad901a4
test "$(git -C "$sam2_root" rev-parse HEAD)" = 2b90b9f5ceec907a1c18123530e92e794ad901a4
export PRISM_SAM2_ROOT="$sam2_root"
if [ ! -d "$source_root/.git" ]; then
  git clone --no-checkout https://github.com/pq-yang/MatAnyone.git "$source_root"
fi
git -C "$source_root" fetch origin e5ddc534c1fff9bb9e54cf476095d29071b7cb4f
git -C "$source_root" checkout --detach e5ddc534c1fff9bb9e54cf476095d29071b7cb4f
test "$(git -C "$source_root" rev-parse HEAD)" = e5ddc534c1fff9bb9e54cf476095d29071b7cb4f
python network/scripts/evaluate_official_matanyone.py \
  --data-root "$data_root" --source-root "$source_root" --output-root "$output_root" \
  --clip-length 8 --frame-index 4 --max-sequences "$test_limit" \
  --run-name "$run_name-matanyone"
python network/scripts/evaluate_comparison_predictions.py \
  --data-root "$data_root" --prediction-root "$output_root/predictions" \
  --output-root "$output_root" --frame-index 4 --max-sequences "$test_limit" \
  --model-name matanyone_official --prompt-policy 'oracle first-frame object mask'
echo "PRISM_MATANYONE_COMPARISON_COMPLETE $output_root"
