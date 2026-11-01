#!/usr/bin/env bash
set -Eeuo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
run_name="${PRISM_BASELINE_RUN_NAME:?PRISM_BASELINE_RUN_NAME is required}"
expected_commit="${PRISM_BASELINE_EXPECTED_COMMIT:?PRISM_BASELINE_EXPECTED_COMMIT is required}"
test_limit="${PRISM_BASELINE_MAX_SEQUENCES:-1}"
data_root=/data/datasets/prism-full-64-materialized-v7r1
output_root="/data/comparisons/prism-test64-v1/$run_name/rose"
source_root=/data/code/ROSE-official-baseline
base_root=/data/cache/prism-baseline-hf/rose-wan-base-9bcbbcf673c0e6a996df6f3f25bee596ffa36d47
weights_root=/data/cache/prism-baseline-hf/rose-weights-8a5e57c9f25e73a4b62d324719cdf3367c13df59
export PYTHONUNBUFFERED=1 HF_HOME=/data/cache/prism-baseline-hf/hub
export TMPDIR=/data/cache/prism-baseline-tmp PIP_NO_CACHE_DIR=1
mkdir -p "$output_root" /data/code "$HF_HOME" "$TMPDIR" "$base_root" "$weights_root/transformer"
exec > >(tee -a "$output_root/job.log") 2>&1
trap 'status=$?; echo "PRISM_ROSE_COMPARISON_EXIT status=$status line=$LINENO"; exit "$status"' EXIT
apt-get update -qq
apt-get install -y -qq git libgl1 libglib2.0-0 ffmpeg
test "$(git rev-parse HEAD)" = "$expected_commit"
test -d "$data_root/test"
test -n "${WANDB_API_KEY:-}"
echo "PRISM_ROSE_COMPARISON_CODE_READY $expected_commit"
python -m pip install --quiet -e './network[data,sam2,experiment,evaluation]'
python -m pip install --quiet 'diffusers==0.31.0' 'transformers==4.46.3' accelerate omegaconf einops timm tomesd torchdiffeq torchsde decord datasets albumentations 'imageio[ffmpeg,pyav]' tensorboard beautifulsoup4 ftfy func_timeout sentencepiece
sam2_root=/data/code/SAM2-official-baseline
if [ ! -d "$sam2_root/.git" ]; then git clone --no-checkout https://github.com/facebookresearch/sam2.git "$sam2_root"; fi
git -C "$sam2_root" fetch origin 2b90b9f5ceec907a1c18123530e92e794ad901a4
git -C "$sam2_root" checkout --detach 2b90b9f5ceec907a1c18123530e92e794ad901a4
test "$(git -C "$sam2_root" rev-parse HEAD)" = 2b90b9f5ceec907a1c18123530e92e794ad901a4
export PRISM_SAM2_ROOT="$sam2_root"
if [ ! -d "$source_root/.git" ]; then git clone --no-checkout https://github.com/Kunbyte-AI/ROSE.git "$source_root"; fi
git -C "$source_root" fetch origin 6be41c5420bf331c6d491277d5a6feaf9b3a779a
git -C "$source_root" checkout --detach 6be41c5420bf331c6d491277d5a6feaf9b3a779a
test "$(git -C "$source_root" rev-parse HEAD)" = 6be41c5420bf331c6d491277d5a6feaf9b3a779a
python - "$base_root" "$weights_root" <<'PY'
from huggingface_hub import snapshot_download
from pathlib import Path
import sys
base, weights = map(Path, sys.argv[1:])
snapshot_download('alibaba-pai/Wan2.1-Fun-1.3B-InP', revision='9bcbbcf673c0e6a996df6f3f25bee596ffa36d47', local_dir=base)
snapshot_download('Kunbyte/ROSE', revision='8a5e57c9f25e73a4b62d324719cdf3367c13df59', local_dir=weights / 'transformer')
assert (base / 'config.json').is_file()
assert (weights / 'transformer' / 'diffusion_pytorch_model.safetensors').is_file()
print('PRISM_ROSE_OFFICIAL_WEIGHTS_READY', flush=True)
PY
python network/scripts/evaluate_official_rose.py \
  --data-root "$data_root" --source-root "$source_root" --base-root "$base_root" \
  --weights-root "$weights_root" --output-root "$output_root" \
  --frame-index 4 --working-side 256 --inference-steps 50 \
  --max-sequences "$test_limit" --run-name "$run_name-rose"
python network/scripts/evaluate_comparison_predictions.py \
  --data-root "$data_root" --prediction-root "$output_root/predictions" \
  --output-root "$output_root" --frame-index 4 --max-sequences "$test_limit" \
  --model-name rose_official --prompt-policy 'oracle per-frame object masks; final frame repeated to 17 frames'
echo "PRISM_ROSE_COMPARISON_COMPLETE $output_root"
