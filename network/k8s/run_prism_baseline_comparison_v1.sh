#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
run_name="${PRISM_BASELINE_RUN_NAME:?PRISM_BASELINE_RUN_NAME is required}"
expected_commit="${PRISM_BASELINE_EXPECTED_COMMIT:?PRISM_BASELINE_EXPECTED_COMMIT is required}"
test_limit="${PRISM_BASELINE_MAX_SEQUENCES:-16}"
data_root=/data/datasets/prism-full-64-materialized-v7r1
output_root="/data/comparisons/prism-test64-v1/$run_name"
source_root=/data/code/SAM2Matting-official-baseline
checkpoint=/data/models/SAM2Matting-SAM2.1Base+.pt
export PYTHONUNBUFFERED=1 OPENCV_IO_ENABLE_OPENEXR=1
export HF_HOME=/data/cache/prism-baseline-hf
export HF_HUB_ETAG_TIMEOUT=60 HF_HUB_DOWNLOAD_TIMEOUT=120
mkdir -p "$HF_HOME" "$output_root" /data/models /data/code
apt-get update -qq
apt-get install -y -qq git libgl1 libglib2.0-0
test "$(git rev-parse HEAD)" = "$expected_commit"
test -d "$data_root/test"
test -n "${WANDB_API_KEY:-}"
echo "PRISM_BASELINE_CODE_READY $expected_commit"

python -m pip install --quiet -e './network[data,sam2,experiment,evaluation]'
python -m pip install --quiet timm einops ftfy regex loguru fvcore scikit-image
PYTHON_BIN=python bash network/scripts/install_official_sam2.sh \
  /data/code/SAM2-official-baseline 2b90b9f5ceec907a1c18123530e92e794ad901a4 none
python - <<'PY'
import torch, torchvision
assert torch.__version__.startswith('2.5.1'), torch.__version__
assert torchvision.__version__.startswith('0.20.1'), torchvision.__version__
assert torch.cuda.is_available()
print('PRISM_BASELINE_RUNTIME_OK', torch.cuda.get_device_name(0), flush=True)
PY

python network/scripts/evaluate_classical_backgrounds.py \
  --data-root "$data_root" --output-root "$output_root" \
  --clip-length 8 --frame-index 4 --max-sequences "$test_limit"
for method in telea_oracle_mask temporal_visible_oracle_masks; do
  python network/scripts/evaluate_comparison_predictions.py \
    --data-root "$data_root" --prediction-root "$output_root/$method/predictions" \
    --output-root "$output_root/$method" --frame-index 4 --max-sequences "$test_limit" \
    --model-name "$method" --prompt-policy 'oracle object masks'
done

if [ ! -d "$source_root/.git" ]; then
  git clone --no-checkout https://github.com/FudanCVL/SAM2Matting.git "$source_root"
fi
git -C "$source_root" fetch origin 73dd721d77b56749248aefe5e8824d7f61b9d13c
git -C "$source_root" checkout --detach 73dd721d77b56749248aefe5e8824d7f61b9d13c
test "$(git -C "$source_root" rev-parse HEAD)" = 73dd721d77b56749248aefe5e8824d7f61b9d13c
python - <<'PY'
from huggingface_hub import hf_hub_download
from pathlib import Path
import hashlib, shutil
target=Path('/data/models/SAM2Matting-SAM2.1Base+.pt')
expected='1f0eb2eda3e8bc9101eafc0b30b8b8fcae1ff83d8fd3adc18e2f3b410fdaae60'
if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != expected:
    source=Path(hf_hub_download(repo_id='FudanCVL/SAM2Matting', filename='checkpoints/SAM2Matting-SAM2.1Base+.pt', revision='4315db9c60d27fde396b09765748a0ca6c97bed5'))
    shutil.copyfile(source, target)
actual=hashlib.sha256(target.read_bytes()).hexdigest()
if actual != expected:
    raise RuntimeError(f'official SAM2Matting checksum mismatch: {actual}')
print('PRISM_BASELINE_SAM2_WEIGHTS_OK', actual, flush=True)
PY
python network/scripts/evaluate_official_sam2matting.py \
  --data-root "$data_root" --output-root "$output_root/sam2matting_baseplus" \
  --source-root "$source_root" --checkpoint "$checkpoint" \
  --variant base_plus --split test --clip-length 8 --preview-frame-index 4 \
  --max-sequences "$test_limit" --run-name "$run_name-sam2matting"
python network/scripts/evaluate_comparison_predictions.py \
  --data-root "$data_root" --prediction-root "$output_root/sam2matting_baseplus/predictions" \
  --output-root "$output_root/sam2matting_baseplus" --frame-index 4 \
  --max-sequences "$test_limit" --model-name sam2matting_baseplus \
  --prompt-policy 'oracle first-frame mask only'
echo "PRISM_BASELINE_COMPARISON_COMPLETE $output_root"
