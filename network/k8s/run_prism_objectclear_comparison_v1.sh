#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
run_name="${PRISM_BASELINE_RUN_NAME:?PRISM_BASELINE_RUN_NAME is required}"
expected_commit="${PRISM_BASELINE_EXPECTED_COMMIT:?PRISM_BASELINE_EXPECTED_COMMIT is required}"
test_limit="${PRISM_BASELINE_MAX_SEQUENCES:-16}"
data_root=/data/datasets/prism-full-64-materialized-v7r1
output_root="/data/comparisons/prism-test64-v1/$run_name/objectclear"
source_root=/data/code/ObjectClear-official-baseline
export PYTHONUNBUFFERED=1 OPENCV_IO_ENABLE_OPENEXR=1
export HF_HOME=/data/cache/prism-baseline-hf
export HF_HUB_ETAG_TIMEOUT=60 HF_HUB_DOWNLOAD_TIMEOUT=120
mkdir -p "$HF_HOME" "$output_root" /data/code
apt-get update -qq
apt-get install -y -qq git libgl1 libglib2.0-0
test "$(git rev-parse HEAD)" = "$expected_commit"
test -d "$data_root/test"
test -n "${WANDB_API_KEY:-}"
echo "PRISM_OBJECTCLEAR_COMPARISON_CODE_READY $expected_commit"

python -m pip install --quiet -e './network[data,experiment,evaluation,objectclear]'
PYTHON_BIN=python bash network/scripts/install_official_sam2.sh \
  /data/code/SAM2-official-baseline 2b90b9f5ceec907a1c18123530e92e794ad901a4 none
if [ ! -d "$source_root/.git" ]; then
  git clone --no-checkout https://github.com/zjx0101/ObjectClear.git "$source_root"
fi
git -C "$source_root" fetch origin c052d91ecd8772744a5ab97527441c813ab83009
git -C "$source_root" checkout --detach c052d91ecd8772744a5ab97527441c813ab83009
git -C "$source_root" reset --hard c052d91ecd8772744a5ab97527441c813ab83009
test "$(git -C "$source_root" rev-parse HEAD)" = c052d91ecd8772744a5ab97527441c813ab83009
python network/scripts/patch_objectclear_bf16.py "$source_root"
python - <<'PY'
import torch, torchvision
assert torch.__version__.startswith('2.5.1'), torch.__version__
assert torchvision.__version__.startswith('0.20.1'), torchvision.__version__
assert torch.cuda.is_available()
print('PRISM_OBJECTCLEAR_COMPARISON_RUNTIME_OK', torch.cuda.get_device_name(0), flush=True)
PY
python network/scripts/evaluate_official_objectclear.py \
  --data-root "$data_root" --output-root "$output_root" --source-root "$source_root" \
  --clip-length 8 --frame-index 4 --max-sequences "$test_limit" --steps 20 \
  --run-name "$run_name-objectclear"
python network/scripts/evaluate_comparison_predictions.py \
  --data-root "$data_root" --prediction-root "$output_root/predictions" \
  --output-root "$output_root" --frame-index 4 --max-sequences "$test_limit" \
  --model-name objectclear_official --prompt-policy 'oracle per-frame object mask'
echo "PRISM_OBJECTCLEAR_COMPARISON_COMPLETE $output_root"
