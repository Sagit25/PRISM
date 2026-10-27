#!/usr/bin/env bash
set -Eeuo pipefail

cd /root/workspace/PRISM
run_name="${PRISM_BASELINE_RUN_NAME:?PRISM_BASELINE_RUN_NAME is required}"
test -n "${WANDB_API_KEY:-}"
export PYTHONUNBUFFERED=1
export HF_HOME=/output/prism-comparison-hf-cache
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DOWNLOAD_TIMEOUT=120
mkdir -p "$HF_HOME" "/output/$run_name"
echo "PRISM_OBJECTCLEAR_BASELINE_CODE_READY $(git rev-parse HEAD)"

git init /root/workspace/ObjectClear
git -C /root/workspace/ObjectClear remote add origin https://github.com/zjx0101/ObjectClear.git
git -C /root/workspace/ObjectClear fetch --depth 1 origin c052d91ecd8772744a5ab97527441c813ab83009
git -C /root/workspace/ObjectClear checkout --detach FETCH_HEAD
test "$(git -C /root/workspace/ObjectClear rev-parse HEAD)" = c052d91ecd8772744a5ab97527441c813ab83009
python network/scripts/patch_objectclear_bf16.py /root/workspace/ObjectClear
python -m pip install --no-cache-dir 'vessl>=0.1.199' -e './network[data,experiment,objectclear]'
PYTHON_BIN=python bash network/scripts/install_official_sam2.sh \
  /root/workspace/PRISM/network/third_party/sam2 \
  2b90b9f5ceec907a1c18123530e92e794ad901a4 none
python -c 'import refractive_mam2.dataset, refractive_mam2.objectclear_completion; print("PRISM_OBJECTCLEAR_BASELINE_IMPORT_OK", flush=True)'
python -c 'import torch,torchvision; assert torch.__version__.startswith("2.5.1"); assert torch.cuda.is_available(); print("PRISM_OBJECTCLEAR_BASELINE_RUNTIME_OK",torch.cuda.get_device_name(0),flush=True)'

python network/scripts/materialize_prism_archives.py \
  --archive-volume prism-full-64-singleflow-pack-v2-369367271785-output \
  --storage-name vessl-storage \
  --output-root /root/workspace/prism-compare-data \
  --download-root /root/workspace/prism-compare-downloads \
  --component metadata --component test --retries 5

python network/scripts/evaluate_official_objectclear.py \
  --data-root /root/workspace/prism-compare-data \
  --output-root "/output/$run_name" \
  --source-root /root/workspace/ObjectClear \
  --clip-length 8 --max-sequences 16 --frame-index 4 \
  --steps 20 --run-name "$run_name"
