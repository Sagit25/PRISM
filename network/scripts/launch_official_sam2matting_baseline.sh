#!/usr/bin/env bash
set -Eeuo pipefail

cd /root/workspace/PRISM
export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1
export OPENCV_IO_ENABLE_OPENEXR=1
export PRISM_SAM2_ROOT=/root/workspace/SAM2Matting
export HF_HOME=/root/workspace/prism-sam2matting-baseline-hf-cache
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DOWNLOAD_TIMEOUT=120
run_name="${PRISM_BASELINE_RUN_NAME:-prism-sam2matting-baseplus-original-test-v1}"
archive_volume=prism-full-64-singleflow-pack-v2-369367271785-output
mkdir -p "$HF_HOME" /root/workspace/models "/output/$run_name"
test -n "${WANDB_API_KEY:-}"
echo "PRISM_SAM2_BASELINE_COMMIT $(git rev-parse HEAD)"

git init "$PRISM_SAM2_ROOT"
git -C "$PRISM_SAM2_ROOT" remote add origin https://github.com/FudanCVL/SAM2Matting.git
git -C "$PRISM_SAM2_ROOT" fetch --depth 1 origin 73dd721d77b56749248aefe5e8824d7f61b9d13c
git -C "$PRISM_SAM2_ROOT" checkout --detach FETCH_HEAD
test "$(git -C "$PRISM_SAM2_ROOT" rev-parse HEAD)" = 73dd721d77b56749248aefe5e8824d7f61b9d13c

python -m pip install --no-cache-dir --upgrade pip
python -m pip install --no-cache-dir vessl -e './network[data,sam2,experiment]'
python -m pip install --no-cache-dir timm einops ftfy regex loguru fvcore scikit-image
python -c 'import torch, torchvision; assert torch.__version__.startswith("2.5.1"); assert torchvision.__version__.startswith("0.20.1"); assert torch.cuda.is_available(); print("PRISM_SAM2_BASELINE_RUNTIME_OK", torch.cuda.get_device_name(0), flush=True)'

python - <<'PY'
from huggingface_hub import hf_hub_download
from pathlib import Path
import hashlib
import shutil

filename = "SAM2Matting-SAM2.1Base+.pt"
source = Path(hf_hub_download(
    repo_id="FudanCVL/SAM2Matting",
    filename=f"checkpoints/{filename}",
    revision="4315db9c60d27fde396b09765748a0ca6c97bed5",
))
target = Path("/root/workspace/models") / filename
shutil.copyfile(source, target)
digest = hashlib.sha256(target.read_bytes()).hexdigest()
expected = "1f0eb2eda3e8bc9101eafc0b30b8b8fcae1ff83d8fd3adc18e2f3b410fdaae60"
if digest != expected:
    raise RuntimeError(f"SAM2Matting Base+ checksum mismatch: {digest}")
print("PRISM_SAM2_BASELINE_WEIGHTS_OK", digest, flush=True)
PY

python network/scripts/materialize_prism_archives.py \
  --archive-volume "$archive_volume" \
  --storage-name vessl-storage \
  --output-root /root/workspace/prism-baseline-data \
  --download-root /root/workspace/prism-baseline-downloads \
  --component metadata --component test \
  --retries 5

python network/scripts/evaluate_official_sam2matting.py \
  --data-root /root/workspace/prism-baseline-data \
  --output-root "/output/$run_name" \
  --source-root "$PRISM_SAM2_ROOT" \
  --checkpoint /root/workspace/models/SAM2Matting-SAM2.1Base+.pt \
  --variant base_plus --split test --clip-length 8 \
  --run-name "$run_name"
