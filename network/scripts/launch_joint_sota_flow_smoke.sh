#!/usr/bin/env bash
set -Eeuo pipefail

cd /root/workspace/PRISM
export PYTHONUNBUFFERED=1
export PRISM_SAM2_ROOT=/root/workspace/SAM2Matting
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
output=/output/prism-joint-sota-flow-smoke-v11-v2
mkdir -p /root/workspace/models "$output"
echo "PRISM_JOINT_SOTA_CODE_OK $(git rev-parse HEAD)"

git init "$PRISM_SAM2_ROOT"
git -C "$PRISM_SAM2_ROOT" remote add origin https://github.com/FudanCVL/SAM2Matting.git
git -C "$PRISM_SAM2_ROOT" fetch --depth 1 origin 73dd721d77b56749248aefe5e8824d7f61b9d13c
git -C "$PRISM_SAM2_ROOT" checkout --detach FETCH_HEAD
test "$(git -C "$PRISM_SAM2_ROOT" rev-parse HEAD)" = 73dd721d77b56749248aefe5e8824d7f61b9d13c

git init /root/workspace/ObjectClear
git -C /root/workspace/ObjectClear remote add origin https://github.com/zjx0101/ObjectClear.git
git -C /root/workspace/ObjectClear fetch --depth 1 origin c052d91ecd8772744a5ab97527441c813ab83009
git -C /root/workspace/ObjectClear checkout --detach FETCH_HEAD
test "$(git -C /root/workspace/ObjectClear rev-parse HEAD)" = c052d91ecd8772744a5ab97527441c813ab83009

python -m pip install --no-cache-dir --upgrade pip
python -m pip install --no-cache-dir -e './network[data,sam2,experiment,evaluation,objectclear]'
python -m pip install --no-cache-dir timm einops ftfy regex loguru fvcore scikit-image
python -c 'import torch, torchvision; assert torch.__version__.startswith("2.5.1"), torch.__version__; assert torchvision.__version__.startswith("0.20.1"), torchvision.__version__; assert torch.cuda.is_available(); print("PRISM_JOINT_SOTA_RUNTIME_OK", torch.__version__, torchvision.__version__, torch.cuda.get_device_name(0), flush=True)'

python - <<'PY'
from huggingface_hub import hf_hub_download
from pathlib import Path
import hashlib
import shutil

source = Path(hf_hub_download(
    repo_id="FudanCVL/SAM2Matting",
    filename="checkpoints/SAM2Matting-SAM2.1Tiny.pt",
    revision="4315db9c60d27fde396b09765748a0ca6c97bed5",
))
target = Path("/root/workspace/models/SAM2Matting-SAM2.1Tiny.pt")
shutil.copyfile(source, target)
digest = hashlib.sha256(target.read_bytes()).hexdigest()
expected = "5b9321e3b51bc20f5b84c208746cc083dd3053dd701590f2e88dc8640afcc39d"
if digest != expected:
    raise RuntimeError(f"SAM2Matting checkpoint checksum mismatch: {digest}")
print("PRISM_JOINT_SOTA_SAM2_WEIGHTS_OK", digest, flush=True)
PY

python network/scripts/smoke_joint_sota_flow.py \
  --sam2matting-root "$PRISM_SAM2_ROOT" \
  --sam2matting-checkpoint /root/workspace/models/SAM2Matting-SAM2.1Tiny.pt \
  --objectclear-root /root/workspace/ObjectClear \
  --objectclear-model jixin0101/ObjectClear \
  --objectclear-revision c73af80888dbd519819d0a521b5c8d0f3cda6859 \
  --steps 4 \
  2>&1 | tee "$output/smoke.log"
echo PRISM_JOINT_SOTA_POST_FUSION_OK

python network/scripts/smoke_joint_sota_flow.py \
  --sam2matting-root "$PRISM_SAM2_ROOT" \
  --sam2matting-checkpoint /root/workspace/models/SAM2Matting-SAM2.1Tiny.pt \
  --objectclear-root /root/workspace/ObjectClear \
  --objectclear-model jixin0101/ObjectClear \
  --objectclear-revision c73af80888dbd519819d0a521b5c8d0f3cda6859 \
  --steps 2 \
  --late-denoise-steps 1 \
  2>&1 | tee -a "$output/smoke.log"
echo PRISM_JOINT_SOTA_INSIDE_DENOISER_OK
echo PRISM_JOINT_SOTA_SMOKE_COMPLETE
