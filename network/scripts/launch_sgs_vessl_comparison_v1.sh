#!/usr/bin/env bash
set -Eeuo pipefail

cd /root/workspace/PRISM
run_name="${PRISM_COMPARE_RUN_NAME:?PRISM_COMPARE_RUN_NAME is required}"
variant="${PRISM_COMPARE_VARIANT:?PRISM_COMPARE_VARIANT is required}"
case "$variant" in
  native|upscale128|relaxed-direct) ;;
  *) echo "Unsupported PRISM_COMPARE_VARIANT=$variant" >&2; exit 2 ;;
esac
test -n "${WANDB_API_KEY:-}"
test -n "${HF_TOKEN:-}"
export PYTHONUNBUFFERED=1
export HF_HOME=/output/prism-comparison-hf-cache
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DOWNLOAD_TIMEOUT=120
mkdir -p "$HF_HOME" "/output/$run_name" /root/workspace/models
echo "PRISM_COMPARE_CODE_READY $(git rev-parse HEAD) variant=$variant"

python -m pip install --no-cache-dir 'vessl>=0.1.199' -e './network[data,sam2,experiment,evaluation,diffusion]'
python -m pip install --no-cache-dir timm einops ftfy regex loguru fvcore scikit-image
python -c 'import torch,torchvision; assert torch.__version__.startswith("2.5.1"); assert torchvision.__version__.startswith("0.20.1"); assert torch.cuda.is_available(); print("PRISM_COMPARE_RUNTIME_OK",torch.cuda.get_device_name(0),flush=True)'
PYTHON_BIN=python network/scripts/install_official_sam2.sh
export PRISM_MEMATTE_CHECKPOINT=/root/workspace/models/MEMatte_AIM500.pth
export PRISM_MEMATTE_CHECKPOINT_VOLUME=prism-full-64-singleflow-pack-v2-369367271785-output
export PRISM_MEMATTE_CHECKPOINT_KEY=models/MEMatte_AIM500.pth
export PRISM_MEMATTE_CHECKPOINT_SHA256=538cf871933dd0663a6bc28a4bec9b870cc2de909c7431b857b72ce427e82a09
PYTHON_BIN=python network/scripts/install_official_mematte.sh

python network/scripts/materialize_prism_archives.py \
  --archive-volume prism-full-64-singleflow-pack-v2-369367271785-output \
  --storage-name vessl-storage \
  --output-root /root/workspace/prism-compare-data \
  --download-root /root/workspace/prism-compare-downloads \
  --component metadata --component test --retries 5

checkpoint=/root/workspace/models/prism-v11-stage4-best.pt
python - "$checkpoint" <<'PY'
from pathlib import Path
import hashlib
import sys
from vessl.storage.volume_v2 import _get_volume_with_federate

destination=Path(sys.argv[1])
expected='bc881efa6a2f8c370cf1e00ce13e927c27c1907a5527bf7286097731920caaa7'
volume=_get_volume_with_federate('vessl-storage','prism-full-64-singleflow-pack-v2-369367271785-output',federation_type='read')
key='/'.join((volume.base_path.strip('/'),'sgs-comparison-v11','v11-stage4-best.pt'))
volume.s3_client.download_file(volume.bucket_name,key,str(destination))
digest=hashlib.sha256(destination.read_bytes()).hexdigest()
if digest!=expected: raise RuntimeError(f'PRISM checkpoint SHA-256 mismatch: {digest}')
print('PRISM_COMPARE_CHECKPOINT_OK',digest,flush=True)
PY

working_side=0
direct_threshold=0.1
consensus_args=(--flux-require-unanimous-direct-consensus --flux-min-consistent-observations 2)
if [[ "$variant" == upscale128 ]]; then working_side=128; fi
if [[ "$variant" == relaxed-direct ]]; then
  direct_threshold=0.5
  consensus_args=(--no-flux-require-unanimous-direct-consensus --flux-min-consistent-observations 1)
fi
echo "PRISM_COMPARE_BACKGROUND_CONFIG working_side=$working_side direct_threshold=$direct_threshold consensus=${consensus_args[*]}"

WANDB_RUN_ID="$run_name" WANDB_RESUME=allow prism-train \
  --test-data /root/workspace/prism-compare-data/test \
  --checkpoint "$checkpoint" \
  --save-dir "/output/$run_name" \
  --stage 4 --loss-profile optical_priority --optical-curriculum \
  --mode test --batch-size 1 --clip-length 8 \
  --max-eval-batches 16 --workers 0 --eval-workers 0 \
  --prompt-mode point --prompt-seed 7 --seed 7 \
  --completion-variant diffusion --completion-backbone ffc \
  --evidence-flux-only --flow-refinement --independent-physical-alpha \
  --background-exclusion-dilation 1 \
  --flux-direct-observation-threshold "$direct_threshold" \
  "${consensus_args[@]}" \
  --flux-spatial-singleton-evidence --no-refine-inverse-evidence \
  --diffusion-model black-forest-labs/FLUX.1-Fill-dev \
  --diffusion-steps 30 --diffusion-guidance-scale 30 \
  --diffusion-mask-dilation 0 \
  --diffusion-min-working-side "$working_side" \
  --diffusion-dtype bfloat16 --diffusion-cpu-offload on \
  --amp-dtype bfloat16 \
  --matte-frame-chunk-size 4 --sam2-temporal-checkpoint-chunk-size 4 \
  --mematte-root /root/workspace/PRISM/network/third_party/MEMatte \
  --mematte-config /root/workspace/PRISM/network/third_party/MEMatte/configs/MEMatte_S_topk0.25_win_global_long.py \
  --mematte-checkpoint "$PRISM_MEMATTE_CHECKPOINT" \
  --compute-lpips \
  --wandb-mode online --wandb-entity humangpt --wandb-project PRISM \
  --wandb-group prism-64-standalone-comparison \
  --wandb-run-name "$run_name" \
  --wandb-tags prism-sgs-v11 comparative-test "$variant" \
  --wandb-image-limit 8
echo "PRISM_COMPARE_COMPLETE variant=$variant output=/output/$run_name"
