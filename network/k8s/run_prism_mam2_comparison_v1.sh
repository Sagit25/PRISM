#!/usr/bin/env bash
set -Eeuo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
run_name="${PRISM_BASELINE_RUN_NAME:?PRISM_BASELINE_RUN_NAME is required}"
expected_commit="${PRISM_BASELINE_EXPECTED_COMMIT:?PRISM_BASELINE_EXPECTED_COMMIT is required}"
test_limit="${PRISM_BASELINE_MAX_SEQUENCES:-16}"
data_root=/data/datasets/prism-full-64-materialized-v7r1
output_root="/data/comparisons/prism-test64-v1/$run_name/mam2"
source_root=/data/code/MAM2-official-baseline
export PYTHONUNBUFFERED=1 HF_HOME=/data/cache/prism-baseline-hf
export TMPDIR=/data/cache/prism-baseline-tmp PIP_NO_CACHE_DIR=1
mkdir -p "$output_root" /data/code "$HF_HOME" "$TMPDIR"
apt-get update -qq
apt-get install -y -qq git libgl1 libglib2.0-0 ffmpeg
test "$(git rev-parse HEAD)" = "$expected_commit"
test -d "$data_root/test"
test -n "${WANDB_API_KEY:-}"
echo "PRISM_MAM2_COMPARISON_CODE_READY $expected_commit"
python -m pip install --quiet -e './network[data,sam2,experiment,evaluation]'
if [ ! -d "$source_root/.git" ]; then git clone --no-checkout https://github.com/ChenyiZhang007/Matting-Anything-2.git "$source_root"; fi
git -C "$source_root" fetch origin 16c3d08a8ffbed497f0b0575b3d5628bec7b64a2
git -C "$source_root" checkout --detach 16c3d08a8ffbed497f0b0575b3d5628bec7b64a2
test "$(git -C "$source_root" rev-parse HEAD)" = 16c3d08a8ffbed497f0b0575b3d5628bec7b64a2
# MAM2 bundles a patched SAM2 with TrimapDownSampler. Resolve it before the
# unrelated Facebook SAM2 checkout or an installed sam2 package can be imported.
export PRISM_SAM2_ROOT="$source_root"
export PYTHONPATH="$source_root${PYTHONPATH:+:$PYTHONPATH}"
cat > /tmp/prism-mam2-baseline-constraints.txt <<'EOF'
torch==2.7.1
torchvision==0.22.1
EOF
python -m pip install --quiet -c /tmp/prism-mam2-baseline-constraints.txt -e "$source_root"
# The base image carries torchaudio 2.5.1, whose binary extension cannot load
# against the network package's PyTorch 2.7.1. MAM2 matting does not use audio.
python -m pip uninstall -y torchaudio
python - <<'PY'
import torch
import torchvision

assert torch.__version__.startswith('2.7.1'), torch.__version__
assert torchvision.__version__.startswith('0.22.1'), torchvision.__version__
assert torch.cuda.is_available(), 'CUDA is unavailable'
torchvision.ops.nms(
    torch.tensor([[0., 0., 1., 1.]], device='cuda'),
    torch.tensor([1.], device='cuda'), 0.5,
)
print('PRISM_MAM2_RUNTIME_OK', torch.__version__, torchvision.__version__, flush=True)
PY
python - "$source_root" <<'PY'
import inspect
import sys
from pathlib import Path

import refractive_mam2.dataset
from sam2.modeling.memory_encoder import TrimapDownSampler

source = Path(sys.argv[1]).resolve()
actual = Path(inspect.getfile(TrimapDownSampler)).resolve()
if not actual.is_relative_to(source):
    raise RuntimeError(f'MAM2 patched SAM2 was not imported from {source}: {actual}')
print('PRISM_MAM2_PATCHED_SAM2_OK', actual, flush=True)
PY
python network/scripts/evaluate_official_mam2.py \
  --data-root "$data_root" --source-root "$source_root" --output-root "$output_root" \
  --frame-index 4 --working-side 256 --max-sequences "$test_limit" \
  --run-name "$run_name-mam2"
python network/scripts/evaluate_comparison_predictions.py \
  --data-root "$data_root" --prediction-root "$output_root/predictions" \
  --output-root "$output_root" --frame-index 4 --max-sequences "$test_limit" \
  --model-name mam2_official --prompt-policy 'oracle first-frame object mask'
echo "PRISM_MAM2_COMPARISON_COMPLETE $output_root"
