#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
run_name="${PRISM_SGS_RUN_NAME:?PRISM_SGS_RUN_NAME is required}"
expected_commit="${PRISM_SGS_EXPECTED_COMMIT:?PRISM_SGS_EXPECTED_COMMIT is required}"
source_root=/data/code/MAM2-official-prism-v1
source_commit=16c3d08a8ffbed497f0b0575b3d5628bec7b64a2
archive_root=/data/datasets/prism-full-64-singleflow-pack-v2-369367271785-output
output_root="/data/checkpoints/$run_name"

test -s "$archive_root/archive_manifest.json"
test -s "$archive_root/dataset_manifest.json"
test -n "${WANDB_API_KEY:-}"
test -n "${HF_TOKEN:-}"
test ! -e "$output_root/checkpoints/stage1a/.training_complete" || {
  echo "Existing Stage 1A result found; refusing a duplicate run name" >&2
  exit 2
}

export DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
apt-get update -qq
apt-get install -y -qq git curl build-essential ninja-build libgl1 libglib2.0-0 ffmpeg
test "$(git rev-parse HEAD)" = "$expected_commit"
mkdir -p /data/code /data/cache/prism-mam2-pam2-hf
if [[ ! -d "$source_root/.git" ]]; then
  git clone --no-checkout https://github.com/ChenyiZhang007/Matting-Anything-2.git "$source_root"
fi
git -C "$source_root" fetch --depth 1 origin "$source_commit"
git -C "$source_root" checkout --detach "$source_commit"
test "$(git -C "$source_root" rev-parse HEAD)" = "$source_commit"
export PRISM_SAM2_ROOT="$source_root" PRISM_OFFICIAL_MAM2_ROOT="$source_root"
export HF_HOME=/data/cache/prism-mam2-pam2-hf
export TMPDIR=/data/cache/prism-mam2-pam2-tmp
mkdir -p "$TMPDIR"

python -m pip install --quiet --upgrade pip
python -m pip install --quiet -e './network[data,sam2,experiment,evaluation,diffusion]'
cat > /tmp/prism-mam2-pam2-constraints.txt <<'EOF'
torch==2.5.1
torchvision==0.20.1
numpy==1.26.4
tensordict==0.6.2
EOF
python -m pip install --quiet -c /tmp/prism-mam2-pam2-constraints.txt \
  -e "$source_root" 'peft==0.15.2' 'tensordict==0.6.2'
python -m pip uninstall -y torchaudio || true

python - <<'PY'
from hashlib import sha256
from huggingface_hub import hf_hub_download, HfApi
from pathlib import Path
import os
import torch
import torchvision

assert torch.__version__.startswith('2.5.1'), torch.__version__
assert torchvision.__version__.startswith('0.20.1'), torchvision.__version__
assert torch.cuda.is_available(), 'CUDA is unavailable'
HfApi().model_info('black-forest-labs/FLUX.1-Fill-dev', token=os.environ['HF_TOKEN'])
expected = {
    'mam2_main.pt': '65e3c27060febae474af1662d3f3f038f024de17715595ca61045efb357603b2',
    'mam2_matter.pth': '538cf871933dd0663a6bc28a4bec9b870cc2de909c7431b857b72ce427e82a09',
}
for filename, digest in expected.items():
    path = Path(hf_hub_download(
        repo_id='Chenyi007/Matting-Anything-2', filename=filename,
        revision='0e0a8d0d9034b00c4baca88f1a0bb1397e724905',
    ))
    checksum = sha256()
    with path.open('rb') as checkpoint:
        for block in iter(lambda: checkpoint.read(8 * 1024 * 1024), b''):
            checksum.update(block)
    actual = checksum.hexdigest()
    if actual != digest:
        raise RuntimeError(f'{filename} checksum mismatch: {actual}')
    print('PRISM_MAM2_CHECKPOINT_OK', filename, actual, flush=True)
print('PRISM_MAM2_PAM2_RUNTIME_OK', torch.__version__, torch.cuda.get_device_name(0), flush=True)
print('PRISM_FLUX_ACCESS_OK', flush=True)
PY

python -m pip install --quiet pytest
PYTHONPATH="$repo_root/network" python -m pytest -q \
  network/tests/test_official_mam2.py network/tests/test_training_stages.py \
  network/tests/test_background.py network/tests/test_pipeline.py
echo PRISM_MAM2_PAM2_TESTS_OK

export PRISM_DATA_ROOT=/data/datasets/prism-full-64-materialized-v7r1
export PRISM_ARCHIVE_ROOT="$archive_root"
export PRISM_MATERIALIZED_ROOT="$PRISM_DATA_ROOT"
export PRISM_MATERIALIZE_ALL=true PRISM_VERIFY_ARCHIVES=true
export PRISM_OUTPUT_ROOT="$output_root"
export PRISM_CACHE_ROOT=/data/cache/prism-shared-models
export PRISM_SHARD_CACHE_ROOT="/data/cache/prism-shards-$run_name"
export PRISM_MAM2_MATTER_BACKEND=official_mam2
export PRISM_SKIP_STAGE1_PRETRAINED=false PRISM_STAGE1_TRANSFER=false
export PRISM_TRAINING_SCHEDULE=oracle_joint
export PRISM_STAGE1A_EPOCHS=4 PRISM_STAGE1B_EPOCHS=4
export PRISM_STAGE2_EPOCHS=4 PRISM_STAGE3_EPOCHS=0 PRISM_STAGE4_EPOCHS=8
export PRISM_MAX_TRAIN_BATCHES=50 PRISM_PARTITION_TRAIN_EPOCHS=true
export PRISM_MAX_EVAL_BATCHES=8 PRISM_VALIDATION_INTERVAL_STAGE34=1
export PRISM_STAGE2_PAIRED_BACKGROUNDS=true
export PRISM_STAGE2_LR=2e-5 PRISM_STAGE4_LR=1e-5
export PRISM_LOSS_PROFILE=balanced_detail_recovery
export PRISM_FIXED_MAM2_ALPHA=true PRISM_INDEPENDENT_PHYSICAL_ALPHA=false
export PRISM_FOREGROUND_REFINEMENT=true PRISM_FOREGROUND_DETAIL_REFINEMENT=true
export PRISM_FLOW_GUIDED_LATENT_BRIDGE=true PRISM_USE_MAM2_FOREGROUND_PRIOR=true
export PRISM_FLOW_REFINEMENT=true PRISM_OPTICAL_CURRICULUM=true
export PRISM_EVIDENCE_FLUX_ONLY=true PRISM_DIFFUSION_PIXEL_REFINEMENT=true
export PRISM_DIFFUSION_MIN_WORKING_SIDE=1024
export PRISM_BACKGROUND_EXCLUSION_DILATION=1 PRISM_DIFFUSION_MASK_DILATION=0
export PRISM_REFINE_INVERSE_EVIDENCE=true
export PRISM_IMAGE_PRETRAIN_CLIP_LENGTH=1 PRISM_VIDEO_REFINE_CLIP_LENGTH=4
export PRISM_FINAL_EVAL_CLIP_LENGTH=8
export PRISM_MATTE_FRAME_CHUNK_SIZE=1 PRISM_SAM2_TEMPORAL_CHUNK_SIZE=1
export PRISM_RUNTIME_CONTRACT_CHECKS=true PRISM_AMP_DTYPE=bfloat16
export PRISM_WANDB_MODE=online PRISM_WANDB_ENTITY=humangpt
export PRISM_WANDB_PROJECT=PRISM PRISM_WANDB_GROUP="$run_name"
export PRISM_EXPERIMENT_ID="$run_name"
export PRISM_RUN_FINAL_EVALUATION=true
echo "PRISM_MAM2_PAM2_STAGE1_START run=$run_name model_commit=$expected_commit source_commit=$source_commit"
exec bash network/scripts/train_prism_all_stages.sh
