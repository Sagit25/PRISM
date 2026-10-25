#!/usr/bin/env bash
set -Eeuo pipefail

cd /root/workspace/PRISM
export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1
export OPENCV_IO_ENABLE_OPENEXR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PRISM_SAM2_ROOT=/root/workspace/SAM2Matting
export HF_HOME=/output/prism-joint-sota-model-cache-v11
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DOWNLOAD_TIMEOUT=120
mkdir -p "$HF_HOME" /root/workspace/models

test -n "${WANDB_API_KEY:-}"
test -n "${HF_TOKEN:-}"
echo "PRISM_JOINT_SOTA_FULL_COMMIT $(git rev-parse HEAD)"

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
python network/scripts/patch_objectclear_bf16.py /root/workspace/ObjectClear

python -m pip install --no-cache-dir --upgrade pip
python -m pip install --no-cache-dir vessl -e './network[data,sam2,experiment,evaluation,objectclear]'
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

run_name="${PRISM_FULL_RUN_NAME:-prism-full-64-joint-sota-flow-v11-v1}"
archive_volume=prism-full-64-singleflow-pack-v2-369367271785-output
export PRISM_DATA_ROOT=/root/workspace/prism-data
export PRISM_ARCHIVE_VOLUME="$archive_volume"
export PRISM_ARCHIVE_STORAGE_NAME=vessl-storage
export PRISM_ARCHIVE_DOWNLOAD_ROOT=/root/workspace/prism-archive-downloads
export PRISM_MATERIALIZED_ROOT=/root/workspace/prism-data
export PRISM_MATERIALIZE_ALL=true
export PRISM_DELETE_ARCHIVES_AFTER_EXTRACT=true
export PRISM_OUTPUT_ROOT="/output/$run_name"
export PRISM_CHECKPOINT_URI="volume://vessl-storage/$archive_volume/$run_name"
export PRISM_RESTORE_CHECKPOINT_URI="$PRISM_CHECKPOINT_URI"
export PRISM_RESTORE_CHECKPOINT_STAGES=2,3,4
export PRISM_CHECKPOINT_INTERVAL_STEPS=50
export PRISM_CHECKPOINT_SYNC_SECONDS=60
export PRISM_RUNTIME_CONTRACT_CHECKS=true
export PRISM_SKIP_STAGE1_PRETRAINED=true
export PRISM_MAM2_MATTER_BACKEND=official_sam2matting
export PRISM_OFFICIAL_SAM2MATTING_ROOT="$PRISM_SAM2_ROOT"
export PRISM_OFFICIAL_SAM2MATTING_CHECKPOINT=/root/workspace/models/SAM2Matting-SAM2.1Tiny.pt
export PRISM_EVIDENCE_FLUX_ONLY=true
export PRISM_DIFFUSION_BACKEND=objectclear
export PRISM_DIFFUSION_MODEL=jixin0101/ObjectClear
export PRISM_DIFFUSION_REVISION=c73af80888dbd519819d0a521b5c8d0f3cda6859
export PRISM_OBJECTCLEAR_SOURCE_ROOT=/root/workspace/ObjectClear
export PRISM_OBJECTCLEAR_LATE_DENOISE_STEPS="${PRISM_OBJECTCLEAR_LATE_DENOISE_STEPS:-1}"
export PRISM_OBJECTCLEAR_FUSED_OUTPUT="${PRISM_OBJECTCLEAR_FUSED_OUTPUT:-false}"
export PRISM_DIFFUSION_CPU_OFFLOAD=off
export PRISM_DIFFUSION_STEPS=4
export PRISM_DIFFUSION_GUIDANCE_SCALE=1
export PRISM_DIFFUSION_MIN_WORKING_SIDE=512
export PRISM_DIFFUSION_MASK_DILATION=0
export PRISM_BACKGROUND_EXCLUSION_DILATION=1
export PRISM_AMP_DTYPE=bfloat16
export PRISM_REFINEMENT_STEPS=1
export PRISM_LOSS_PROFILE=optical_priority
# Opt-in next-run preset: the currently pinned v11 runs keep their original
# objective and architecture.  A new run name and fresh Stage 2 training are
# required because these extra PAM heads cannot be resumed from donor models
# with a different architecture.  The two phases are Stage 2 and Stage 4.
if [[ "${PRISM_JOINT_SHARP_PRESET:-false}" == "true" ]]; then
  if [[ -z "${PRISM_FULL_RUN_NAME:-}" ]]; then
    echo "PRISM_JOINT_SHARP_PRESET requires a distinct PRISM_FULL_RUN_NAME" >&2
    exit 2
  fi
  if [[ -n "${PRISM_STAGE2_DONOR_URI:-}" || -n "${PRISM_STAGE3_DONOR_URI:-}" ]]; then
    echo "joint_sharp changes PAM architecture; old Stage 2/3 donor checkpoints are incompatible" >&2
    exit 2
  fi
  export PRISM_FLOW_REFINEMENT=true
  export PRISM_FOREGROUND_REFINEMENT=true
  export PRISM_FLOW_GUIDED_LATENT_BRIDGE=true
  export PRISM_USE_MAM2_FOREGROUND_PRIOR=true
  export PRISM_REFINEMENT_STEPS=2
  export PRISM_LOSS_PROFILE=joint_sharp
  export PRISM_TRAINING_SCHEDULE=oracle_joint
  export PRISM_RESTORE_CHECKPOINT_STAGES=2,4
  echo "PRISM_JOINT_SHARP_CONFIGURED flow_refinement=true foreground_refinement=true latent_bridge=true steps=2 schedule=oracle_joint"
fi
export PRISM_IMAGE_PRETRAIN_CLIP_LENGTH=1
export PRISM_VIDEO_REFINE_CLIP_LENGTH=4
export PRISM_MATTE_FRAME_CHUNK_SIZE=2
export PRISM_SAM2_TEMPORAL_CHUNK_SIZE=2
export PRISM_PAIRED_MICROBATCH_CHECKPOINTING=true
export PRISM_STAGE2_PAIRED_BACKGROUNDS=true
export PRISM_STAGE2_EPOCHS=8
export PRISM_STAGE3_EPOCHS=10
export PRISM_STAGE4_EPOCHS=12
if [[ "${PRISM_JOINT_SHARP_PRESET:-false}" == "true" ]]; then
  export PRISM_STAGE2_EPOCHS=3
fi
export PRISM_VALIDATION_INTERVAL_STAGE34=5
export PRISM_RUN_FINAL_EVALUATION=false
export PRISM_WANDB_MODE=online
export PRISM_WANDB_PROJECT=PRISM
export PRISM_WANDB_ENTITY=humangpt
export PRISM_WANDB_GROUP="$run_name"
export PRISM_EXPERIMENT_ID="$run_name"

if [[ -n "${PRISM_STAGE2_DONOR_URI:-}" ]]; then
  python network/scripts/sync_prism_checkpoints.py restore \
    --source-uri "$PRISM_STAGE2_DONOR_URI" \
    --output-root "$PRISM_OUTPUT_ROOT" --stage 2
  test -s "$PRISM_OUTPUT_ROOT/checkpoints/stage2/prism_stage2_best.pt"
  test -f "$PRISM_OUTPUT_ROOT/checkpoints/stage2/.training_complete"
  echo "PRISM_JOINT_SOTA_STAGE2_DONOR_VERIFIED source=$PRISM_STAGE2_DONOR_URI"
fi

if [[ -n "${PRISM_STAGE3_DONOR_URI:-}" ]]; then
  python network/scripts/sync_prism_checkpoints.py restore \
    --source-uri "$PRISM_STAGE3_DONOR_URI" \
    --output-root "$PRISM_OUTPUT_ROOT" --stage 3
  test -s "$PRISM_OUTPUT_ROOT/checkpoints/stage3/prism_stage3_best.pt"
  test -f "$PRISM_OUTPUT_ROOT/checkpoints/stage3/.training_complete"
  echo "PRISM_JOINT_SOTA_STAGE3_DONOR_VERIFIED source=$PRISM_STAGE3_DONOR_URI"
fi

bash network/scripts/train_prism_all_stages.sh

required_stages=(2 3 4)
if [[ "${PRISM_JOINT_SHARP_PRESET:-false}" == "true" ]]; then
  required_stages=(2 4)
fi
for stage in "${required_stages[@]}"; do
  test -s "$PRISM_OUTPUT_ROOT/checkpoints/stage$stage/prism_stage${stage}_best.pt"
  test -f "$PRISM_OUTPUT_ROOT/checkpoints/stage$stage/.training_complete"
done
echo PRISM_JOINT_SOTA_FULL_TRAINING_COMPLETE

objectclear_output_arg=--no-objectclear-fused-output
if [[ "$PRISM_OBJECTCLEAR_FUSED_OUTPUT" == "true" ]]; then
  objectclear_output_arg=--objectclear-fused-output
fi
final_model_args=(--refinement-steps "$PRISM_REFINEMENT_STEPS" --loss-profile "$PRISM_LOSS_PROFILE")
if [[ "${PRISM_JOINT_SHARP_PRESET:-false}" == "true" ]]; then
  final_model_args+=(
    --flow-refinement --foreground-refinement
    --flow-guided-latent-bridge --use-mam2-foreground-prior
  )
fi
WANDB_RUN_ID="${run_name}-test" WANDB_RESUME=allow prism-train \
  --test-data "$PRISM_DATA_ROOT/test" \
  --checkpoint "$PRISM_OUTPUT_ROOT/checkpoints/stage4/prism_stage4_best.pt" \
  --save-dir "$PRISM_OUTPUT_ROOT/results/test" \
  --stage 4 --mode test --batch-size 2 --clip-length 4 \
  --workers 2 --eval-workers 2 --paired-eval \
  --mam2-matter-backend official_sam2matting \
  --official-sam2matting-root "$PRISM_SAM2_ROOT" \
  --official-sam2matting-checkpoint "$PRISM_OFFICIAL_SAM2MATTING_CHECKPOINT" \
  --completion-variant diffusion --evidence-flux-only \
  --diffusion-backend objectclear --diffusion-model "$PRISM_DIFFUSION_MODEL" \
  --diffusion-revision "$PRISM_DIFFUSION_REVISION" \
  --objectclear-source-root "$PRISM_OBJECTCLEAR_SOURCE_ROOT" \
  --objectclear-dtype bfloat16 "$objectclear_output_arg" \
  --objectclear-late-denoise-steps "$PRISM_OBJECTCLEAR_LATE_DENOISE_STEPS" \
  --diffusion-cpu-offload off \
  --diffusion-steps 4 --diffusion-guidance-scale 1 \
  --diffusion-min-working-side 512 \
  --background-exclusion-dilation 1 "${final_model_args[@]}" \
  --amp-dtype bfloat16 \
  --prompt-mode point --prompt-seed 7 --seed 7 \
  --wandb-mode online --wandb-project PRISM --wandb-entity humangpt \
  --wandb-group "$run_name" --wandb-run-name "${run_name}-test" \
  --wandb-image-interval 200 --wandb-image-limit 2
echo PRISM_JOINT_SOTA_FULL_COMPLETE
