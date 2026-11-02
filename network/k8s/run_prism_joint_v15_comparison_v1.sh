#!/usr/bin/env bash
set -Eeuo pipefail

comparison_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
run_name="${PRISM_SGS_RUN_NAME:?}"
comparison_commit="${PRISM_SGS_EXPECTED_COMMIT:?}"
model_commit=694539ecfa8acac573d76c71a27ec9d01ce047f0
model_root="/data/code/PRISM-joint-v15-$model_commit"
data_root=/data/datasets/prism-full-64-materialized-v7r1
checkpoint=/data/checkpoints/vessl-final-migration/joint-v15-flow-stage4-best.pt
out="/data/comparisons/prism-test64-v1/$run_name/joint_v15_flow"
limit="${PRISM_COMPARISON_MAX_SEQUENCES:-8}"
[[ "$limit" =~ ^[0-9]+$ ]] || { echo 'invalid comparison limit' >&2; exit 2; }
apt-get update -qq
apt-get install -y -qq git libgl1 libglib2.0-0
test "$(git -C "$comparison_root" rev-parse HEAD)" = "$comparison_commit"
test "$(git -C "$model_root" rev-parse HEAD)" = "$model_commit"
test -s "$checkpoint"
test -d "$data_root/test"
test -d /data/code/SAM2Matting-official-baseline/.git
test -d /data/code/ObjectClear-official-baseline/.git
test "$(git -C /data/code/SAM2Matting-official-baseline rev-parse HEAD)" = 73dd721d77b56749248aefe5e8824d7f61b9d13c
test "$(git -C /data/code/ObjectClear-official-baseline rev-parse HEAD)" = c052d91ecd8772744a5ab97527441c813ab83009
test -s /data/models/SAM2Matting-SAM2.1Base+.pt
mkdir -p "$out" /data/cache/prism-baseline-tmp
export HF_HOME=/data/cache/prism-baseline-hf
export TMPDIR=/data/cache/prism-baseline-tmp
export PYTHONUNBUFFERED=1 OPENCV_IO_ENABLE_OPENEXR=1 PIP_NO_CACHE_DIR=1
export PRISM_SAM2_ROOT=/data/code/SAM2Matting-official-baseline
export PRISM_COMPARISON_MAX_SEQUENCES="$limit"
cd "$model_root"
python -m pip install --quiet -e './network[data,sam2,experiment,evaluation,objectclear]'
python -m pip install --quiet timm einops ftfy regex loguru fvcore scikit-image
python "$comparison_root/network/scripts/patch_objectclear_bf16.py" /data/code/ObjectClear-official-baseline
python - <<'PY'
import hashlib, torch
from pathlib import Path
assert torch.cuda.is_available()
checkpoint = Path('/data/checkpoints/vessl-final-migration/joint-v15-flow-stage4-best.pt')
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == '44d1d4f6c9d526eedb0705643a0ac72888e14ea6508c666e5b4a5c61cc246891'
assert torch.load(checkpoint, map_location='cpu', weights_only=True)['format_version'] == 8
print('PRISM_JOINT_V15_CHECKPOINT_OK', flush=True)
PY
eval_args=()
if [[ "$limit" != 0 ]]; then
  eval_args+=(--max-eval-batches "$limit")
else
  limit=960
fi
test_output="$out/native_test"
mkdir -p "$test_output"
echo "PRISM_JOINT_V15_EVAL_START model_commit=$model_commit comparison_commit=$comparison_commit sequences=$limit"
WANDB_RUN_ID="${run_name}-test" WANDB_RESUME=allow prism-train \
  --test-data "$data_root/test" --checkpoint "$checkpoint" \
  --save-dir "$test_output" --stage 4 --mode test \
  --batch-size 1 --clip-length 8 --qualitative-limit "$limit" "${eval_args[@]}" \
  --workers 2 --eval-workers 2 \
  --mam2-matter-backend official_sam2matting \
  --official-sam2matting-root "$PRISM_SAM2_ROOT" \
  --official-sam2matting-checkpoint /data/models/SAM2Matting-SAM2.1Base+.pt \
  --official-sam2matting-variant base_plus \
  --completion-variant diffusion --evidence-flux-only \
  --diffusion-backend objectclear --diffusion-model jixin0101/ObjectClear \
  --diffusion-revision c73af80888dbd519819d0a521b5c8d0f3cda6859 \
  --objectclear-source-root /data/code/ObjectClear-official-baseline \
  --objectclear-dtype bfloat16 --objectclear-fused-output \
  --objectclear-late-denoise-steps 0 --objectclear-pixel-refinement \
  --diffusion-cpu-offload off --diffusion-steps 8 \
  --diffusion-guidance-scale 1 --diffusion-min-working-side 512 \
  --background-exclusion-dilation 1 \
  --flow-refinement --foreground-refinement --flow-guided-latent-bridge \
  --use-mam2-foreground-prior --learned-object-support \
  --independent-physical-alpha --refinement-steps 2 \
  --loss-profile flow_recovery --amp-dtype bfloat16 \
  --matte-frame-chunk-size 2 --sam2-temporal-checkpoint-chunk-size 2 \
  --prompt-mode point --prompt-seed 7 --seed 7 \
  --wandb-mode online --wandb-project PRISM --wandb-entity humangpt \
  --wandb-group "$run_name" --wandb-run-name "${run_name}-test" \
  --wandb-image-interval 200 --wandb-image-limit 2
cd "$comparison_root"
python network/scripts/export_existing_prism_comparison.py \
  --data-root "$data_root" --source-root "$test_output/qualitative" \
  --output-root "$out" --max-sequences "$limit" --frame-index 4 \
  --source-run prism-joint-sota-v15-physical-alpha-flow-v2
python network/scripts/evaluate_comparison_predictions.py \
  --data-root "$data_root" --prediction-root "$out/predictions" \
  --output-root "$out" --frame-index 4 --max-sequences "$limit" \
  --model-name prism_joint_v15_flow \
  --prompt-policy 'first-frame point prompt; 8-frame context; no oracle test target'
comparison_base=/data/comparisons/prism-test64-v1
panels=("Joint v15=$out/predictions")
for entry in \
  "Telea=$comparison_base/prism-kubectl-baseline-comparison-v1-full/telea/predictions" \
  "ObjectClear=$comparison_base/prism-kubectl-objectclear-comparison-v1-full/objectclear/predictions" \
  "PRISM v25=$comparison_base/prism-kubectl-prism-v25-comparison-v1-full/prism_v25/predictions"; do
  path="${entry#*=}"
  if [[ -f "$path/0000.npz" ]]; then panels+=("$entry"); fi
done
python network/scripts/make_comparison_contact_sheets.py \
  --data-root "$data_root" --output-root "$out/matched_images" \
  --frame-index 4 --indices 0 1 2 3 4 5 6 7 \
  --prediction "${panels[@]}"
echo "PRISM_JOINT_V15_COMPARISON_COMPLETE $out"
