#!/usr/bin/env bash
# Train only the post-PAM depth/normal head from a frozen, verified Stage 4 model.
set -Eeuo pipefail

root=/eval/prism-stage5-128-v1
code="$root/network"
runtime=/eval/comparisons/prism-test128-v1/prism-kubectl-128-v26-common-v1/.venv
data=/data/datasets/prism-main-128-20261105/prism_main_128_20261105_merged
donor="${PRISM_STAGE5_DONOR_CHECKPOINT:?set PRISM_STAGE5_DONOR_CHECKPOINT}"
expected_sha="${PRISM_STAGE5_EXPECTED_SHA256:?set PRISM_STAGE5_EXPECTED_SHA256}"
run_name="${PRISM_STAGE5_RUN_NAME:?set PRISM_STAGE5_RUN_NAME}"
mode="${PRISM_STAGE5_MODE:-smoke}"
out="$root/$run_name"

[[ "$mode" == smoke || "$mode" == full ]] || { echo "invalid Stage 5 mode" >&2; exit 2; }
[[ -s "$donor" && -f "$data/.prism_128_ready" && -x "$runtime/bin/python" ]] || {
  echo "Stage 5 donor, merged dataset, or Python runtime is missing" >&2; exit 2;
}
[[ "$(sha256sum "$donor" | cut -d' ' -f1)" == "$expected_sha" ]] || {
  echo "Stage 5 donor SHA-256 mismatch" >&2; exit 2;
}
mkdir -p "$out/checkpoints" "$out/wandb" "$out/tmp"
export PYTHONPATH="$root/python_path:$code/third_party/sam2:${PYTHONPATH:-}"
export PATH="$runtime/bin:$PATH"
export HF_HOME=/data/cache/prism-mam2-pam2-hf
export TORCH_HOME=/eval/cache/torch
export WANDB_DIR="$out/wandb" TMPDIR="$out/tmp"
export PYTHONUNBUFFERED=1 OPENCV_IO_ENABLE_OPENEXR=1
[[ -n "${HF_TOKEN:-}" && -n "${WANDB_API_KEY:-}" ]] || {
  echo "HF_TOKEN and WANDB_API_KEY secrets are required" >&2; exit 2;
}

alpha_mode="$(python - "$donor" <<'PY'
import sys, torch
d = torch.load(sys.argv[1], map_location='cpu', weights_only=True)
assert d['format_version'] == 8
assert d['metadata']['arguments']['stage'] == '4'
m = d['pipeline_config']['matter']
assert m['fixed_mam2_alpha'] and m['use_mam2_foreground_prior']
assert d['pipeline_config']['background']['evidence_flux_only']
print('residual' if m.get('pam_alpha_residual', False) else 'fixed')
PY
)"
alpha_args=()
if [[ "$alpha_mode" == residual ]]; then
  alpha_args=(--pam-alpha-residual --pam-alpha-max-delta 0.25)
fi
loader_sha="$(sha256sum "$code/src/dataset.py" | cut -d' ' -f1)"
printf 'PRISM_STAGE5_START run=%s mode=%s donor_sha256=%s alpha_mode=%s base_code=354c6895e388bc9afcad5f7ab36ba2d9906144ee normal_loader_sha256=%s\n' \
  "$run_name" "$mode" "$expected_sha" "$alpha_mode" "$loader_sha"

train_data="$data/train"
val_data="$data/validation"
epochs=14
train_batches=256
eval_batches=16
wandb_mode=online
if [[ "$mode" == smoke ]]; then
  smoke=/data/datasets/prism-main-128-20261105/prism_main_128_smoke_20261105
  train_data="$smoke/train"
  val_data="$smoke/validation"
  epochs=1
  train_batches=1
  eval_batches=1
  wandb_mode=disabled
fi

checkpoint_args=(--checkpoint "$donor")
if [[ "$mode" == full && -s "$out/checkpoints/prism_stage4_resume.pt" ]]; then
  checkpoint_args=(--resume "$out/checkpoints/prism_stage4_resume.pt")
fi

python -m refractive_mam2.train \
  --train-data "$train_data" --val-data "$val_data" \
  "${checkpoint_args[@]}" --save-dir "$out/checkpoints" \
  --stage 4 --mode train --epochs "$epochs" \
  --batch-size 1 --clip-length 4 --workers 2 --eval-workers 2 \
  --max-train-batches "$train_batches" --partition-train-epochs \
  --max-eval-batches "$eval_batches" --validate-every-epochs 3 \
  --checkpoint-interval-steps 50 --progress-interval-steps 25 \
  --lr 1e-4 --scheduler cosine --gradient-clip 1 \
  --prompt-mode point --prompt-seed 7 --seed 7 \
  --sam2-checkpoint "$code/checkpoints/sam2.1_hiera_large.pt" \
  --mam2-matter-backend external_mematte \
  --mematte-root "$code/third_party/MEMatte" \
  --mematte-config "$code/third_party/MEMatte/configs/MEMatte_S_topk0.25_win_global_long.py" \
  --mematte-checkpoint "$code/checkpoints/MEMatte_AIM500.pth" \
  --mematte-max-tokens 12000 --mematte-patch-decoder \
  --mematte-train-decoder --mematte-train-backbone \
  --matte-frame-chunk-size 4 --sam2-temporal-checkpoint-chunk-size 4 \
  --fixed-mam2-alpha "${alpha_args[@]}" \
  --flow-refinement --foreground-refinement --foreground-detail-refinement \
  --flow-guided-latent-bridge --use-mam2-foreground-prior \
  --evidence-flux-only --flux-spatial-singleton-evidence \
  --refine-inverse-evidence --background-exclusion-dilation 1 \
  --flux-direct-observation-threshold 0.1 \
  --flux-min-consistent-observations 2 \
  --refinement-steps 3 --refractive-kernel-size 1 \
  --refractive-kernel-radius-fraction 0.015625 \
  --refractive-kernel-center-bias 4.0 \
  --completion-variant diffusion --completion-backbone ffc \
  --diffusion-model black-forest-labs/FLUX.1-Fill-dev \
  --diffusion-steps 30 --diffusion-guidance-scale 30 \
  --diffusion-mask-dilation 0 --diffusion-min-working-side 1024 \
  --diffusion-pixel-refinement --diffusion-dtype bfloat16 \
  --diffusion-cpu-offload auto --amp-dtype bfloat16 \
  --loss-profile balanced_detail_recovery \
  --geometry-post-head --geometry-head-only --allow-unpaired-joint \
  --wandb-mode "$wandb_mode" --wandb-project PRISM \
  --wandb-entity humangpt --wandb-group prism-128-stage5-geometry \
  --wandb-run-name "$run_name" --wandb-image-interval 200 \
  --wandb-scalar-interval-steps 10 --wandb-image-limit 2
echo "PRISM_STAGE5_TRAIN_COMPLETE run=$run_name checkpoint=$out/checkpoints/prism_stage4_best.pt"
