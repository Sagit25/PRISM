#!/usr/bin/env bash
# Sequential, matched-budget Stage-4 ablations on the professor's GPU1.
set -Eeuo pipefail

code=/home/sukhun/prism-training-v27-pam-alpha-20261107
donor_checkout=/home/sukhun/prism-training-v26-128-20261103
python_bin="$donor_checkout/.venv/bin/python"
source_data=/home/sukhun/prism-datasets/128-20261103-merged
checkpoint=/home/sukhun/prism-training/v27-128-pam-alpha/checkpoints/stage4/prism_stage4_best.pt
output=/home/sukhun/prism-training/v27-128-pam-alpha/ablation-stage4-128-v1
expected_code=354c6895e388bc9afcad5f7ab36ba2d9906144ee
expected_checkpoint=38fd4a5a7b49dfc0a342df5c864bc235eba9e920d6e47d0ba754d1129b7c5214
subset_script="$output/create_prism_ablation_subset.py"
mode="${PRISM_ABLATION_QUEUE_MODE:-full}"

[[ "$mode" == smoke || "$mode" == full ]] || { echo "invalid mode" >&2; exit 2; }
[[ -x "$python_bin" && -f "$subset_script" ]] || { echo "runtime or subset script missing" >&2; exit 2; }
[[ "$(git -C "$code" rev-parse HEAD)" == "$expected_code" ]] || { echo "code SHA mismatch" >&2; exit 2; }
[[ "$(sha256sum "$checkpoint" | cut -d' ' -f1)" == "$expected_checkpoint" ]] || { echo "checkpoint SHA mismatch" >&2; exit 2; }
mkdir -p "$output/subset" "$output/wandb" "$output/tmp"

credential_file=/home/sukhun/.config/prism/credentials.json
while IFS= read -r -d '' credential_name && IFS= read -r -d '' credential_value; do
  export "$credential_name=$credential_value"
done < <("$python_bin" - "$credential_file" <<'PY'
import base64, json, os, stat, sys
path = sys.argv[1]
status = os.stat(path)
if status.st_uid != os.getuid() or stat.S_IMODE(status.st_mode) != 0o600:
    raise SystemExit("credential file must be user-owned mode 600")
values = json.load(open(path, encoding="utf-8"))
for key in ("HF_TOKEN", "WANDB_API_KEY"):
    value = base64.b64decode(values[key], validate=True)
    if not value or b"\x00" in value:
        raise SystemExit("invalid credential entry")
    sys.stdout.buffer.write(key.encode() + b"\x00" + value + b"\x00")
PY
)

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$code/.runtime:$donor_checkout/network/third_party/sam2:${PYTHONPATH:-}"
export PATH="$(dirname "$python_bin"):$PATH"
export HF_HOME=/home/sukhun/prism-model-cache/huggingface
export TORCH_HOME=/home/sukhun/prism-model-cache/torch
export WANDB_DIR="$output/wandb" TMPDIR="$output/tmp"
export OPENCV_IO_ENABLE_OPENEXR=1 PYTHONUNBUFFERED=1

"$python_bin" "$subset_script" "$source_data/validation" "$output/subset/validation" --count 16
"$python_bin" "$subset_script" "$source_data/test" "$output/subset/test" --count 32

variants=(full no_inverse no_joint_refinement detached_paths no_residual scalar_transmission)
epochs=2
train_batches=32
eval_batches=16
qualitative_limit=8
wandb_mode="${PRISM_ABLATION_WANDB_MODE:-online}"
[[ "$wandb_mode" == online || "$wandb_mode" == offline ]] || {
  echo "invalid W&B mode" >&2; exit 2;
}
if [[ "$mode" == smoke ]]; then
  variants=(full)
  epochs=1
  train_batches=1
  eval_batches=1
  qualitative_limit=1
  wandb_mode=disabled
fi

for variant in "${variants[@]}"; do
  run_dir="$output/$mode/$variant"
  if [[ -f "$run_dir/.complete" ]]; then
    echo "PRISM_ABLATION_ALREADY_COMPLETE variant=$variant"
    continue
  fi
  mkdir -p "$run_dir"
  echo "PRISM_ABLATION_START variant=$variant mode=$mode donor=$expected_checkpoint"
  "$python_bin" -m refractive_mam2.train \
    --train-data "$source_data/train" \
    --val-data "$output/subset/validation" \
    --test-data "$output/subset/test" \
    --checkpoint "$checkpoint" --save-dir "$run_dir" \
    --stage 4 --mode both --ablation "$variant" \
    --epochs "$epochs" --validate-every-epochs 1 \
    --batch-size 2 --clip-length 4 --workers 4 --eval-workers 2 \
    --max-train-batches "$train_batches" --max-eval-batches "$eval_batches" \
    --checkpoint-interval-steps 32 --progress-interval-steps 16 \
    --lr 1e-5 --scheduler cosine --gradient-clip 3 \
    --no-strict-contract --no-runtime-contract-checks \
    --paired-backgrounds --compute-lpips \
    --prompt-mode point --prompt-seed 7 --seed 7 \
    --sam2-checkpoint "$donor_checkout/network/checkpoints/sam2.1_hiera_large.pt" \
    --mam2-matter-backend external_mematte \
    --mematte-root "$donor_checkout/network/third_party/MEMatte" \
    --mematte-config "$donor_checkout/network/third_party/MEMatte/configs/MEMatte_S_topk0.25_win_global_long.py" \
    --mematte-checkpoint "$donor_checkout/network/checkpoints/MEMatte_AIM500.pth" \
    --mematte-max-tokens 12000 --mematte-patch-decoder \
    --mematte-train-decoder --mematte-train-backbone \
    --matte-frame-chunk-size 4 --sam2-temporal-checkpoint-chunk-size 4 \
    --fixed-mam2-alpha --pam-alpha-residual --pam-alpha-max-delta 0.25 \
    --flow-refinement --foreground-refinement --foreground-detail-refinement \
    --flow-guided-latent-bridge --use-mam2-foreground-prior \
    --evidence-flux-only --flux-spatial-singleton-evidence \
    --flux-require-unanimous-direct-consensus --refine-inverse-evidence \
    --background-exclusion-dilation 1 \
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
    --activation-checkpointing --matte-full-activation-checkpointing \
    --sam2-temporal-activation-checkpointing \
    --loss-profile pam_alpha_refinement \
    --qualitative-limit "$qualitative_limit" \
    --wandb-mode "$wandb_mode" --wandb-project PRISM \
    --wandb-entity humangpt --wandb-group prism-v27-ablation-128 \
    --wandb-run-name "prism-gpu1-v27-ablation-${variant}-${mode}-v1" \
    --wandb-image-limit 2 --wandb-scalar-interval-steps 10 \
    > "$run_dir/train.log" 2>&1
  [[ -s "$run_dir/prism_stage4_metrics.json" ]] || { echo "metrics missing: $variant" >&2; exit 3; }
  touch "$run_dir/.complete"
  echo "PRISM_ABLATION_COMPLETE variant=$variant metrics=$run_dir/prism_stage4_metrics.json"
done
