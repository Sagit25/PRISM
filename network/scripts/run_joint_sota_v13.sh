#!/usr/bin/env bash
set -Eeuo pipefail

# Reuse the verified Base+ / ObjectClear setup from launch_joint_sota_full_v11.sh.
# Every probe and fine-tune has a distinct output directory and W&B run.
case "${PRISM_V13_MODE:-}" in
  diagnose|flow_finetune|visible_finetune) ;;
  *) echo "PRISM_V13_MODE must be diagnose, flow_finetune, or visible_finetune" >&2; exit 2 ;;
esac
: "${PRISM_V13_DONOR_URI:?PRISM_V13_DONOR_URI is required}"
test -n "${PRISM_FULL_RUN_NAME:-}"
test -n "${PRISM_OUTPUT_ROOT:-}"
test -n "${PRISM_DATA_ROOT:-}"
test -n "${PRISM_ARCHIVE_VOLUME:-}"

mkdir -p "$PRISM_OUTPUT_ROOT" "$PRISM_DATA_ROOT"
materialize_components=(--component metadata --component test)
if [[ "$PRISM_V13_MODE" != diagnose ]]; then
  materialize_components+=(--component train --component validation)
fi
python network/scripts/materialize_prism_archives.py \
  --archive-volume "$PRISM_ARCHIVE_VOLUME" \
  --storage-name "$PRISM_ARCHIVE_STORAGE_NAME" \
  --download-root "$PRISM_ARCHIVE_DOWNLOAD_ROOT" \
  --output-root "$PRISM_DATA_ROOT" \
  "${materialize_components[@]}" \
  --delete-after-extract
python network/scripts/sync_prism_checkpoints.py restore \
  --source-uri "$PRISM_V13_DONOR_URI" \
  --output-root "$PRISM_OUTPUT_ROOT/donor" --stage 4
donor="$PRISM_OUTPUT_ROOT/donor/checkpoints/stage4/prism_stage4_best.pt"
test -s "$donor"
echo "PRISM_V13_DONOR_READY uri=$PRISM_V13_DONOR_URI mode=$PRISM_V13_MODE"

common=(
  --stage 4 --mam2-matter-backend official_sam2matting
  --official-sam2matting-root "$PRISM_SAM2_ROOT"
  --official-sam2matting-checkpoint "$PRISM_OFFICIAL_SAM2MATTING_CHECKPOINT"
  --official-sam2matting-variant "$PRISM_OFFICIAL_SAM2MATTING_VARIANT"
  --completion-variant diffusion --evidence-flux-only
  --diffusion-backend objectclear --diffusion-model "$PRISM_DIFFUSION_MODEL"
  --diffusion-revision "$PRISM_DIFFUSION_REVISION"
  --objectclear-source-root "$PRISM_OBJECTCLEAR_SOURCE_ROOT"
  --objectclear-dtype bfloat16 --objectclear-fused-output
  --objectclear-late-denoise-steps 0 --objectclear-pixel-refinement
  --diffusion-cpu-offload off --diffusion-steps 8
  --diffusion-guidance-scale 1 --diffusion-min-working-side 512
  --background-exclusion-dilation 1
  --flow-refinement --foreground-refinement --flow-guided-latent-bridge
  --use-mam2-foreground-prior --learned-object-support
  --refinement-steps 2 --amp-dtype bfloat16
  --matte-frame-chunk-size 2 --sam2-temporal-checkpoint-chunk-size 2
  --prompt-mode point --prompt-seed 7 --seed 7
  --workers 2 --eval-workers 2
  --wandb-mode online --wandb-project PRISM --wandb-entity humangpt
  --wandb-group "$PRISM_FULL_RUN_NAME"
  --wandb-image-interval 200 --wandb-image-limit 2
)
if [[ "${PRISM_INDEPENDENT_PHYSICAL_ALPHA:-false}" == "true" ]]; then
  common+=(--independent-physical-alpha)
  echo "PRISM_INDEPENDENT_PHYSICAL_ALPHA_ENABLED"
fi

run_test() {
  local checkpoint="$1" clip="$2" suffix="$3" profile="$4"
  local batch=2
  local paired_args=(--paired-eval)
  if [[ "$clip" == 8 ]]; then
    batch=1
    paired_args=()
  fi
  local result_dir="$PRISM_OUTPUT_ROOT/results/$suffix"
  mkdir -p "$result_dir"
  WANDB_RUN_ID="${PRISM_FULL_RUN_NAME}-${suffix}" WANDB_RESUME=allow \
    prism-train "${common[@]}" \
    --test-data "$PRISM_DATA_ROOT/test" --checkpoint "$checkpoint" \
    --save-dir "$result_dir" --mode test --loss-profile "$profile" \
    --clip-length "$clip" --batch-size "$batch" "${paired_args[@]}" \
    --wandb-run-name "${PRISM_FULL_RUN_NAME}-${suffix}"
  echo "PRISM_V13_TEST_COMPLETE clip=$clip result=$result_dir"
}

if [[ "$PRISM_V13_MODE" == diagnose ]]; then
  run_test "$donor" 4 t4 detail_recovery
  run_test "$donor" 8 t8 detail_recovery
  echo PRISM_V13_DIAGNOSTICS_COMPLETE
  exit 0
fi

profile=flow_recovery
if [[ "$PRISM_V13_MODE" == visible_finetune ]]; then
  # Same donor and T=4 evaluation as the v13 flow fine-tune. The only new
  # objective uses measured background from either temporal direction.
  profile=visible_background_recovery
  run_test "$donor" 4 donor_t4 "$profile"
fi

stage_dir="$PRISM_OUTPUT_ROOT/checkpoints/stage4"
mkdir -p "$stage_dir"
WANDB_RUN_ID="${PRISM_FULL_RUN_NAME}-stage4" WANDB_RESUME=allow \
  prism-train "${common[@]}" \
  --train-data "$PRISM_DATA_ROOT/train" \
  --val-data "$PRISM_DATA_ROOT/validation" \
  --checkpoint "$donor" --save-dir "$stage_dir" \
  --mode train --loss-profile "$profile" \
  --epochs "${PRISM_V13_EPOCHS:-3}" --validate-every-epochs 1 \
  --clip-length 4 --batch-size 2 --paired-backgrounds \
  --paired-microbatch-checkpointing \
  --activation-checkpointing --matte-full-activation-checkpointing \
  --sam2-temporal-activation-checkpointing \
  --lr "${PRISM_STAGE4_LR:-1e-5}" --scheduler cosine \
  --gradient-clip 0.5 --cudnn-benchmark \
  --runtime-contract-checks --parameter-finiteness-interval-steps 10 \
  --checkpoint-interval-steps 50 \
  --wandb-run-name "${PRISM_FULL_RUN_NAME}-stage4"
best="$stage_dir/prism_stage4_best.pt"
test -s "$best"
python network/scripts/sync_prism_checkpoints.py upload \
  --source "$best" \
  --destination-uri "$PRISM_CHECKPOINT_URI/checkpoints/stage4/prism_stage4_best.pt"
run_test "$best" 4 t4 "$profile"
if [[ "$PRISM_V13_MODE" == visible_finetune ]]; then
  echo PRISM_V14_VISIBLE_BACKGROUND_FINETUNE_COMPLETE
else
  echo PRISM_V13_FLOW_FINETUNE_COMPLETE
fi
