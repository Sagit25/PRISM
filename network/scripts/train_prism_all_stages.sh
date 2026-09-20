#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"

data_root="${PRISM_DATA_ROOT:-/input/prism-main}"
archive_root="${PRISM_ARCHIVE_ROOT:-$data_root}"
archive_volume="${PRISM_ARCHIVE_VOLUME:-}"
archive_storage_name="${PRISM_ARCHIVE_STORAGE_NAME:-vessl-storage}"
archive_download_root="${PRISM_ARCHIVE_DOWNLOAD_ROOT:-/tmp/prism-archive-downloads}"
archive_max_shards="${PRISM_ARCHIVE_MAX_SHARDS_PER_COMPONENT:-}"
materialized_root="${PRISM_MATERIALIZED_ROOT:-/root/workspace/prism-data}"
extract_workers="${PRISM_EXTRACT_WORKERS:-4}"
verify_archives="${PRISM_VERIFY_ARCHIVES:-true}"
delete_archives_after_extract="${PRISM_DELETE_ARCHIVES_AFTER_EXTRACT:-false}"
output_root="${PRISM_OUTPUT_ROOT:-/output/prism-training-v1}"
seed="${PRISM_SEED:-7}"
clip_length="${PRISM_CLIP_LENGTH:-4}"
workers="${PRISM_WORKERS:-4}"
wandb_project="${PRISM_WANDB_PROJECT:-PRISM}"
wandb_group="${PRISM_WANDB_GROUP:-main-v6-fresnel}"
wandb_requested_mode="${PRISM_WANDB_MODE:-online}"
diffusion_model="${PRISM_DIFFUSION_MODEL:-black-forest-labs/FLUX.1-Fill-dev}"
diffusion_steps="${PRISM_DIFFUSION_STEPS:-30}"
experiment_id="${PRISM_EXPERIMENT_ID:-prism-v6-fresnel-seed${seed}}"
checkpoint_uri="${PRISM_CHECKPOINT_URI:-}"
checkpoint_sync_seconds="${PRISM_CHECKPOINT_SYNC_SECONDS:-300}"
amp_dtype="${PRISM_AMP_DTYPE:-bfloat16}"
matte_frame_chunk_size="${PRISM_MATTE_FRAME_CHUNK_SIZE:-1}"
sam2_temporal_chunk_size="${PRISM_SAM2_TEMPORAL_CHUNK_SIZE:-1}"
sam2_temporal_detach_interval="${PRISM_SAM2_TEMPORAL_DETACH_INTERVAL:-1}"

stage1a_epochs="${PRISM_STAGE1A_EPOCHS:-10}"
stage1b_epochs="${PRISM_STAGE1B_EPOCHS:-10}"
stage2_epochs="${PRISM_STAGE2_EPOCHS:-10}"
stage3_epochs="${PRISM_STAGE3_EPOCHS:-15}"
stage4_epochs="${PRISM_STAGE4_EPOCHS:-20}"

if [[ ! -f "$data_root/dataset_manifest.json" ]]; then
  if [[ -n "$archive_volume" ]]; then
    materialize_args=(
      --archive-volume "$archive_volume"
      --storage-name "$archive_storage_name"
      --download-root "$archive_download_root"
      --output-root "$materialized_root"
    )
    if [[ -n "$archive_max_shards" ]]; then
      materialize_args+=(--max-shards-per-component "$archive_max_shards")
    fi
    python "$script_dir/materialize_prism_archives.py" "${materialize_args[@]}"
    data_root="$materialized_root"
  else
    mapfile -t archive_candidates < <(find "$archive_root" -type f -name '*.tar' -print 2>/dev/null | head -n 1)
    if (( ${#archive_candidates[@]} > 0 )); then
      materialize_args=(
        --archive-root "$archive_root"
        --output-root "$materialized_root"
        --workers "$extract_workers"
      )
      if [[ "$verify_archives" != "true" ]]; then
        materialize_args+=(--skip-sha256)
      fi
      if [[ "$delete_archives_after_extract" == "true" ]]; then
        materialize_args+=(--delete-after-extract)
      fi
      python "$script_dir/materialize_prism_archives.py" "${materialize_args[@]}"
      data_root="$materialized_root"
    fi
  fi
fi

train_root="$data_root/train"
validation_root="$data_root/validation"
test_root="$data_root/test"

for required in \
  "$train_root" \
  "$validation_root" \
  "$test_root" \
  "$data_root/dataset_manifest.json"
do
  if [[ ! -e "$required" ]]; then
    echo "Required PRISM dataset path is missing: $required" >&2
    exit 2
  fi
done

mkdir -p \
  "$output_root/checkpoints" \
  "$output_root/results" \
  "$output_root/wandb" \
  "$output_root/cache/huggingface" \
  "$output_root/cache/torch"

export OPENCV_IO_ENABLE_OPENEXR=1
export WANDB_DIR="$output_root/wandb"
export WANDB_CACHE_DIR="$output_root/cache/wandb"
export HF_HOME="$output_root/cache/huggingface"
export TORCH_HOME="$output_root/cache/torch"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

wandb_mode="$wandb_requested_mode"
if [[ "$wandb_mode" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is absent; preserving complete W&B logs in offline mode."
  echo "Set WANDB_API_KEY in VESSL and restart to sync live without losing checkpoints."
  wandb_mode="offline"
fi

checkpoint_uploader_pid=""
start_checkpoint_uploader() {
  if [[ -z "$checkpoint_uri" ]]; then
    return
  fi
  if ! command -v vessl >/dev/null 2>&1; then
    echo "PRISM_CHECKPOINT_URI was set but the vessl CLI is unavailable." >&2
    exit 5
  fi
  (
    declare -A uploaded_signatures=()
    while true; do
      while IFS= read -r checkpoint; do
        relative="${checkpoint#"$output_root"/}"
        signature="$(stat -c '%s:%Y' "$checkpoint")"
        if [[ "${uploaded_signatures[$relative]:-}" == "$signature" ]]; then
          continue
        fi
        destination="${checkpoint_uri%/}/$relative"
        if vessl storage copy-file "$checkpoint" "$destination"; then
          uploaded_signatures[$relative]="$signature"
          echo "PRISM_CHECKPOINT_UPLOADED $destination"
        else
          echo "Checkpoint upload will be retried: $checkpoint" >&2
        fi
      done < <(
        find "$output_root/checkpoints" "$output_root/results" \
          -type f \( -name '*.pt' -o -name '*.json' -o -name '.*complete' \) \
          -print 2>/dev/null
      )
      sleep "$checkpoint_sync_seconds"
    done
  ) &
  checkpoint_uploader_pid="$!"
}

stop_checkpoint_uploader() {
  if [[ -n "$checkpoint_uploader_pid" ]]; then
    kill "$checkpoint_uploader_pid" 2>/dev/null || true
    wait "$checkpoint_uploader_pid" 2>/dev/null || true
  fi
}

trap stop_checkpoint_uploader EXIT
start_checkpoint_uploader

latest_epoch_checkpoint() {
  local stage_dir="$1"
  local stage="$2"
  local candidate=""
  shopt -s nullglob
  local checkpoints=("$stage_dir"/prism_stage"$stage"_epoch*.pt)
  shopt -u nullglob
  if (( ${#checkpoints[@]} > 0 )); then
    candidate="$(printf '%s\n' "${checkpoints[@]}" | sort | tail -n 1)"
  fi
  printf '%s' "$candidate"
}

run_stage() {
  local stage="$1"
  local epochs="$2"
  local batch_size="$3"
  local previous_checkpoint="${4:-}"
  local paired="${5:-false}"
  local final_mode="${6:-train}"
  local stage_dir="$output_root/checkpoints/stage$stage"
  local complete_marker="$stage_dir/.training_complete"
  local resume_checkpoint=""

  mkdir -p "$stage_dir"
  if [[ -f "$complete_marker" && -s "$stage_dir/prism_stage${stage}_best.pt" ]]; then
    echo "Stage $stage is already complete; reusing $stage_dir/prism_stage${stage}_best.pt"
    return
  fi

  resume_checkpoint="$(latest_epoch_checkpoint "$stage_dir" "$stage")"
  local checkpoint_args=()
  if [[ -n "$resume_checkpoint" ]]; then
    checkpoint_args=(--resume "$resume_checkpoint")
    echo "Resuming Stage $stage from $resume_checkpoint"
  elif [[ -n "$previous_checkpoint" ]]; then
    if [[ ! -s "$previous_checkpoint" ]]; then
      echo "Previous Stage checkpoint is missing: $previous_checkpoint" >&2
      exit 3
    fi
    checkpoint_args=(--checkpoint "$previous_checkpoint")
  fi

  local pair_args=()
  if [[ "$paired" == "true" ]]; then
    pair_args=(--paired-backgrounds)
  fi

  local test_args=()
  if [[ "$final_mode" == "both" ]]; then
    test_args=(--test-data "$test_root" --paired-eval --compute-lpips)
  fi

  WANDB_RUN_ID="${experiment_id}-stage${stage}" \
  WANDB_RESUME=allow \
  prism-train \
    --train-data "$train_root" \
    --val-data "$validation_root" \
    "${test_args[@]}" \
    "${checkpoint_args[@]}" \
    --save-dir "$stage_dir" \
    --stage "$stage" \
    --mode "$final_mode" \
    --epochs "$epochs" \
    --batch-size "$batch_size" \
    --clip-length "$clip_length" \
    --workers "$workers" \
    --lr 1e-4 \
    --scheduler cosine \
    --gradient-clip 1.0 \
    --amp-dtype "$amp_dtype" \
    --activation-checkpointing \
    --matte-full-activation-checkpointing \
    --matte-frame-chunk-size "$matte_frame_chunk_size" \
    --sam2-temporal-activation-checkpointing \
    --sam2-temporal-checkpoint-chunk-size "$sam2_temporal_chunk_size" \
    --sam2-temporal-detach-interval "$sam2_temporal_detach_interval" \
    --prompt-mode point \
    --prompt-seed "$seed" \
    --seed "$seed" \
    --completion-variant base \
    --completion-backbone ffc \
    --wandb-mode "$wandb_mode" \
    --wandb-project "$wandb_project" \
    --wandb-group "$wandb_group" \
    --wandb-run-name "${experiment_id}-stage${stage}" \
    --wandb-tags full-fresnel "stage${stage}" prism-base \
    --wandb-image-interval 200 \
    --wandb-image-limit 2 \
    "${pair_args[@]}"

  touch "$complete_marker"
  sync
}

stage1a_best="$output_root/checkpoints/stage1a/prism_stage1a_best.pt"
stage1b_best="$output_root/checkpoints/stage1b/prism_stage1b_best.pt"
stage2_best="$output_root/checkpoints/stage2/prism_stage2_best.pt"
stage3_best="$output_root/checkpoints/stage3/prism_stage3_best.pt"
stage4_best="$output_root/checkpoints/stage4/prism_stage4_best.pt"

run_stage 1a "$stage1a_epochs" 1 "" false train
run_stage 1b "$stage1b_epochs" 1 "$stage1a_best" false train
run_stage 2 "$stage2_epochs" 1 "$stage1b_best" false train
run_stage 3 "$stage3_epochs" 2 "$stage2_best" true train
run_stage 4 "$stage4_epochs" 2 "$stage3_best" true both

diffusion_dir="$output_root/results/prism-diffusion-flux-fill"
diffusion_marker="$diffusion_dir/.evaluation_complete"
mkdir -p "$diffusion_dir"
if [[ ! -f "$diffusion_marker" ]]; then
  if [[ "$diffusion_model" == black-forest-labs/* && -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is required for the gated FLUX.1 Fill checkpoint." >&2
    echo "Stage 1A-4 training is complete; set HF_TOKEN and restart for diffusion evaluation." >&2
    exit 4
  fi
  WANDB_RUN_ID="${experiment_id}-diffusion" \
  WANDB_RESUME=allow \
  prism-train \
    --test-data "$test_root" \
    --checkpoint "$stage4_best" \
    --save-dir "$diffusion_dir" \
    --stage 4 \
    --mode test \
    --batch-size 2 \
    --clip-length "$clip_length" \
    --workers "$workers" \
    --prompt-mode point \
    --prompt-seed "$seed" \
    --seed "$seed" \
    --completion-variant diffusion \
    --completion-backbone ffc \
    --diffusion-model "$diffusion_model" \
    --diffusion-steps "$diffusion_steps" \
    --diffusion-guidance-scale 30 \
    --diffusion-dtype bfloat16 \
    --amp-dtype "$amp_dtype" \
    --matte-frame-chunk-size "$matte_frame_chunk_size" \
    --sam2-temporal-checkpoint-chunk-size "$sam2_temporal_chunk_size" \
    --sam2-temporal-detach-interval "$sam2_temporal_detach_interval" \
    --paired-eval \
    --compute-lpips \
    --wandb-mode "$wandb_mode" \
    --wandb-project "$wandb_project" \
    --wandb-group "$wandb_group" \
    --wandb-run-name "${experiment_id}-diffusion-flux-fill" \
    --wandb-tags full-fresnel stage4 prism-diffusion flux-fill \
    --wandb-image-limit 2
  touch "$diffusion_marker"
  sync
fi

echo "PRISM_ALL_STAGES_COMPLETE output=$output_root"
