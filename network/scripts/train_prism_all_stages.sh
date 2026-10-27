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
shard_cache_root="${PRISM_SHARD_CACHE_ROOT:-/root/workspace/prism-shard-cache}"
shard_shuffle_buffer="${PRISM_SHARD_SHUFFLE_BUFFER:-16}"
shard_download_retries="${PRISM_SHARD_DOWNLOAD_RETRIES:-5}"
archive_prefetch="${PRISM_ARCHIVE_PREFETCH:-true}"
materialize_all="${PRISM_MATERIALIZE_ALL:-false}"
archive_decode_workers="${PRISM_ARCHIVE_DECODE_WORKERS:-4}"
extract_workers="${PRISM_EXTRACT_WORKERS:-4}"
verify_archives="${PRISM_VERIFY_ARCHIVES:-true}"
delete_archives_after_extract="${PRISM_DELETE_ARCHIVES_AFTER_EXTRACT:-false}"
output_root="${PRISM_OUTPUT_ROOT:-/output/prism-training-full-v8}"
cache_root="${PRISM_CACHE_ROOT:-/root/workspace/prism-model-cache}"
seed="${PRISM_SEED:-7}"
# Stages 1A--2 learn per-frame semantics, optics, and optional geometry.  The
# joint stages alone pay for temporal propagation and shared-background
# inversion. PRISM_CLIP_LENGTH remains a compatibility fallback for the video
# stages; new runs should use the explicit schedule variables below.
legacy_clip_length="${PRISM_CLIP_LENGTH:-4}"
image_pretrain_clip_length="${PRISM_IMAGE_PRETRAIN_CLIP_LENGTH:-1}"
video_refine_clip_length="${PRISM_VIDEO_REFINE_CLIP_LENGTH:-$legacy_clip_length}"
final_eval_clip_length="${PRISM_FINAL_EVAL_CLIP_LENGTH:-8}"
temporal_ablation_clip_lengths="${PRISM_TEMPORAL_ABLATION_CLIP_LENGTHS:-1,2,4,8}"
stage2_paired_backgrounds="${PRISM_STAGE2_PAIRED_BACKGROUNDS:-true}"
training_schedule="${PRISM_TRAINING_SCHEDULE:-staged}"
if [[ "$training_schedule" != "staged" && "$training_schedule" != "oracle_joint" ]]; then
  echo "PRISM_TRAINING_SCHEDULE must be staged or oracle_joint" >&2
  exit 2
fi
workers="${PRISM_WORKERS:-4}"
eval_workers="${PRISM_EVAL_WORKERS:-4}"
wandb_project="${PRISM_WANDB_PROJECT:-PRISM}"
wandb_entity="${PRISM_WANDB_ENTITY:-}"
wandb_group="${PRISM_WANDB_GROUP:-full-prism-v8}"
wandb_requested_mode="${PRISM_WANDB_MODE:-online}"
diffusion_model="${PRISM_DIFFUSION_MODEL:-black-forest-labs/FLUX.1-Fill-dev}"
diffusion_steps="${PRISM_DIFFUSION_STEPS:-30}"
diffusion_guidance_scale="${PRISM_DIFFUSION_GUIDANCE_SCALE:-30}"
diffusion_mask_dilation="${PRISM_DIFFUSION_MASK_DILATION:-0}"
diffusion_min_working_side="${PRISM_DIFFUSION_MIN_WORKING_SIDE:-128}"
diffusion_pixel_refinement="${PRISM_DIFFUSION_PIXEL_REFINEMENT:-false}"
diffusion_pixel_refinement_args=()
if [[ "$diffusion_pixel_refinement" == "true" && "${PRISM_DIFFUSION_BACKEND:-auto}" != "objectclear" ]]; then
  diffusion_pixel_refinement_args=(--diffusion-pixel-refinement)
fi
diffusion_batch_size="${PRISM_DIFFUSION_BATCH_SIZE:-2}"
experiment_id="${PRISM_EXPERIMENT_ID:-prism-v8-full-temporal-kernel-seed${seed}}"
checkpoint_uri="${PRISM_CHECKPOINT_URI:-}"
restore_checkpoint_uri="${PRISM_RESTORE_CHECKPOINT_URI:-}"
restore_checkpoint_stages="${PRISM_RESTORE_CHECKPOINT_STAGES:-1a,1b,2,3,4}"
checkpoint_sync_seconds="${PRISM_CHECKPOINT_SYNC_SECONDS:-60}"
checkpoint_interval_steps="${PRISM_CHECKPOINT_INTERVAL_STEPS:-100}"
parameter_finiteness_interval_steps="${PRISM_PARAMETER_FINITENESS_INTERVAL_STEPS:-50}"
progress_interval_steps="${PRISM_PROGRESS_INTERVAL_STEPS:-50}"
wandb_scalar_interval_steps="${PRISM_WANDB_SCALAR_INTERVAL_STEPS:-10}"
wandb_image_interval="${PRISM_WANDB_IMAGE_INTERVAL:-200}"
validation_interval_early="${PRISM_VALIDATION_INTERVAL_EARLY:-1}"
validation_interval_stage34="${PRISM_VALIDATION_INTERVAL_STAGE34:-3}"
diffusion_cpu_offload="${PRISM_DIFFUSION_CPU_OFFLOAD:-auto}"
amp_dtype="${PRISM_AMP_DTYPE:-bfloat16}"
matte_frame_chunk_size="${PRISM_MATTE_FRAME_CHUNK_SIZE:-}"
sam2_temporal_chunk_size="${PRISM_SAM2_TEMPORAL_CHUNK_SIZE:-}"
sam2_temporal_detach_interval="${PRISM_SAM2_TEMPORAL_DETACH_INTERVAL:-0}"
paired_microbatch_checkpointing="${PRISM_PAIRED_MICROBATCH_CHECKPOINTING:-false}"
mam2_matter_backend="${PRISM_MAM2_MATTER_BACKEND:-external_mematte}"
skip_stage1_pretrained="${PRISM_SKIP_STAGE1_PRETRAINED:-false}"
official_sam2matting_root="${PRISM_OFFICIAL_SAM2MATTING_ROOT:-}"
official_sam2matting_checkpoint="${PRISM_OFFICIAL_SAM2MATTING_CHECKPOINT:-}"
official_sam2matting_variant="${PRISM_OFFICIAL_SAM2MATTING_VARIANT:-tiny}"
diffusion_backend="${PRISM_DIFFUSION_BACKEND:-auto}"
diffusion_revision="${PRISM_DIFFUSION_REVISION:-}"
objectclear_source_root="${PRISM_OBJECTCLEAR_SOURCE_ROOT:-}"
objectclear_late_denoise_steps="${PRISM_OBJECTCLEAR_LATE_DENOISE_STEPS:-0}"
objectclear_fused_output="${PRISM_OBJECTCLEAR_FUSED_OUTPUT:-false}"
objectclear_pixel_refinement="${PRISM_OBJECTCLEAR_PIXEL_REFINEMENT:-false}"
refinement_steps="${PRISM_REFINEMENT_STEPS:-3}"
mematte_root="${PRISM_MEMATTE_ROOT:-$repo_root/network/third_party/MEMatte}"
mematte_config="${PRISM_MEMATTE_CONFIG:-$mematte_root/configs/MEMatte_S_topk0.25_win_global_long.py}"
mematte_checkpoint="${PRISM_MEMATTE_CHECKPOINT:-$repo_root/network/checkpoints/MEMatte_ViTS_DIM.pth}"
mematte_max_tokens="${PRISM_MEMATTE_MAX_TOKENS:-12000}"
mematte_train_backbone="${PRISM_MEMATTE_TRAIN_BACKBONE:-true}"
runtime_contract_checks="${PRISM_RUNTIME_CONTRACT_CHECKS:-false}"
predict_geometry="${PRISM_PREDICT_GEOMETRY:-false}"
flow_refinement="${PRISM_FLOW_REFINEMENT:-false}"
foreground_refinement="${PRISM_FOREGROUND_REFINEMENT:-false}"
learned_object_support="${PRISM_LEARNED_OBJECT_SUPPORT:-false}"
independent_physical_alpha="${PRISM_INDEPENDENT_PHYSICAL_ALPHA:-false}"
loss_profile="${PRISM_LOSS_PROFILE:-standard}"
optical_curriculum="${PRISM_OPTICAL_CURRICULUM:-false}"
flow_refinement_width="${PRISM_FLOW_REFINEMENT_WIDTH:-32}"
flow_refinement_max_fraction="${PRISM_FLOW_REFINEMENT_MAX_FRACTION:-0.0625}"
foreground_refinement_width="${PRISM_FOREGROUND_REFINEMENT_WIDTH:-32}"
foreground_refinement_logit_scale="${PRISM_FOREGROUND_REFINEMENT_LOGIT_SCALE:-2.0}"
flow_guided_latent_bridge="${PRISM_FLOW_GUIDED_LATENT_BRIDGE:-false}"
latent_bridge_width="${PRISM_LATENT_BRIDGE_WIDTH:-32}"
latent_bridge_flow_fraction="${PRISM_LATENT_BRIDGE_FLOW_FRACTION:-0.0625}"
latent_bridge_foreground_logit_scale="${PRISM_LATENT_BRIDGE_FOREGROUND_LOGIT_SCALE:-1.0}"
latent_bridge_transmission_logit_scale="${PRISM_LATENT_BRIDGE_TRANSMISSION_LOGIT_SCALE:-1.0}"
use_fba_foreground_prior="${PRISM_USE_FBA_FOREGROUND_PRIOR:-false}"
use_mam2_foreground_prior="${PRISM_USE_MAM2_FOREGROUND_PRIOR:-false}"
fba_source_root="${PRISM_FBA_SOURCE_ROOT:-}"
fba_checkpoint="${PRISM_FBA_CHECKPOINT:-}"
fba_input_side="${PRISM_FBA_INPUT_SIDE:-320}"
fba_train_decoder="${PRISM_FBA_TRAIN_DECODER:-true}"
fba_decoder_lr_scale="${PRISM_FBA_DECODER_LR_SCALE:-0.1}"
background_pixel_refiner_lr_scale="${PRISM_BACKGROUND_PIXEL_REFINER_LR_SCALE:-1.0}"
evidence_flux_only="${PRISM_EVIDENCE_FLUX_ONLY:-true}"
background_exclusion_dilation="${PRISM_BACKGROUND_EXCLUSION_DILATION:-1}"
flux_direct_observation_threshold="${PRISM_FLUX_DIRECT_OBSERVATION_THRESHOLD:-0.1}"
flux_require_unanimous_direct_consensus="${PRISM_FLUX_REQUIRE_UNANIMOUS_DIRECT_CONSENSUS:-true}"
flux_min_consistent_observations="${PRISM_FLUX_MIN_CONSISTENT_OBSERVATIONS:-2}"
flux_spatial_singleton_evidence="${PRISM_FLUX_SPATIAL_SINGLETON_EVIDENCE:-false}"
refine_inverse_evidence="${PRISM_REFINE_INVERSE_EVIDENCE:-true}"
geometry_min_depth="${PRISM_GEOMETRY_MIN_DEPTH:-0.001}"
max_train_batches="${PRISM_MAX_TRAIN_BATCHES:-}"
partition_train_epochs="${PRISM_PARTITION_TRAIN_EPOCHS:-false}"
max_eval_batches="${PRISM_MAX_EVAL_BATCHES:-}"
run_final_evaluation="${PRISM_RUN_FINAL_EVALUATION:-true}"
keep_epoch_checkpoints="${PRISM_KEEP_EPOCH_CHECKPOINTS:-false}"
stage1_manifests="${PRISM_STAGE1_MANIFESTS:-}"
stage1_transfer="${PRISM_STAGE1_TRANSFER:-false}"

stage1a_epochs="${PRISM_STAGE1A_EPOCHS:-10}"
stage1b_epochs="${PRISM_STAGE1B_EPOCHS:-10}"
stage2_epochs="${PRISM_STAGE2_EPOCHS:-10}"
stage3_epochs="${PRISM_STAGE3_EPOCHS:-15}"
stage4_epochs="${PRISM_STAGE4_EPOCHS:-20}"
stage4_learning_rate="${PRISM_STAGE4_LR:-1e-4}"
stage4_initial_checkpoint="${PRISM_STAGE4_INITIAL_CHECKPOINT:-}"
stage2_learning_rate="${PRISM_STAGE2_LR:-1e-4}"
gradient_clip="${PRISM_GRADIENT_CLIP:-1.0}"

if [[ "$evidence_flux_only" == "true" && -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required before Stage 3 for evidence+FLUX training." >&2
  exit 4
fi
if [[ "$skip_stage1_pretrained" == "true" ]]; then
  if [[ "$mam2_matter_backend" != "official_sam2matting" \
    || ! -f "$official_sam2matting_checkpoint" \
    || ! -f "$official_sam2matting_root/sam2/sam2matting_video_predictor.py" ]]; then
    echo "Skipping Stage 1 requires verified pretrained SAM2Matting source and checkpoint" >&2
    exit 2
  fi
fi
if [[ "$diffusion_backend" == "objectclear" ]]; then
  if [[ ! -f "$objectclear_source_root/objectclear/pipelines/pipeline_objectclear.py" \
    || -z "$diffusion_revision" ]]; then
    echo "ObjectClear training requires its pinned source and model revision" >&2
    exit 2
  fi
fi

for scheduled_length in \
  "$image_pretrain_clip_length" \
  "$video_refine_clip_length" \
  "$final_eval_clip_length"; do
  if [[ ! "$scheduled_length" =~ ^[1-9][0-9]*$ ]]; then
    echo "PRISM clip lengths must be positive integers: $scheduled_length" >&2
    exit 2
  fi
done

IFS=',' read -r -a temporal_ablation_lengths <<< "$temporal_ablation_clip_lengths"
for scheduled_length in "${temporal_ablation_lengths[@]}"; do
  if [[ ! "$scheduled_length" =~ ^[1-9][0-9]*$ ]]; then
    echo "PRISM_TEMPORAL_ABLATION_CLIP_LENGTHS must contain positive integers." >&2
    exit 2
  fi
done

mkdir -p "$output_root/checkpoints" "$output_root/results"

wandb_entity_args=()
if [[ -n "$wandb_entity" ]]; then
  wandb_entity_args+=(--wandb-entity "$wandb_entity")
fi

geometry_args=(--no-predict-geometry)
if [[ "$predict_geometry" == "true" ]]; then
  geometry_args=(--predict-geometry --geometry-min-depth "$geometry_min_depth")
fi

flow_refinement_args=(
  --no-flow-refinement
  --flow-refinement-width "$flow_refinement_width"
  --flow-refinement-max-fraction "$flow_refinement_max_fraction"
)
if [[ "$flow_refinement" == "true" ]]; then
  flow_refinement_args[0]=--flow-refinement
fi

foreground_refinement_args=(
  --no-foreground-refinement
  --foreground-refinement-width "$foreground_refinement_width"
  --foreground-refinement-logit-scale "$foreground_refinement_logit_scale"
)
if [[ "$foreground_refinement" == "true" ]]; then
  foreground_refinement_args[0]=--foreground-refinement
fi
object_support_args=(--no-learned-object-support)
if [[ "$learned_object_support" == "true" ]]; then
  object_support_args=(--learned-object-support)
fi

latent_bridge_args=(
  --no-flow-guided-latent-bridge
  --latent-bridge-width "$latent_bridge_width"
  --latent-bridge-flow-fraction "$latent_bridge_flow_fraction"
  --latent-bridge-foreground-logit-scale "$latent_bridge_foreground_logit_scale"
  --latent-bridge-transmission-logit-scale "$latent_bridge_transmission_logit_scale"
)
if [[ "$flow_guided_latent_bridge" == "true" ]]; then
  latent_bridge_args[0]=--flow-guided-latent-bridge
fi
mam2_prior_args=(--no-use-mam2-foreground-prior)
if [[ "$use_mam2_foreground_prior" == "true" ]]; then
  if [[ "$flow_guided_latent_bridge" != "true" || "$use_fba_foreground_prior" == "true" ]]; then
    echo "MAM2 prior requires flow-guided bridge and excludes FBA prior" >&2
    exit 2
  fi
  mam2_prior_args=(--use-mam2-foreground-prior)
fi

fba_prior_args=(--no-use-fba-foreground-prior --fba-input-side "$fba_input_side" --no-fba-train-decoder)
if [[ "$use_fba_foreground_prior" == "true" ]]; then
  if [[ "$flow_guided_latent_bridge" != "true" ]]; then
    echo "PRISM_USE_FBA_FOREGROUND_PRIOR requires PRISM_FLOW_GUIDED_LATENT_BRIDGE=true" >&2
    exit 2
  fi
  if [[ -z "$fba_source_root" || -z "$fba_checkpoint" ]]; then
    echo "FBA prior requires PRISM_FBA_SOURCE_ROOT and PRISM_FBA_CHECKPOINT" >&2
    exit 2
  fi
  if [[ ! -f "$fba_source_root/networks/models.py" || ! -f "$fba_checkpoint" ]]; then
    echo "FBA source or checkpoint is absent; stage the official pretrained assets before allocating a GPU" >&2
    exit 2
  fi
  fba_prior_args=(--use-fba-foreground-prior --fba-source-root "$fba_source_root" --fba-checkpoint "$fba_checkpoint" --fba-input-side "$fba_input_side" --no-fba-train-decoder)
  if [[ "$fba_train_decoder" == "true" ]]; then
    fba_prior_args[7]=--fba-train-decoder
  fi
fi

evidence_flux_args=(--no-evidence-flux-only)
if [[ "$evidence_flux_only" == "true" ]]; then
  evidence_flux_args=(--evidence-flux-only)
fi
singleton_evidence_args=()
if [[ "$flux_spatial_singleton_evidence" == "true" ]]; then
  singleton_evidence_args=(--flux-spatial-singleton-evidence)
fi
consensus_args=(--flux-require-unanimous-direct-consensus)
if [[ "$flux_require_unanimous_direct_consensus" == "false" ]]; then
  consensus_args=(--no-flux-require-unanimous-direct-consensus)
elif [[ "$flux_require_unanimous_direct_consensus" != "true" ]]; then
  echo "PRISM_FLUX_REQUIRE_UNANIMOUS_DIRECT_CONSENSUS must be true or false" >&2
  exit 2
fi
if [[ ! "$flux_min_consistent_observations" =~ ^[1-9][0-9]*$ ]]; then
  echo "PRISM_FLUX_MIN_CONSISTENT_OBSERVATIONS must be positive" >&2
  exit 2
fi
echo "PRISM_FLUX_OBSERVATION_CONFIG threshold=$flux_direct_observation_threshold unanimous=$flux_require_unanimous_direct_consensus min_observations=$flux_min_consistent_observations singleton=$flux_spatial_singleton_evidence working_side=$diffusion_min_working_side"
inverse_refinement_args=(--refine-inverse-evidence)
if [[ "$refine_inverse_evidence" == "false" ]]; then
  inverse_refinement_args=(--no-refine-inverse-evidence)
  echo "PRISM_FLUX_PRESERVE_INVERSE_CONTEXT=true"
elif [[ "$refine_inverse_evidence" != "true" ]]; then
  echo "PRISM_REFINE_INVERSE_EVIDENCE must be true or false" >&2
  exit 2
fi

optical_curriculum_args=(--no-optical-curriculum)
if [[ "$optical_curriculum" == "true" ]]; then
  optical_curriculum_args=(--optical-curriculum)
fi

batch_limit_args=()
if [[ -n "$max_train_batches" ]]; then
  batch_limit_args+=(--max-train-batches "$max_train_batches")
fi
if [[ "$partition_train_epochs" == "true" ]]; then
  if [[ -z "$max_train_batches" ]]; then
    echo "PRISM_PARTITION_TRAIN_EPOCHS requires PRISM_MAX_TRAIN_BATCHES" >&2
    exit 2
  fi
  batch_limit_args+=(--partition-train-epochs)
elif [[ "$partition_train_epochs" != "false" ]]; then
  echo "PRISM_PARTITION_TRAIN_EPOCHS must be true or false" >&2
  exit 2
fi
if [[ -n "$max_eval_batches" ]]; then
  batch_limit_args+=(--max-eval-batches "$max_eval_batches")
fi

if [[ -n "$restore_checkpoint_uri" ]]; then
  IFS=',' read -r -a restore_stages <<< "$restore_checkpoint_stages"
  restore_args=()
  for restore_stage in "${restore_stages[@]}"; do
    restore_args+=(--stage "$restore_stage")
  done
  python "$script_dir/sync_prism_checkpoints.py" restore \
    --source-uri "$restore_checkpoint_uri" \
    --output-root "$output_root" \
    "${restore_args[@]}"
fi

if [[ "$stage1_transfer" == "true" ]]; then
  for stage in 1a 1b; do
    checkpoint="$output_root/checkpoints/stage$stage/prism_stage${stage}_best.pt"
    marker="$output_root/checkpoints/stage$stage/.training_complete"
    if [[ ! -s "$checkpoint" || ! -f "$marker" ]]; then
      echo "Stage 1 transfer requires a completed donor checkpoint: $checkpoint" >&2
      exit 4
    fi
  done
  echo "PRISM_STAGE1_DONOR_RESTORED_AND_VERIFIED"
fi

all_training_stages_complete=true
required_training_stages=(1a 1b 2 3 4)
if [[ "$skip_stage1_pretrained" == "true" ]]; then
  required_training_stages=(2 3 4)
fi
if [[ "$training_schedule" == "oracle_joint" ]]; then
  required_training_stages=(2 4)
  if [[ "$skip_stage1_pretrained" != "true" ]]; then
    required_training_stages=(1a 1b 2 4)
  fi
fi
for stage in "${required_training_stages[@]}"; do
  if [[ ! -f "$output_root/checkpoints/stage$stage/.training_complete" ]]; then
    all_training_stages_complete=false
    break
  fi
done

materialize_remote_components() {
  local components=("$@")
  local materialize_args=(
    --archive-volume "$archive_volume"
    --storage-name "$archive_storage_name"
    --download-root "$archive_download_root"
    --output-root "$materialized_root"
  )
  for component in "${components[@]}"; do
    materialize_args+=(--component "$component")
  done
  if [[ -n "$archive_max_shards" ]]; then
    materialize_args+=(--max-shards-per-component "$archive_max_shards")
  fi
  python "$script_dir/materialize_prism_archives.py" "${materialize_args[@]}"
  data_root="$materialized_root"
}

if [[ -n "$archive_volume" ]]; then
  if [[ "$materialize_all" == "true" ]]; then
    # Keep the full 64px dataset on local disk across all stages. The remote
    # materializer verifies every tar SHA-256 and deletes each tar after extract.
    materialize_remote_components metadata train validation test
    echo "PRISM_DATA_LOADING_MODE full_local data_root=$data_root"
  else
    # Validation is cached once while training uses bounded shard streaming.
    materialize_remote_components metadata validation
    echo "PRISM_DATA_LOADING_MODE shard_streaming data_root=$data_root"
  fi
elif [[ ! -f "$data_root/dataset_manifest.json" ]]; then
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

train_root="$data_root/train"
validation_root="$data_root/validation"
test_root="$data_root/test"

required_paths=("$data_root/dataset_manifest.json")
if [[ "$materialize_all" == "true" ]]; then
  required_paths+=("$train_root" "$validation_root" "$test_root")
elif [[ -z "$archive_volume" && "$all_training_stages_complete" == "true" ]]; then
  required_paths+=("$test_root")
elif [[ -z "$archive_volume" ]]; then
  required_paths+=("$train_root" "$validation_root")
fi
for required in "${required_paths[@]}"; do
  if [[ ! -e "$required" ]]; then
    echo "Required PRISM dataset path is missing: $required" >&2
    exit 2
  fi
done

mkdir -p \
  "$output_root/results" \
  "$output_root/wandb" \
  "$output_root/cache/wandb" \
  "$cache_root/huggingface" \
  "$cache_root/torch"

export OPENCV_IO_ENABLE_OPENEXR=1
export WANDB_DIR="$output_root/wandb"
export WANDB_CACHE_DIR="$output_root/cache/wandb"
export HF_HOME="${HF_HOME:-$cache_root/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$cache_root/torch}"
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
        if python "$script_dir/sync_prism_checkpoints.py" upload \
          --source "$checkpoint" \
          --destination-uri "$destination"; then
          uploaded_signatures[$relative]="$signature"
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

stream_args=()
if [[ -n "$archive_volume" && "$materialize_all" != "true" ]]; then
  stream_args=(
    --archive-volume "$archive_volume"
    --archive-storage-name "$archive_storage_name"
    --archive-cache-root "$shard_cache_root"
    --archive-shuffle-buffer "$shard_shuffle_buffer"
    --archive-download-retries "$shard_download_retries"
    --archive-decode-workers "$archive_decode_workers"
    --materialized-eval-data
  )
  if [[ -n "$archive_max_shards" ]]; then
    stream_args+=(--archive-max-shards "$archive_max_shards")
  fi
  if [[ "$archive_prefetch" == "true" ]]; then
    stream_args+=(--archive-prefetch)
  else
    stream_args+=(--no-archive-prefetch)
  fi
  # One loader process prepares CPU batches concurrently with CUDA training.
  # More than one would duplicate remote downloads and corrupt the shared cache.
  workers=1
fi

latest_epoch_checkpoint() {
  local stage_dir="$1"
  local stage="$2"
  local candidate=""
  local resume_checkpoint="$stage_dir/prism_stage${stage}_resume.pt"
  if [[ -f "$resume_checkpoint" && -s "$resume_checkpoint" ]]; then
    printf '%s' "$resume_checkpoint"
    return
  fi
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
  local stage_clip_length="$video_refine_clip_length"
  local stage_schedule_tag="video-refine"
  local validation_interval="$validation_interval_stage34"
  if [[ "$stage" == "1a" || "$stage" == "1b" || "$stage" == "2" ]]; then
    stage_clip_length="$image_pretrain_clip_length"
    stage_schedule_tag="image-pretrain"
    validation_interval="$validation_interval_early"
  fi
  local stage_matte_chunk_size="${matte_frame_chunk_size:-$stage_clip_length}"
  local stage_sam2_chunk_size="${sam2_temporal_chunk_size:-$stage_clip_length}"
  local -a physical_alpha_args=()
  if [[ "$stage" == "4" && "$independent_physical_alpha" == "true" ]]; then
    physical_alpha_args=(--independent-physical-alpha)
    echo "PRISM_INDEPENDENT_PHYSICAL_ALPHA_ENABLED stage=$stage"
  fi
  local stage_learning_rate="1e-4"
  if [[ "$stage" == "2" ]]; then
    stage_learning_rate="$stage2_learning_rate"
  elif [[ "$stage" == "4" ]]; then
    stage_learning_rate="$stage4_learning_rate"
  fi

  mkdir -p "$stage_dir"
  if [[ -f "$complete_marker" \
    && -f "$stage_dir/prism_stage${stage}_best.pt" \
    && -s "$stage_dir/prism_stage${stage}_best.pt" ]]; then
    if [[ -n "${PRISM_SGS_STAGE234_DONOR_RUN:-}" \
      && ( "$stage" == "2" || "$stage" == "3" ) ]]; then
      # A completed donor from the older pinned SGS code does not store
      # the later background-evidence config fields. These stages are only
      # skipped, never loaded for the new Stage 4 fine-tune; verify the
      # training contract they actually carried.
      python "$script_dir/verify_prism_stage_checkpoint.py" \
        "$stage_dir/prism_stage${stage}_best.pt" \
        --expect arguments stage "$stage" \
        --expect arguments loss_profile "$loss_profile" \
        --expect arguments optical_curriculum "$optical_curriculum"
      echo "PRISM_LEGACY_DONOR_STAGE_VERIFIED stage=$stage donor=$PRISM_SGS_STAGE234_DONOR_RUN"
      return
    fi
    if [[ "$stage" == "2" || "$stage" == "3" || "$stage" == "4" ]]; then
      local expected_flux=false
      local expected_completion=base
      local expected_diffusion_side=0
      local expected_fba_decoder=false
      local -a flux_observation_guards=()
      local -a physical_alpha_checkpoint_guard=()
      if [[ "$stage" == "4" ]]; then
        physical_alpha_checkpoint_guard=(
          --expect matter independent_physical_alpha "$independent_physical_alpha"
        )
      fi
      if [[ "$use_fba_foreground_prior" == "true" ]]; then
        expected_fba_decoder="$fba_train_decoder"
      fi
      if [[ "$evidence_flux_only" == "true" && ( "$stage" == "3" || "$stage" == "4" ) ]]; then
        expected_flux=true
        expected_completion=diffusion
        expected_diffusion_side="$diffusion_min_working_side"
        flux_observation_guards=(
          --expect background flux_require_unanimous_direct_consensus "$flux_require_unanimous_direct_consensus"
          --expect background flux_min_consistent_observations "$flux_min_consistent_observations"
        )
      fi
      python "$script_dir/verify_prism_stage_checkpoint.py" \
        "$stage_dir/prism_stage${stage}_best.pt" \
        --expect arguments stage "$stage" \
        --expect arguments clip_length "$stage_clip_length" \
        --expect arguments loss_profile "$loss_profile" \
        --expect arguments optical_curriculum "$optical_curriculum" \
        --expect arguments fba_input_side "$fba_input_side" \
        --expect arguments fba_train_decoder "$expected_fba_decoder" \
        --expect matter flow_guided_latent_bridge "$flow_guided_latent_bridge" \
        --expect matter use_fba_foreground_prior "$use_fba_foreground_prior" \
        --expect matter use_mam2_foreground_prior "$use_mam2_foreground_prior" \
        --expect matter refine_refractive_flow "$flow_refinement" \
        --expect matter refine_straight_foreground "$foreground_refinement" \
        --expect matter learned_object_support "$learned_object_support" \
        --expect matter predict_geometry "$predict_geometry" \
        "${physical_alpha_checkpoint_guard[@]}" \
        --expect background evidence_flux_only "$expected_flux" \
        --expect background completion_variant "$expected_completion" \
        --expect background diffusion_min_working_side "$expected_diffusion_side" \
        --expect background refine_inverse_evidence "$refine_inverse_evidence" \
        "${flux_observation_guards[@]}"
    fi
    echo "Stage $stage is already complete; reusing $stage_dir/prism_stage${stage}_best.pt"
    return
  fi

  echo "PRISM_STAGE_SCHEDULE stage=$stage clip_length=$stage_clip_length mode=$stage_schedule_tag paired=$paired"

  resume_checkpoint="$(latest_epoch_checkpoint "$stage_dir" "$stage")"
  local checkpoint_args=()
  local stage1_transfer_args=()
  local stage_fba_args=("${fba_prior_args[@]}")
  local stage_mam2_args=("${mam2_prior_args[@]}")
  if [[ "$stage" == "1a" || "$stage" == "1b" ]]; then
    # Stage 1 trains only MAM2. Avoid loading FBA weights and occupying GPU
    # memory before the flow-guided operator is trained in Stage 2.
    stage_fba_args=(--no-use-fba-foreground-prior --fba-input-side "$fba_input_side" --no-fba-train-decoder)
    stage_mam2_args=(--no-use-mam2-foreground-prior)
  fi
  if [[ -n "$resume_checkpoint" ]]; then
    checkpoint_args=(--resume "$resume_checkpoint")
    echo "Resuming Stage $stage from $resume_checkpoint"
  elif [[ -n "$previous_checkpoint" ]]; then
    if [[ ! -f "$previous_checkpoint" || ! -s "$previous_checkpoint" ]]; then
      echo "Previous Stage checkpoint is missing: $previous_checkpoint" >&2
      exit 3
    fi
    checkpoint_args=(--checkpoint "$previous_checkpoint")
    if [[ "$stage" == "2" && ( "$stage1_transfer" == "true" || "$use_fba_foreground_prior" == "true" ) ]]; then
      stage1_transfer_args=(--checkpoint-mam2-only)
    fi
  fi

  local pair_args=()
  if [[ "$paired" == "true" ]]; then
    pair_args=(--paired-backgrounds)
  fi

  local paired_checkpoint_args=(--no-paired-microbatch-checkpointing)
  if [[ "$paired_microbatch_checkpointing" == "true" ]]; then
    paired_checkpoint_args=(--paired-microbatch-checkpointing)
  fi

  local test_args=()
  if [[ "$final_mode" == "both" ]]; then
    test_args=(--test-data "$test_root" --paired-eval --compute-lpips)
  fi

  local mematte_backbone_args=(--mematte-train-backbone)
  if [[ "$mematte_train_backbone" != "true" ]]; then
    mematte_backbone_args=(--no-mematte-train-backbone)
  fi

  local contract_args=(--no-runtime-contract-checks)
  if [[ "$runtime_contract_checks" == "true" ]]; then
    contract_args=(--runtime-contract-checks)
  fi

  local epoch_checkpoint_args=(--no-keep-epoch-checkpoints)
  if [[ "$keep_epoch_checkpoints" == "true" ]]; then
    epoch_checkpoint_args=(--keep-epoch-checkpoints)
  fi

  local stage1_manifest_args=()
  if [[ "$stage" == "1a" || "$stage" == "1b" ]]; then
    IFS=',' read -r -a stage1_manifest_paths <<< "$stage1_manifests"
    for manifest_path in "${stage1_manifest_paths[@]}"; do
      if [[ -n "$manifest_path" ]]; then
        stage1_manifest_args+=(--stage1-manifest "$manifest_path")
      fi
    done
  fi

  local completion_args=(--completion-variant base --no-evidence-flux-only)
  local completion_tag="prism-base"
  local official_backend_args=()
  if [[ "$mam2_matter_backend" == "official_sam2matting" ]]; then
    official_backend_args=(
      --official-sam2matting-root "$official_sam2matting_root"
      --official-sam2matting-checkpoint "$official_sam2matting_checkpoint"
      --official-sam2matting-variant "$official_sam2matting_variant"
    )
  fi
  if [[ "$evidence_flux_only" == "true" && ( "$stage" == "3" || "$stage" == "4" ) ]]; then
    completion_tag="frozen-flux-every-train-forward"
    completion_args=(
      --completion-variant diffusion
      --evidence-flux-only
      --diffusion-model "$diffusion_model"
      --diffusion-steps "$diffusion_steps"
      --diffusion-guidance-scale "$diffusion_guidance_scale"
      --diffusion-mask-dilation "$diffusion_mask_dilation"
      --diffusion-min-working-side "$diffusion_min_working_side"
      "${diffusion_pixel_refinement_args[@]}"
      --diffusion-dtype bfloat16
    )
    if [[ "$diffusion_backend" == "objectclear" ]]; then
      completion_tag="frozen-objectclear-flow-guided"
      local objectclear_output_arg=--no-objectclear-fused-output
      if [[ "$objectclear_fused_output" == "true" ]]; then
        objectclear_output_arg=--objectclear-fused-output
      fi
      local objectclear_pixel_arg=--no-objectclear-pixel-refinement
      if [[ "$objectclear_pixel_refinement" == "true" ]]; then
        objectclear_pixel_arg=--objectclear-pixel-refinement
      fi
      completion_args+=(
        --diffusion-backend objectclear
        --diffusion-revision "$diffusion_revision"
        --objectclear-source-root "$objectclear_source_root"
        --objectclear-dtype bfloat16
        "$objectclear_output_arg"
        --objectclear-late-denoise-steps "$objectclear_late_denoise_steps"
        "$objectclear_pixel_arg"
      )
    fi
    echo "PRISM_FLUX_TRAIN_FORWARD_CONFIGURED stage=$stage steps=$diffusion_steps optical_curriculum=$optical_curriculum"
  fi

  WANDB_RUN_ID="${experiment_id}-stage${stage}" \
  WANDB_RESUME=allow \
  prism-train \
    "${stream_args[@]}" \
    --train-data "$train_root" \
    --val-data "$validation_root" \
    "${test_args[@]}" \
    "${checkpoint_args[@]}" \
    "${stage1_transfer_args[@]}" \
    --save-dir "$stage_dir" \
    --stage "$stage" \
    --loss-profile "$loss_profile" \
    "${optical_curriculum_args[@]}" \
    --mode "$final_mode" \
    --epochs "$epochs" \
    --validate-every-epochs "$validation_interval" \
    --checkpoint-interval-steps "$checkpoint_interval_steps" \
    --parameter-finiteness-interval-steps "$parameter_finiteness_interval_steps" \
    --progress-interval-steps "$progress_interval_steps" \
    "${contract_args[@]}" \
    "${epoch_checkpoint_args[@]}" \
    "${stage1_manifest_args[@]}" \
    --batch-size "$batch_size" \
    --clip-length "$stage_clip_length" \
    --refinement-steps "$refinement_steps" \
    --workers "$workers" \
    --eval-workers "$eval_workers" \
    --lr "$stage_learning_rate" \
    --fba-decoder-lr-scale "$fba_decoder_lr_scale" \
    --background-pixel-refiner-lr-scale "$background_pixel_refiner_lr_scale" \
    --scheduler cosine \
    --gradient-clip "$gradient_clip" \
    --cudnn-benchmark \
    --amp-dtype "$amp_dtype" \
    --activation-checkpointing \
    --matte-full-activation-checkpointing \
    --matte-frame-chunk-size "$stage_matte_chunk_size" \
    --sam2-temporal-activation-checkpointing \
    --sam2-temporal-checkpoint-chunk-size "$stage_sam2_chunk_size" \
    --sam2-temporal-detach-interval "$sam2_temporal_detach_interval" \
    --mam2-matter-backend "$mam2_matter_backend" \
    "${official_backend_args[@]}" \
    --mematte-root "$mematte_root" \
    --mematte-config "$mematte_config" \
    --mematte-checkpoint "$mematte_checkpoint" \
    --mematte-max-tokens "$mematte_max_tokens" \
    "${mematte_backbone_args[@]}" \
    "${paired_checkpoint_args[@]}" \
    --prompt-mode point \
    --prompt-seed "$seed" \
    --seed "$seed" \
    "${completion_args[@]}" \
    --diffusion-cpu-offload "$diffusion_cpu_offload" \
    --completion-backbone ffc \
    --background-exclusion-dilation "$background_exclusion_dilation" \
    --flux-direct-observation-threshold "$flux_direct_observation_threshold" \
    "${consensus_args[@]}" \
    --flux-min-consistent-observations "$flux_min_consistent_observations" \
    "${singleton_evidence_args[@]}" \
    "${inverse_refinement_args[@]}" \
    --wandb-mode "$wandb_mode" \
    --wandb-project "$wandb_project" \
    "${wandb_entity_args[@]}" \
    --wandb-group "$wandb_group" \
    --wandb-scalar-interval-steps "$wandb_scalar_interval_steps" \
    --wandb-run-name "${experiment_id}-stage${stage}" \
    --wandb-tags full-fresnel "stage${stage}" "$completion_tag" "$stage_schedule_tag" "clip-t${stage_clip_length}" \
    --wandb-image-interval "$wandb_image_interval" \
    --wandb-image-limit 2 \
    "${pair_args[@]}" \
    "${geometry_args[@]}" \
    "${flow_refinement_args[@]}" \
    "${foreground_refinement_args[@]}" \
    "${physical_alpha_args[@]}" \
    "${object_support_args[@]}" \
    "${latent_bridge_args[@]}" \
    "${stage_mam2_args[@]}" \
    "${stage_fba_args[@]}" \
    "${batch_limit_args[@]}"

  touch "$complete_marker"
  sync
}

stage1a_best="$output_root/checkpoints/stage1a/prism_stage1a_best.pt"
stage1b_best="$output_root/checkpoints/stage1b/prism_stage1b_best.pt"
stage2_best="$output_root/checkpoints/stage2/prism_stage2_best.pt"
stage3_best="$output_root/checkpoints/stage3/prism_stage3_best.pt"
stage4_best="$output_root/checkpoints/stage4/prism_stage4_best.pt"

stage2_initial_checkpoint="$stage1b_best"
if [[ "$skip_stage1_pretrained" == "true" ]]; then
  stage2_initial_checkpoint=""
  echo "PRISM_PRETRAINED_STAGE1_SKIPPED backend=$mam2_matter_backend"
else
  run_stage 1a "$stage1a_epochs" 1 "" false train
  run_stage 1b "$stage1b_epochs" 1 "$stage1a_best" false train
fi
if [[ "$stage2_paired_backgrounds" == "true" ]]; then
  run_stage 2 "$stage2_epochs" 2 "$stage2_initial_checkpoint" true train
else
  run_stage 2 "$stage2_epochs" 1 "$stage2_initial_checkpoint" false train
fi
if [[ "$training_schedule" == "oracle_joint" ]]; then
  # Pretrained matting and completion do not need separate fitting stages.
  # Briefly anchor the optical operator to GT B, then jointly update PAM,
  # the matte alpha heads, and the completion adapter from Stage 2 weights.
  echo "PRISM_TWO_PHASE_SCHEDULE stage2=oracle-background stage4=joint stage3=skipped"
  run_stage 4 "$stage4_epochs" 2 "${stage4_initial_checkpoint:-$stage2_best}" true train
else
  run_stage 3 "$stage3_epochs" 2 "$stage2_best" true train
  run_stage 4 "$stage4_epochs" 2 "${stage4_initial_checkpoint:-$stage3_best}" true train
fi

if [[ "$run_final_evaluation" != "true" ]]; then
  echo "PRISM_TRAINING_SMOKE_COMPLETE final_evaluation=skipped output=$output_root"
  exit 0
fi

if [[ -n "$archive_volume" && "$materialize_all" != "true" ]]; then
  # Test is materialized only after training, avoiding both repeated downloads
  # and unnecessary disk use during the multi-stage optimization.
  materialize_remote_components test
  test_root="$data_root/test"
fi

if [[ -z "$archive_volume" && ! -d "$test_root" ]]; then
  echo "Required PRISM test dataset path is missing: $test_root" >&2
  exit 2
fi

evaluation_physical_alpha_args=()
if [[ "$independent_physical_alpha" == "true" ]]; then
  evaluation_physical_alpha_args=(--independent-physical-alpha)
fi

run_base_evaluation() {
  local eval_clip_length="$1"
  local eval_dir="$2"
  local run_suffix="$3"
  local eval_marker="$eval_dir/.evaluation_complete"
  local eval_matte_chunk_size="${matte_frame_chunk_size:-$eval_clip_length}"
  local eval_sam2_chunk_size="${sam2_temporal_chunk_size:-$eval_clip_length}"
  mkdir -p "$eval_dir"
  if [[ -f "$eval_marker" ]]; then
    echo "Base evaluation T=$eval_clip_length is already complete; reusing $eval_dir"
    return
  fi
  echo "PRISM_TEMPORAL_ABLATION_START clip_length=$eval_clip_length output=$eval_dir"
  WANDB_RUN_ID="${experiment_id}-${run_suffix}" \
  WANDB_RESUME=allow \
  prism-train \
    "${stream_args[@]}" \
    --test-data "$test_root" \
    --checkpoint "$stage4_best" \
    --save-dir "$eval_dir" \
    --stage 4 \
    --loss-profile "$loss_profile" \
    --mode test \
    "${evaluation_physical_alpha_args[@]}" \
    --batch-size 2 \
    --clip-length "$eval_clip_length" \
    --workers "$workers" \
    --eval-workers "$eval_workers" \
    --prompt-mode point \
    --prompt-seed "$seed" \
    --seed "$seed" \
    --completion-variant base \
    --completion-backbone ffc \
    --background-exclusion-dilation "$background_exclusion_dilation" \
    --flux-direct-observation-threshold "$flux_direct_observation_threshold" \
    "${consensus_args[@]}" \
    --flux-min-consistent-observations "$flux_min_consistent_observations" \
    "${singleton_evidence_args[@]}" \
    "${inverse_refinement_args[@]}" \
    --amp-dtype "$amp_dtype" \
    --matte-frame-chunk-size "$eval_matte_chunk_size" \
    --sam2-temporal-checkpoint-chunk-size "$eval_sam2_chunk_size" \
    --sam2-temporal-detach-interval "$sam2_temporal_detach_interval" \
    --paired-eval \
    --compute-lpips \
    --compute-connectivity \
    --wandb-mode "$wandb_mode" \
    --wandb-project "$wandb_project" \
    "${wandb_entity_args[@]}" \
    --wandb-group "$wandb_group" \
    --wandb-run-name "${experiment_id}-${run_suffix}" \
    --wandb-tags full-fresnel stage4 prism-base temporal-ablation "clip-t${eval_clip_length}" \
    --wandb-image-limit 2 \
    "${geometry_args[@]}" \
    "${flow_refinement_args[@]}" \
    "${foreground_refinement_args[@]}" \
    "${object_support_args[@]}" \
    "${latent_bridge_args[@]}" \
    "${mam2_prior_args[@]}" \
    "${fba_prior_args[@]}" \
    "${batch_limit_args[@]}"
  touch "$eval_marker"
  sync
  echo "PRISM_TEMPORAL_ABLATION_COMPLETE clip_length=$eval_clip_length output=$eval_dir"
}

# The canonical PRISM-Base result uses the longest requested evaluation clip,
# while matched shorter-clip runs quantify how temporal evidence changes
# background recovery and operator disentanglement.
if [[ "$evidence_flux_only" == "true" ]]; then
  echo "PRISM_EVIDENCE_FLUX_ONLY: skipping learned-base and temporal ablation evaluations"
else
  run_base_evaluation \
    "$final_eval_clip_length" \
    "$output_root/results/prism-base" \
    "base-eval-t${final_eval_clip_length}"
  for ablation_clip_length in "${temporal_ablation_lengths[@]}"; do
    if [[ "$ablation_clip_length" == "$final_eval_clip_length" ]]; then
      continue
    fi
    run_base_evaluation \
      "$ablation_clip_length" \
      "$output_root/results/temporal-ablation/t${ablation_clip_length}" \
      "base-eval-t${ablation_clip_length}"
  done
fi

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
    "${stream_args[@]}" \
    --test-data "$test_root" \
    --checkpoint "$stage4_best" \
    --save-dir "$diffusion_dir" \
    --stage 4 \
    --loss-profile "$loss_profile" \
    --mode test \
    "${evaluation_physical_alpha_args[@]}" \
    --batch-size "$diffusion_batch_size" \
    --clip-length "$final_eval_clip_length" \
    --workers "$workers" \
    --eval-workers "$eval_workers" \
    --prompt-mode point \
    --prompt-seed "$seed" \
    --seed "$seed" \
    --completion-variant diffusion \
    --completion-backbone ffc \
    --background-exclusion-dilation "$background_exclusion_dilation" \
    --flux-direct-observation-threshold "$flux_direct_observation_threshold" \
    "${consensus_args[@]}" \
    --flux-min-consistent-observations "$flux_min_consistent_observations" \
    "${singleton_evidence_args[@]}" \
    "${inverse_refinement_args[@]}" \
    --diffusion-model "$diffusion_model" \
    --diffusion-steps "$diffusion_steps" \
    --diffusion-guidance-scale 30 \
    --diffusion-mask-dilation "$diffusion_mask_dilation" \
    --diffusion-min-working-side "$diffusion_min_working_side" \
    "${diffusion_pixel_refinement_args[@]}" \
    --diffusion-dtype bfloat16 \
    --amp-dtype "$amp_dtype" \
    --matte-frame-chunk-size "${matte_frame_chunk_size:-$final_eval_clip_length}" \
    --sam2-temporal-checkpoint-chunk-size "${sam2_temporal_chunk_size:-$final_eval_clip_length}" \
    --sam2-temporal-detach-interval "$sam2_temporal_detach_interval" \
    --paired-eval \
    --compute-lpips \
    --compute-connectivity \
    --wandb-mode "$wandb_mode" \
    --wandb-project "$wandb_project" \
    "${wandb_entity_args[@]}" \
    --wandb-group "$wandb_group" \
    --wandb-run-name "${experiment_id}-diffusion-flux-fill" \
    --wandb-tags full-fresnel stage4 prism-diffusion flux-fill "clip-t${final_eval_clip_length}" \
    --wandb-image-limit 2 \
    "${geometry_args[@]}" \
    "${flow_refinement_args[@]}" \
    "${foreground_refinement_args[@]}" \
    "${object_support_args[@]}" \
    "${latent_bridge_args[@]}" \
    "${mam2_prior_args[@]}" \
    "${fba_prior_args[@]}" \
    "${evidence_flux_args[@]}" \
    "${batch_limit_args[@]}"
  touch "$diffusion_marker"
  sync
fi

echo "PRISM_ALL_STAGES_COMPLETE output=$output_root"
