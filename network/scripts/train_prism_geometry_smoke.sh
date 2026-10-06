#!/usr/bin/env bash
set -Eeuo pipefail

# End-to-end geometry smoke profile. It exercises the same Stage 1A-4 code,
# official MAM2 assets, shard streaming, checkpoints, and W&B panels as the
# full experiment, but bounds both data transfer and optimizer/eval work.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export PRISM_PREDICT_GEOMETRY="${PRISM_PREDICT_GEOMETRY:-true}"
export PRISM_GEOMETRY_MIN_DEPTH="${PRISM_GEOMETRY_MIN_DEPTH:-0.001}"
export PRISM_ARCHIVE_MAX_SHARDS_PER_COMPONENT="${PRISM_ARCHIVE_MAX_SHARDS_PER_COMPONENT:-1}"
export PRISM_STAGE1A_EPOCHS="${PRISM_STAGE1A_EPOCHS:-1}"
export PRISM_STAGE1B_EPOCHS="${PRISM_STAGE1B_EPOCHS:-1}"
export PRISM_STAGE2_EPOCHS="${PRISM_STAGE2_EPOCHS:-1}"
export PRISM_STAGE3_EPOCHS="${PRISM_STAGE3_EPOCHS:-1}"
export PRISM_STAGE4_EPOCHS="${PRISM_STAGE4_EPOCHS:-1}"
export PRISM_MAX_TRAIN_BATCHES="${PRISM_MAX_TRAIN_BATCHES:-2}"
export PRISM_MAX_EVAL_BATCHES="${PRISM_MAX_EVAL_BATCHES:-2}"
# Match the publication schedule: learn semantics/optics/geometry from single
# frames, then exercise temporal background recovery on the full four-frame
# miniature sequence.
export PRISM_IMAGE_PRETRAIN_CLIP_LENGTH="${PRISM_IMAGE_PRETRAIN_CLIP_LENGTH:-1}"
export PRISM_VIDEO_REFINE_CLIP_LENGTH="${PRISM_VIDEO_REFINE_CLIP_LENGTH:-4}"
export PRISM_FINAL_EVAL_CLIP_LENGTH="${PRISM_FINAL_EVAL_CLIP_LENGTH:-4}"
export PRISM_TEMPORAL_ABLATION_CLIP_LENGTHS="${PRISM_TEMPORAL_ABLATION_CLIP_LENGTHS:-1,2,4}"
export PRISM_STAGE2_PAIRED_BACKGROUNDS="${PRISM_STAGE2_PAIRED_BACKGROUNDS:-true}"
export PRISM_CHECKPOINT_INTERVAL_STEPS="${PRISM_CHECKPOINT_INTERVAL_STEPS:-1}"
export PRISM_RUNTIME_CONTRACT_CHECKS="${PRISM_RUNTIME_CONTRACT_CHECKS:-true}"
export PRISM_RUN_FINAL_EVALUATION="${PRISM_RUN_FINAL_EVALUATION:-false}"
export PRISM_OUTPUT_ROOT="${PRISM_OUTPUT_ROOT:-/output/prism-geometry-smoke-v18}"
export PRISM_EXPERIMENT_ID="${PRISM_EXPERIMENT_ID:-prism-geometry-smoke-v18}"
export PRISM_WANDB_GROUP="${PRISM_WANDB_GROUP:-prism-geometry-smoke-v18}"

# Never restore a non-geometry checkpoint into this isolated smoke profile.
export PRISM_RESTORE_CHECKPOINT_URI=""

exec "$script_dir/train_prism_all_stages.sh" "$@"
