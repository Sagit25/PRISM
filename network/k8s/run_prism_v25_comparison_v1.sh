#!/usr/bin/env bash
set -Eeuo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
run_name="${PRISM_SGS_RUN_NAME:?PRISM_SGS_RUN_NAME is required}"
count="${PRISM_COMPARISON_MAX_SEQUENCES:-8}"
if ! [[ "$count" =~ ^[0-9]+$ ]]; then
  echo "PRISM_COMPARISON_MAX_SEQUENCES must be nonnegative" >&2
  exit 2
fi
export PRISM_COMPARISON_EXPORT_ROOT="/data/comparisons/prism-test64-v1/$run_name/prism_v25"
export PRISM_COMPARISON_EXPORT_FRAME_INDEX=4
if [[ "$count" == 0 ]]; then
  export PRISM_COMPARISON_QUALITATIVE_LIMIT=960
  unset PRISM_MAX_FINAL_EVAL_BATCHES
else
  export PRISM_COMPARISON_QUALITATIVE_LIMIT="$count"
  export PRISM_MAX_FINAL_EVAL_BATCHES="$(( (count + 1) / 2 ))"
fi
mkdir -p "$PRISM_COMPARISON_EXPORT_ROOT"
bash network/k8s/run_prism_sgs_v7r1.sh
test -f "/data/checkpoints/$run_name/results/prism-diffusion-flux-fill/.evaluation_complete"
python -m pip install --quiet scikit-image
python network/scripts/align_comparison_predictions.py \
  --data-root /data/datasets/prism-full-64-materialized-v7r1 \
  --prediction-root "$PRISM_COMPARISON_EXPORT_ROOT" \
  --max-sequences "$count"
python network/scripts/evaluate_comparison_predictions.py \
  --data-root /data/datasets/prism-full-64-materialized-v7r1 \
  --prediction-root "$PRISM_COMPARISON_EXPORT_ROOT/predictions" \
  --output-root "$PRISM_COMPARISON_EXPORT_ROOT" \
  --frame-index 4 --max-sequences "$count" \
  --model-name prism_v25 --prompt-policy 'PRISM point prompt on first frame; no oracle test target'
echo "PRISM_V25_COMPARISON_COMPLETE $PRISM_COMPARISON_EXPORT_ROOT"
