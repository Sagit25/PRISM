#!/usr/bin/env bash
set -Eeuo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
run_name="${PRISM_BASELINE_RUN_NAME:?PRISM_BASELINE_RUN_NAME is required}"
expected_commit="${PRISM_BASELINE_EXPECTED_COMMIT:?PRISM_BASELINE_EXPECTED_COMMIT is required}"
test_limit="${PRISM_BASELINE_MAX_SEQUENCES:-1}"
data_root=/data/datasets/prism-full-64-materialized-v7r1
output_root="/data/comparisons/prism-test64-v1/$run_name/flux_fill_hardmask"
export PYTHONUNBUFFERED=1 HF_HOME=/data/cache/prism-shared-models/huggingface
export TMPDIR=/data/cache/prism-baseline-tmp PIP_NO_CACHE_DIR=1
export PYTHONPATH="/data/code/SAM2-official-baseline:${PYTHONPATH:-}"
mkdir -p "$output_root" "$HF_HOME" "$TMPDIR"
test "${repo_root##*/}" = "PRISM-flux-hardmask-v1-$expected_commit"
test -f /data/code/SAM2-official-baseline/sam2/__init__.py
test -d "$data_root/test"
test -n "${WANDB_API_KEY:-}"
test -n "${HF_TOKEN:-}"
echo "PRISM_FLUX_HARDMASK_CODE_READY $expected_commit"
python -m pip install --quiet -e './network[data,sam2,experiment,evaluation,diffusion]'
python network/scripts/evaluate_flux_fill_hardmask_background.py \
  --data-root "$data_root" --output-root "$output_root" \
  --frame-index 4 --working-side "${PRISM_BASELINE_WORKING_SIDE:-256}" \
  --steps "${PRISM_BASELINE_STEPS:-8}" --max-sequences "$test_limit" \
  --run-name "$run_name-flux-hardmask"
python network/scripts/evaluate_comparison_predictions.py \
  --data-root "$data_root" --prediction-root "$output_root/predictions" \
  --output-root "$output_root" --frame-index 4 --max-sequences "$test_limit" \
  --model-name flux_fill_hardmask --prompt-policy 'oracle frame-4 object hard mask; one RGB image; no flow or temporal evidence'
echo "PRISM_FLUX_HARDMASK_COMPARISON_COMPLETE $output_root"
if [[ "$test_limit" == 0 ]]; then
  final_root=/data/comparisons/prism-test64-v1/latest-sota-20261104
  mkdir -p "$final_root"
  summaries=("FLUX_Fill_hardmask=$output_root/summary.json")
  predictions=("FLUX Fill hard mask=$output_root/predictions")
  add_complete_method() {
    local name="$1" source="$2"
    if [[ ! -f "$source/summary.json" ]]; then return 0; fi
    if python - "$source/summary.json" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
raise SystemExit(0 if payload.get('sequences') == 960 else 1)
PY
    then
      summaries+=("$name=$source/summary.json")
      predictions+=("$name=$source/predictions")
    fi
  }
  comparison_root=/data/comparisons/prism-test64-v1
  add_complete_method PRISM_v26 "$comparison_root/prism-kubectl-v26-comparison-full-v2/prism_v26"
  add_complete_method PRISM_joint_v15 "$comparison_root/prism-kubectl-joint-v15-comparison-v1-full/joint_v15_flow"
  add_complete_method ROSE "$comparison_root/prism-kubectl-rose-comparison-v9-full/rose"
  add_complete_method ObjectClear "$comparison_root/prism-kubectl-objectclear-comparison-v1-full/objectclear"
  add_complete_method MatAnyone2 "$comparison_root/prism-kubectl-matanyone2-comparison-v2-full/matanyone2"
  add_complete_method MAM2 "$comparison_root/prism-kubectl-mam2-comparison-v7-full/mam2"
  add_complete_method BiRefNet "$comparison_root/prism-kubectl-birefnet-matting-v4-gated-full-full/birefnet_matting"
  python network/scripts/merge_comparison_summaries.py \
    --summary "${summaries[@]}" --output "$final_root/metrics_960.csv"
  python network/scripts/make_comparison_contact_sheets.py \
    --data-root "$data_root" --output-root "$final_root/matched_images" \
    --frame-index 4 --indices 0 1 2 3 4 5 6 7 120 240 480 720 \
    --prediction "${predictions[@]}"
  printf '%s\n' "${summaries[@]}" > "$final_root/sources.txt"
  touch "$final_root/.comparison_complete"
  echo "PRISM_LATEST_SOTA_COMPARISON_COMPLETE $final_root"
fi
