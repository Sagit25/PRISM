#!/usr/bin/env bash
set -Eeuo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
export DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
apt-get update -qq
apt-get install -y -qq git libgl1 libglib2.0-0
test "$(git rev-parse HEAD)" = "${PRISM_SGS_EXPECTED_COMMIT:?}"
python -m pip install --quiet -e './network[data,evaluation]' scikit-image
SAM2_BUILD_CUDA=0 PYTHON_BIN=python \
  network/scripts/install_official_sam2.sh \
  "$repo_root/network/third_party/sam2" \
  2b90b9f5ceec907a1c18123530e92e794ad901a4 none

base=/data/comparisons/prism-test64-v1
data=/data/datasets/prism-full-64-materialized-v7r1
output="$base/extended_metrics_v1"
mkdir -p "$output"

# Each entry is NAME=relative prediction directory. The original summary's
# sequence count is the gate, so partial/failed methods are never mislabeled.
methods=(
  'Telea=prism-kubectl-baseline-comparison-v1-full/telea_oracle_mask'
  'Temporal=prism-kubectl-baseline-comparison-v1-full/temporal_visible_oracle_masks'
  'SAM2Matting=prism-kubectl-baseline-comparison-v1-full/sam2matting_baseplus'
  'ObjectClear=prism-kubectl-objectclear-comparison-v1-full/objectclear'
  'MatAnyone=prism-kubectl-matanyone-comparison-v1-full/matanyone'
  'ProPainter=prism-kubectl-propainter-comparison-v1-full/propainter'
  'MatAnyone2=prism-kubectl-matanyone2-comparison-v1-full/matanyone2'
  'MAM2=prism-kubectl-mam2-comparison-v1-full/mam2'
  'ROSE=prism-kubectl-rose-comparison-v1-full/rose'
  'PRISM_v26=prism-kubectl-64-fixed-mam2-alpha-v26-v1/prism_v26'
  'PRISM_joint_v15=prism-kubectl-joint-v15-comparison-v1-full/joint_v15_flow'
)

full_summary_exists() {
  local summary="$1"
  [[ -s "$summary" ]] || return 1
  python - "$summary" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get("sequences") == 960 else 1)
PY
}

extend_available() {
  local entry name relative root
  for entry in "${methods[@]}"; do
    name="${entry%%=*}"
    relative="${entry#*=}"
    root="$base/$relative"
    if full_summary_exists "$root/summary.json" && [[ ! -s "$root/summary_extended.json" ]]; then
      python network/scripts/extend_comparison_metrics.py \
        --data-root "$data" \
        --prediction-root "$root/predictions" \
        --output-root "$root"
    fi
  done
}

merge_available() {
  local destination="$1" entry name relative summary
  local -a summaries=()
  for entry in "${methods[@]}"; do
    name="${entry%%=*}"
    relative="${entry#*=}"
    summary="$base/$relative/summary_extended.json"
    if full_summary_exists "$summary"; then
      summaries+=("$name=$summary")
    fi
  done
  if ((${#summaries[@]} == 0)); then
    echo 'No completed 960-sequence evaluations available' >&2
    exit 2
  fi
  python network/scripts/merge_comparison_summaries.py \
    --summary "${summaries[@]}" --output "$destination"
  echo "PRISM_EXTENDED_METRICS_TABLE $destination methods=${#summaries[@]}"
}

extend_available
merge_available "$output/interim_full960.csv"
touch "$output/.interim_complete"

while ! full_summary_exists "$base/prism-kubectl-64-fixed-mam2-alpha-v26-v1/prism_v26/summary.json" \
  || ! full_summary_exists "$base/prism-kubectl-joint-v15-comparison-v1-full/joint_v15_flow/summary.json"; do
  echo 'PRISM_EXTENDED_METRICS_WAITING_FOR_BOTH_PRISM_METHODS'
  sleep 60
done
extend_available
merge_available "$output/final_full960.csv"
touch "$output/.final_complete"
