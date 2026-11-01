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
run_name="${PRISM_CLASSICAL_RUN_NAME:?}"
limit="${PRISM_CLASSICAL_MAX_SEQUENCES:-16}"
root="$base/$run_name"
case "$limit" in ''|*[!0-9]*) echo "Invalid sequence limit: $limit" >&2; exit 2 ;; esac
test -d "$data/test"
python network/scripts/evaluate_additional_classical_baselines.py \
  --data-root "$data" --output-root "$root" --max-sequences "$limit"

methods=(
  'navier_stokes_oracle_mask|oracle frame-4 object mask'
  'biharmonic_oracle_mask|oracle frame-4 object mask'
  'nearest_visible_oracle_masks|oracle per-frame object masks'
  'grabcut_oracle_box|oracle frame-4 GT bounding box'
)
summaries=()
for entry in "${methods[@]}"; do
  name="${entry%%|*}"
  privilege="${entry#*|}"
  python network/scripts/evaluate_comparison_predictions.py \
    --data-root "$data" --prediction-root "$root/$name/predictions" \
    --output-root "$root/$name" --max-sequences "$limit" --frame-index 4 \
    --model-name "$name" --prompt-policy "$privilege"
  python network/scripts/extend_comparison_metrics.py \
    --data-root "$data" --prediction-root "$root/$name/predictions" \
    --output-root "$root/$name"
  summaries+=("$name=$root/$name/summary_extended.json")
done
python network/scripts/merge_comparison_summaries.py \
  --summary "${summaries[@]}" --output "$root/extended_metrics.csv"
if [[ "$limit" == 0 || "$limit" -ge 960 ]]; then
  previous=(
    "Telea=$base/prism-kubectl-baseline-comparison-v1-full/telea_oracle_mask/summary_extended.json"
    "Temporal=$base/prism-kubectl-baseline-comparison-v1-full/temporal_visible_oracle_masks/summary_extended.json"
    "SAM2Matting=$base/prism-kubectl-baseline-comparison-v1-full/sam2matting_baseplus/summary_extended.json"
    "ObjectClear=$base/prism-kubectl-objectclear-comparison-v1-full/objectclear/summary_extended.json"
    "MatAnyone=$base/prism-kubectl-matanyone-comparison-v1-full/matanyone/summary_extended.json"
    "ProPainter=$base/prism-kubectl-propainter-comparison-v1-full/propainter/summary_extended.json"
  )
  if [[ -s "${previous[0]#*=}" ]]; then
    python network/scripts/merge_comparison_summaries.py \
      --summary "${previous[@]}" "${summaries[@]}" \
      --output "$root/classical_plus_existing_full960.csv"
  fi
fi

if [[ "$limit" == 0 || "$limit" -ge 121 ]]; then
  indices=(0 120)
else
  indices=(0 5)
fi
python network/scripts/make_comparison_contact_sheets.py \
  --data-root "$data" --output-root "$root/matched_images" \
  --indices "${indices[@]}" \
  --prediction \
    "Telea=$base/prism-kubectl-baseline-comparison-v1-full/telea_oracle_mask/predictions" \
    "Temporal=$base/prism-kubectl-baseline-comparison-v1-full/temporal_visible_oracle_masks/predictions" \
    "Navier-Stokes=$root/navier_stokes_oracle_mask/predictions" \
    "Biharmonic=$root/biharmonic_oracle_mask/predictions" \
    "Nearest-visible=$root/nearest_visible_oracle_masks/predictions" \
    "GrabCut=$root/grabcut_oracle_box/predictions"
echo "PRISM_CLASSICAL_ADDITIONAL_EVALUATION_COMPLETE $root sequences=$limit"
