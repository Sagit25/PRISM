#!/usr/bin/env bash
set -Eeuo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
apt-get update -qq
apt-get install -y -qq git libgl1 libglib2.0-0
test "$(git rev-parse HEAD)" = "${PRISM_SGS_EXPECTED_COMMIT:?}"
export OPENCV_IO_ENABLE_OPENEXR=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
python -m pip install --quiet -e './network[data,evaluation]'
# The contact-sheet loader imports the PRISM package, whose semantic modules
# require the pinned SAM2 source even though this CPU job does no inference.
SAM2_BUILD_CUDA=0 PYTHON_BIN=python \
  network/scripts/install_official_sam2.sh \
  "$repo_root/network/third_party/sam2" \
  2b90b9f5ceec907a1c18123530e92e794ad901a4 none
base=/data/comparisons/prism-test64-v1
data=/data/datasets/prism-full-64-materialized-v7r1
v26="$base/prism-kubectl-v26-comparison-full-v2/prism_v26"
joint="$base/prism-kubectl-joint-v15-comparison-v1-full/joint_v15_flow"
final="$base/final-v26-joint-v15-v2"
mkdir -p "$final"
while ! python - "$v26/summary.json" "$joint/summary.json" <<'PY'
import json,sys
from pathlib import Path
for p in map(Path,sys.argv[1:]):
    if not p.is_file() or json.loads(p.read_text()).get('sequences') != 960:
        raise SystemExit(1)
PY
do
  echo 'PRISM_COMPARISON_FINALIZE_WAITING_FOR_BOTH_FULL_EVALUATIONS'
  sleep 60
done
summaries=("PRISM_v26=$v26/summary.json" "PRISM_joint_v15=$joint/summary.json")
predictions=("PRISM v26=$v26/predictions" "Joint v15=$joint/predictions")
for entry in \
  "Telea=$base/prism-kubectl-baseline-comparison-v1-full/telea_oracle_mask" \
  "Temporal=$base/prism-kubectl-baseline-comparison-v1-full/temporal_visible_oracle_masks" \
  "SAM2Matting=$base/prism-kubectl-baseline-comparison-v1-full/sam2matting_baseplus" \
  "ObjectClear=$base/prism-kubectl-objectclear-comparison-v1-full/objectclear" \
  "MatAnyone=$base/prism-kubectl-matanyone-comparison-v1-full/matanyone" \
  "ProPainter=$base/prism-kubectl-propainter-comparison-v1-full/propainter" \
  "Navier-Stokes=$base/prism-kubectl-classical-additional-v1-full/navier_stokes_oracle_mask" \
  "Biharmonic=$base/prism-kubectl-classical-additional-v1-full/biharmonic_oracle_mask" \
  "Nearest-visible=$base/prism-kubectl-classical-additional-v1-full/nearest_visible_oracle_masks" \
  "GrabCut=$base/prism-kubectl-classical-additional-v1-full/grabcut_oracle_box" \
  "MatAnyone2=$base/prism-kubectl-matanyone2-comparison-v2-full/matanyone2" \
  "MAM2=$base/prism-kubectl-mam2-comparison-v2-full/mam2" \
  "ROSE=$base/prism-kubectl-rose-comparison-v2-full/rose"; do
  name="${entry%%=*}"
  path="${entry#*=}"
  if [[ -f "$path/summary.json" ]] && \
      python - "$path/summary.json" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get('sequences') == 960 else 1)
PY
  then
    summaries+=("$name=$path/summary.json")
    predictions+=("$name=$path/predictions")
  fi
done
python network/scripts/merge_comparison_summaries.py \
  --summary "${summaries[@]}" --output "$final/metrics_960.csv"
python network/scripts/make_comparison_contact_sheets.py \
  --data-root "$data" --output-root "$final/matched_images" \
  --frame-index 4 --indices 0 1 2 3 4 5 6 7 120 240 480 720 \
  --prediction "${predictions[@]}"
printf '%s\n' "${summaries[@]}" > "$final/sources.txt"
touch "$final/.comparison_complete"
echo "PRISM_TWO_METHODS_COMPARISON_COMPLETE $final"
