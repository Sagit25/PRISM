#!/usr/bin/env bash
set -Eeuo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
apt-get update -qq
apt-get install -y -qq git libgl1 libglib2.0-0
test "$(git rev-parse HEAD)" = "${PRISM_SGS_EXPECTED_COMMIT:?}"
export OPENCV_IO_ENABLE_OPENEXR=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
python -m pip install --quiet -e './network[data,evaluation]'
base=/data/comparisons/prism-test64-v1
data=/data/datasets/prism-full-64-materialized-v7r1
v25="$base/prism-kubectl-prism-v25-comparison-v1-full/prism_v25"
joint="$base/prism-kubectl-joint-v15-comparison-v1-full/joint_v15_flow"
final="$base/final-two-prism-methods"
mkdir -p "$final"
while ! python - "$v25/summary.json" "$joint/summary.json" <<'PY'
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
summaries=("PRISM_v25=$v25/summary.json" "PRISM_joint_v15=$joint/summary.json")
predictions=("PRISM v25=$v25/predictions" "Joint v15=$joint/predictions")
for entry in \
  "Telea=$base/prism-kubectl-baseline-comparison-v1-full/telea_oracle_mask" \
  "Temporal=$base/prism-kubectl-baseline-comparison-v1-full/temporal_visible_oracle_masks" \
  "SAM2Matting=$base/prism-kubectl-baseline-comparison-v1-full/sam2matting_baseplus" \
  "ObjectClear=$base/prism-kubectl-objectclear-comparison-v1-full/objectclear" \
  "MatAnyone=$base/prism-kubectl-matanyone-comparison-v1-full/matanyone" \
  "ProPainter=$base/prism-kubectl-propainter-comparison-v1-full/propainter" \
  "MatAnyone2=$base/prism-kubectl-matanyone2-comparison-v1-full/matanyone2" \
  "MAM2=$base/prism-kubectl-mam2-comparison-v1-full/mam2"; do
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
