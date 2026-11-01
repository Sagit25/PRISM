#!/usr/bin/env bash
set -Eeuo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
test "$(git rev-parse HEAD)" = "${PRISM_SGS_EXPECTED_COMMIT:?}"
export PRISM_BASELINE_EXPECTED_COMMIT="$PRISM_SGS_EXPECTED_COMMIT"
base=/data/comparisons/prism-test64-v1
report="$base/morning-20261102"
mkdir -p "$report/logs"
status="$report/batch_status.tsv"
printf 'method\tstate\tsummary\n' > "$status"
failed=0

run_one() {
  local name="$1" script="$2" limit="$3" output="$4" expected="$5"
  echo "PRISM_COMPARISON_BATCH_START method=$name limit=$limit" | tee -a "$report/logs/$name.log"
  if PRISM_BASELINE_RUN_NAME="$name" PRISM_BASELINE_MAX_SEQUENCES="$limit" \
      bash "$repo_root/$script" 2>&1 | tee -a "$report/logs/$name.log"; then
    if python - "$output/summary.json" "$expected" <<'PY'
import json,sys
from pathlib import Path
path=Path(sys.argv[1])
assert path.is_file() and json.loads(path.read_text()).get('sequences') == int(sys.argv[2])
PY
    then
      printf '%s\tcomplete\t%s\n' "$name" "$output/summary.json" >> "$status"
      echo "PRISM_COMPARISON_BATCH_COMPLETE method=$name"
      return 0
    fi
  fi
  printf '%s\tfailed\t%s\n' "$name" "$output/summary.json" >> "$status"
  echo "PRISM_COMPARISON_BATCH_FAILED method=$name" >&2
  failed=1
  return 1
}

# The earlier v26 Job completed training. The donor setting in the manifest
# makes this an evaluation-only pass over all 960 held-out sequences.
if bash network/k8s/run_prism_v26_train_compare_v1.sh 2>&1 | \
    tee -a "$report/logs/prism_v26.log"; then
  if python - "$base/$PRISM_SGS_RUN_NAME/prism_v26/summary.json" <<'PY'
import json,sys
from pathlib import Path
path=Path(sys.argv[1])
assert path.is_file() and json.loads(path.read_text()).get('sequences') == 960
PY
  then
    printf '%s\tcomplete\t%s\n' PRISM_v26 "$base/$PRISM_SGS_RUN_NAME/prism_v26/summary.json" >> "$status"
  else
    printf '%s\tfailed\t%s\n' PRISM_v26 "$base/$PRISM_SGS_RUN_NAME/prism_v26/summary.json" >> "$status"
    failed=1
  fi
else
  printf '%s\tfailed\t%s\n' PRISM_v26 "$base/$PRISM_SGS_RUN_NAME/prism_v26/summary.json" >> "$status"
  failed=1
fi

run_one prism-kubectl-matanyone2-comparison-v2-full \
  network/k8s/run_prism_matanyone2_comparison_v1.sh 0 \
  "$base/prism-kubectl-matanyone2-comparison-v2-full/matanyone2" 960 || true
run_one prism-kubectl-mam2-comparison-v2-full \
  network/k8s/run_prism_mam2_comparison_v1.sh 0 \
  "$base/prism-kubectl-mam2-comparison-v2-full/mam2" 960 || true
if run_one prism-kubectl-rose-comparison-v2-smoke \
  network/k8s/run_prism_rose_comparison_v1.sh 1 \
  "$base/prism-kubectl-rose-comparison-v2-smoke/rose" 1; then
  run_one prism-kubectl-rose-comparison-v2-full \
    network/k8s/run_prism_rose_comparison_v1.sh 0 \
    "$base/prism-kubectl-rose-comparison-v2-full/rose" 960 || true
else
  printf '%s\tskipped_after_smoke_failure\t%s\n' ROSE_full \
    "$base/prism-kubectl-rose-comparison-v2-full/rose/summary.json" >> "$status"
fi
echo "PRISM_COMPARISON_BATCH_FINISHED failed=$failed status=$status"
exit "$failed"
