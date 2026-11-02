#!/usr/bin/env bash
set -Eeuo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
test "$(cat .git/HEAD)" = "${PRISM_SGS_EXPECTED_COMMIT:?}"
base=/data/comparisons/prism-test64-v1
output="$base/morning-20261102"
end_kst="${PRISM_REPORT_END_KST:-202611030900}"
[[ "$end_kst" =~ ^[0-9]{12}$ ]] || { echo "Invalid PRISM_REPORT_END_KST: $end_kst" >&2; exit 2; }
mkdir -p "$output"
while :; do
  python network/scripts/build_comparison_morning_report.py --root "$base" --output "$output"
  if [[ "$(TZ=Asia/Seoul date +%Y%m%d%H%M)" -ge "$end_kst" ]]; then
    touch "$output/.latest_snapshot_complete"
    break
  fi
  sleep 300
done
