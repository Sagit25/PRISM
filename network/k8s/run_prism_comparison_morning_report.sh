#!/usr/bin/env bash
set -Eeuo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
test "$(git rev-parse HEAD)" = "${PRISM_SGS_EXPECTED_COMMIT:?}"
base=/data/comparisons/prism-test64-v1
output="$base/morning-20261102"
mkdir -p "$output"
while :; do
  python network/scripts/build_comparison_morning_report.py --root "$base" --output "$output"
  if [[ "$(TZ=Asia/Seoul date +%Y%m%d%H%M)" -ge 202611020900 ]]; then
    touch "$output/.morning_snapshot_complete"
    break
  fi
  sleep 300
done
