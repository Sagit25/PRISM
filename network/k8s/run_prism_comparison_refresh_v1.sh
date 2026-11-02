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
output="$base/final-20261103"
report="$base/morning-20261103"
end_kst="${PRISM_REPORT_END_KST:-202611030900}"
[[ "$end_kst" =~ ^[0-9]{12}$ ]] || { echo "Invalid report deadline: $end_kst" >&2; exit 2; }
mkdir -p "$output" "$report"
while :; do
  python network/scripts/refresh_comparison_results.py \
    --root "$base" --data-root "$data" --output "$output"
  python network/scripts/build_comparison_morning_report.py \
    --root "$base" --output "$report"
  completed="$(awk -F '\t' 'NR>1 && ($2=="extended" || $2=="base_only") {n++} END {print n+0}' "$output/status.tsv")"
  if [[ "$completed" == 15 || "$(TZ=Asia/Seoul date +%Y%m%d%H%M)" -ge "$end_kst" ]]; then
    touch "$output/.snapshot_complete" "$report/.snapshot_complete"
    echo "PRISM_COMPARISON_MORNING_SNAPSHOT methods=$completed/15 output=$output"
    break
  fi
  sleep 300
done
