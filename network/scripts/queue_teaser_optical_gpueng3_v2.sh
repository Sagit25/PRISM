#!/usr/bin/env bash
# Run the repaired four-scene optical export after Stage 5's full evaluation.
set -Eeuo pipefail

base=/home/ruai-u6151/prism-inverse-ab
stage5=/home/ruai-u6151/prism-stage5-after-teaser/prism-gpueng3-stage5-moge2-resume-v1
out="$base/output/prism-gpueng3-128-teaser-optical-four-v2"
mkdir -p "$out"
exec 9>"$out/queue.lock"
flock -n 9 || exit 0
exec >>"$out/queue.log" 2>&1
echo "PRISM_TEASER_OPTICAL_V2_QUEUED $(date -Iseconds)"

deadline=$((SECONDS + 345600))
until grep -q 'PRISM_GPUENG3_STAGE5_EVAL_COMPLETE' "$stage5/eval.log"; do
  (( SECONDS < deadline )) || {
    echo 'Stage 5 evaluation did not complete before the teaser deadline' >&2
    exit 3
  }
  sleep 60
done

echo "PRISM_TEASER_OPTICAL_V2_START $(date -Iseconds)"
bash "$base/experiment/run_teaser_optical_gpueng3_v2.sh" \
  >"$out/runner.log" 2>&1
grep -q 'PRISM_TEASER_OPTICAL_EXPORT_COMPLETE sequences=4' "$out/runner.log"
echo "PRISM_TEASER_OPTICAL_V2_COMPLETE $(date -Iseconds)"
