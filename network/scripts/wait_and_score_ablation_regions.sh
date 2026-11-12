#!/usr/bin/env bash
# One-time, CPU-only analysis after a separately running Stage-4 test finishes.
set -Eeuo pipefail

: "${PRISM_ABLATION_OUTPUT:?set ablation output root}"
: "${PRISM_DATA_ROOT:?set 128px dataset root}"
: "${PRISM_ANALYSIS_PYTHON:?set Python with PRISM dependencies}"
: "${PRISM_ANALYSIS_SCRIPT:?set summarize_ablation_region_metrics.py path}"
: "${PRISM_ANALYSIS_RUN_NAME:?set W&B run name}"
: "${PRISM_CREDENTIAL_FILE:?set private credential file path}"

eval_dir="$PRISM_ABLATION_OUTPUT/results/prism-diffusion-flux-fill"
marker="$eval_dir/.evaluation_complete"
trace_root="$eval_dir/refinement_trace"
analysis_root="$PRISM_ABLATION_OUTPUT/analysis/region-metrics"
mkdir -p "$analysis_root"

# A waiter uses no GPU and exits if the expected evaluation never completes.
for ((attempt=0; attempt<43200; attempt++)); do
  if [[ -f "$marker" ]]; then
    break
  fi
  sleep 60
done
if [[ ! -f "$marker" || ! -d "$trace_root" ]]; then
  echo "PRISM_ABLATION_REGION_FAIL: completed trace is unavailable" >&2
  exit 3
fi
if [[ -f "$analysis_root/.complete" ]]; then
  echo "PRISM_ABLATION_REGION_ALREADY_COMPLETE output=$analysis_root"
  exit 0
fi

while IFS= read -r -d '' credential_name && IFS= read -r -d '' credential_value; do
  export "$credential_name=$credential_value"
done < <("$PRISM_ANALYSIS_PYTHON" - "$PRISM_CREDENTIAL_FILE" <<'PY'
import base64, json, os, stat, sys
path = sys.argv[1]
status = os.stat(path)
if status.st_uid != os.getuid() or stat.S_IMODE(status.st_mode) != 0o600:
    raise SystemExit("credential file must be user-owned mode 600")
value = json.load(open(path, encoding="utf-8"))["WANDB_API_KEY"]
secret = base64.b64decode(value, validate=True)
if not secret or b"\x00" in secret:
    raise SystemExit("invalid W&B credential")
sys.stdout.buffer.write(b"WANDB_API_KEY\x00" + secret + b"\x00")
PY
)
: "${WANDB_API_KEY:?W&B credential unavailable}"
export OPENCV_IO_ENABLE_OPENEXR=1
# The training environment already has refractive_mam2 installed. Putting
# network/src directly on PYTHONPATH would shadow Python's stdlib types.py.
unset PYTHONPATH
export WANDB_DIR="$analysis_root"

"$PRISM_ANALYSIS_PYTHON" "$PRISM_ANALYSIS_SCRIPT" \
  --data-root "$PRISM_DATA_ROOT" \
  --trace-root "$trace_root" \
  --output-root "$analysis_root" \
  --wandb-mode online \
  --wandb-entity "${PRISM_WANDB_ENTITY:-sukhuny-seoul-national-university}" \
  --wandb-project PRISM \
  --wandb-group "${PRISM_WANDB_GROUP:-prism-128-scratch-ablation-20261111}" \
  --wandb-run-name "$PRISM_ANALYSIS_RUN_NAME"
touch "$analysis_root/.complete"
