#!/usr/bin/env bash
# Serialize independently initialized 128px ablations after GPU1 is free.
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
commit="$(git -C "$repo_root" rev-parse HEAD)"
if [[ -n "${PRISM_EXPECTED_COMMIT:-}" && "$commit" != "$PRISM_EXPECTED_COMMIT" ]]; then
  echo "PRISM_ABLATION_QUEUE_FAIL: code SHA mismatch" >&2
  exit 2
fi
if [[ -n "$(git -C "$repo_root" status --porcelain --untracked-files=no)" ]]; then
  echo "PRISM_ABLATION_QUEUE_FAIL: modified tracked code" >&2
  exit 2
fi

queue_root="${PRISM_ABLATION_QUEUE_ROOT:-/home/sukhun/prism-ablation-scratch-128-20261111}"
mkdir -p "$queue_root"
exec 9>"$queue_root/.queue.lock"
if ! flock -n 9; then
  echo "PRISM_ABLATION_QUEUE_FAIL: another ablation queue is active" >&2
  exit 2
fi
legacy_root="${PRISM_LEGACY_ASSET_ROOT:-/home/sukhun/prism-training-v26-128-20261103}"
export PRISM_TRAIN_PYTHON="$legacy_root/.venv/bin/python"
export PRISM_SAM2_ROOT="$legacy_root/network/third_party/sam2"
export PRISM_SAM2_CHECKPOINT="$legacy_root/network/checkpoints/sam2.1_hiera_large.pt"
export PRISM_MEMATTE_ROOT="$legacy_root/network/third_party/MEMatte"
export PRISM_MEMATTE_CHECKPOINT="$legacy_root/network/checkpoints/MEMatte_AIM500.pth"
if [[ ! -x "$PRISM_TRAIN_PYTHON" ]]; then
  echo "PRISM_ABLATION_QUEUE_FAIL: training environment is missing" >&2
  exit 2
fi

# Every arm starts with new Stage 1A weights. It may use the same public
# SAM2/MEMatte/FLUX priors, but never imports a PRISM donor checkpoint.
default_variants="identity_flow full no_inverse no_flow_supervision no_residual scalar_transmission single_frame"
read -r -a variants <<< "${PRISM_ABLATION_VARIANTS:-$default_variants}"
for variant in "${variants[@]}"; do
  case "$variant" in
    identity_flow|full|no_inverse|no_flow_supervision|no_residual|scalar_transmission|single_frame) ;;
    *) echo "PRISM_ABLATION_QUEUE_FAIL: unknown variant=$variant" >&2; exit 2 ;;
  esac
done
if [[ -n "${PRISM_ABLATION_WAIT_PID:-}" ]]; then
  while :; do
    process_state="$(ps -p "$PRISM_ABLATION_WAIT_PID" -o stat= 2>/dev/null || true)"
    if [[ -z "$process_state" || "$process_state" == Z* ]]; then break; fi
    echo "PRISM_ABLATION_WAIT_FOR_PREVIOUS pid=$PRISM_ABLATION_WAIT_PID state=$process_state"
    sleep 60
  done
  previous_log="${PRISM_ABLATION_PREVIOUS_LOG:?previous log required with wait pid}"
  if ! grep -q 'PRISM_GPU1_TRAIN_COMPLETE' "$previous_log"; then
    echo "PRISM_ABLATION_QUEUE_FAIL: previous variant did not complete" >&2
    exit 1
  fi
  if [[ -n "${PRISM_DATA_CONTRACT_STAMP:-}" ]]; then
    if [[ "$(grep -c '^PASS sequences=' "$previous_log")" -ne 3 ]] \
      || ! grep -q 'PRISM_GPU1_DATA_CONTRACT_OK' "$previous_log"; then
      echo "PRISM_ABLATION_QUEUE_FAIL: previous full data contract is absent" >&2
      exit 1
    fi
    data_root="${PRISM_128_MERGED_ROOT:-/home/sukhun/prism-datasets/128-20261103-merged}"
    printf '%s %s\n' "$data_root" \
      "$(sha256sum "$data_root/dataset_manifest.json" | cut -d' ' -f1)" \
      >"$PRISM_DATA_CONTRACT_STAMP"
    echo "PRISM_ABLATION_CONTRACT_CACHED stamp=$PRISM_DATA_CONTRACT_STAMP"
  fi
fi
for variant in "${variants[@]}"; do
  output_root="$queue_root/$variant"
  if [[ -f "$output_root/queue-complete" ]]; then
    echo "PRISM_ABLATION_ALREADY_COMPLETE variant=$variant"
    continue
  fi
  while :; do
    free_mib="$(nvidia-smi --id=1 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= 87040 )); then
      break
    fi
    echo "PRISM_ABLATION_WAIT variant=$variant gpu1_free_mib=$free_mib"
    sleep 60
  done
  if [[ ! -f "$queue_root/.environment-ready" ]]; then
    "$PRISM_TRAIN_PYTHON" -m pip install --no-deps --no-build-isolation \
      -e "$repo_root/network" >"$queue_root/environment-install.log" 2>&1
    "$PRISM_TRAIN_PYTHON" -c 'import refractive_mam2; print(refractive_mam2.__file__)' \
      | grep -F "$repo_root/network/src/__init__.py" >/dev/null
    touch "$queue_root/.environment-ready"
  fi
  mkdir -p "$output_root"
  export PRISM_ABLATION="$variant"
  export PRISM_OUTPUT_ROOT="$output_root"
  export PRISM_EXPERIMENT_ID="prism-128-scratch-${variant}-20261111"
  export PRISM_WANDB_GROUP="prism-128-scratch-ablation-20261111"
  export PRISM_EXPECTED_COMMIT="$commit"
  export PRISM_STAGE1A_EPOCHS=20
  export PRISM_STAGE1B_EPOCHS=20
  export PRISM_STAGE2_EPOCHS=15
  export PRISM_STAGE3_EPOCHS=15
  export PRISM_STAGE4_EPOCHS=25
  export PRISM_RUN_FINAL_EVALUATION=true
  export PRISM_TEMPORAL_ABLATION_CLIP_LENGTHS=8
  unset PRISM_FIXED_EVAL_FRAME_INDEX PRISM_STAGE1_MANIFESTS PRISM_STAGE2_INITIAL_CHECKPOINT
  unset PRISM_STAGE4_INITIAL_CHECKPOINT PRISM_STAGE2_MAM2_DONOR_CHECKPOINT
  unset PRISM_STAGE3_BACKGROUND_DONOR_CHECKPOINT PRISM_STAGE4_BACKGROUND_DONOR_CHECKPOINT
  unset PRISM_RESTORE_CHECKPOINT_URI
  if [[ "$variant" == "single_frame" ]]; then
    export PRISM_VIDEO_REFINE_CLIP_LENGTH=1
    export PRISM_FINAL_EVAL_CLIP_LENGTH=1
    export PRISM_TEMPORAL_ABLATION_CLIP_LENGTHS=1
    export PRISM_FIXED_EVAL_FRAME_INDEX=4
  else
    export PRISM_VIDEO_REFINE_CLIP_LENGTH=4
    export PRISM_FINAL_EVAL_CLIP_LENGTH=8
  fi
  echo "PRISM_ABLATION_START variant=$variant commit=$commit output=$output_root"
  if bash "$script_dir/run_prism_v26_128_gpu1.sh" >"$output_root/queue.log" 2>&1; then
    touch "$output_root/queue-complete"
    echo "PRISM_ABLATION_COMPLETE variant=$variant output=$output_root"
  else
    echo "PRISM_ABLATION_FAILED variant=$variant output=$output_root" >&2
    exit 1
  fi
done
echo "PRISM_ABLATION_QUEUE_COMPLETE variants=${variants[*]}"
