#!/usr/bin/env bash
# Finish the approved GPUeng8 transfer after both resumable SSH copies land.
set -Eeuo pipefail

home_dir=/home/ruai-u6151
work_root="$home_dir/prism-ablation-scratch-128-20261111"
dataset_root="$home_dir/prism-datasets/128-20261103-merged"
dataset_archive="$work_root/dataset128-gpueng8.tar"
runtime_archive="$work_root/runtime-gpueng8.tar"
expected_dataset_bytes=61978624000
expected_runtime_bytes=8848947200
expected_manifest=4bcaf6af8025337f63b930b08bf1b5147a6337d2cf215816f767abcd93e39a79
expected_commit=564df2de052e655ae033558d7de7b3920f1dbaef

wait_for_size() {
  local archive="$1" expected="$2"
  while [[ ! -f "$archive" || $(stat -c %s "$archive") -lt "$expected" ]]; do
    sleep 60
  done
  [[ $(stat -c %s "$archive") -eq "$expected" ]]
  tar -tf "$archive" >/dev/null
}

wait_for_size "$runtime_archive" "$expected_runtime_bytes"
tar -C "$home_dir" -xf "$runtime_archive"
[[ "$(git -C "$work_root/code" rev-parse HEAD)" == "$expected_commit" ]]
"$home_dir/prism-venv/bin/python" -c 'import torch, wandb; print("PRISM_GPUENG8_RUNTIME_OK", torch.__version__)'
touch "$work_root/.gpueng8-runtime-ready"
echo PRISM_GPUENG8_RUNTIME_READY

wait_for_size "$dataset_archive" "$expected_dataset_bytes"
mkdir -p "$dataset_root"
tar -C "$dataset_root" -xf "$dataset_archive"
[[ -f "$dataset_root/.transfer-complete" ]]
[[ "$(sha256sum "$dataset_root/dataset_manifest.json" | cut -d' ' -f1)" == "$expected_manifest" ]]
touch "$work_root/.gpueng8-data-ready"
echo PRISM_GPUENG8_DATA_READY
