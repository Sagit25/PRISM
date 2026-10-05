#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
network_root="$(cd -- "$script_dir/.." && pwd)"
destination="${PRISM_MEMATTE_ROOT:-$network_root/third_party/MEMatte}"
checkpoint="${PRISM_MEMATTE_CHECKPOINT:-$network_root/checkpoints/MEMatte_ViTS_DIM.pth}"
revision="$(tr -d '[:space:]' < "$network_root/third_party/MEMATTE_PINNED_REVISION")"
python_bin="${PYTHON_BIN:-python}"
detectron2_revision="1e3e13bbf607b54f62205c4c33922521822fb298"
checkpoint_id="122p3sdhJVb7vg4IXELeC9C3HEG9Mlh5z"

if [[ ! -d "$destination/.git" ]]; then
  mkdir -p "$(dirname -- "$destination")"
  git clone https://github.com/linyiheng123/MEMatte.git "$destination"
fi

git -C "$destination" fetch --depth 1 origin "$revision"
git -C "$destination" checkout --detach "$revision"

"$python_bin" -m pip install --quiet \
  "git+https://github.com/facebookresearch/detectron2.git@$detectron2_revision" \
  "timm==0.5.4" fairscale easydict scikit-image gdown

mkdir -p "$(dirname -- "$checkpoint")"
if [[ ! -s "$checkpoint" ]]; then
  "$python_bin" -m gdown "$checkpoint_id" --output "$checkpoint"
fi

"$python_bin" - "$destination" "$checkpoint" "$revision" <<'PY'
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1])
checkpoint = Path(sys.argv[2])
expected = sys.argv[3]
actual = subprocess.check_output(
    ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
).strip()
if actual != expected:
    raise SystemExit(f"MEMatte revision mismatch: {actual} != {expected}")
if checkpoint.stat().st_size < 10_000_000:
    raise SystemExit(f"MEMatte checkpoint is unexpectedly small: {checkpoint}")
print(f"MEMATTE_READY revision={actual} checkpoint={checkpoint}")
PY
