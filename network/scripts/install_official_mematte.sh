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

# Detectron2's setup imports torch while resolving its build requirements.
# Disabling build isolation keeps the already validated PRISM torch/CUDA build
# visible instead of creating a temporary environment without torch.
"$python_bin" -m pip install --quiet --no-build-isolation \
  "git+https://github.com/facebookresearch/detectron2.git@$detectron2_revision"
"$python_bin" -m pip install --quiet \
  "timm==0.5.4" fairscale easydict scikit-image gdown

mkdir -p "$(dirname -- "$checkpoint")"
if [[ -n "${PRISM_MEMATTE_CHECKPOINT_VOLUME:-}" ]]; then
  "$python_bin" - "$checkpoint" <<'PY'
import hashlib
import os
from pathlib import Path
import re
import sys

from vessl.storage.volume_v2 import _get_volume_with_federate

destination = Path(sys.argv[1])
storage = os.environ.get("PRISM_MEMATTE_CHECKPOINT_STORAGE", "vessl-storage")
volume_name = os.environ["PRISM_MEMATTE_CHECKPOINT_VOLUME"]
relative_key = os.environ["PRISM_MEMATTE_CHECKPOINT_KEY"].strip("/")
expected_sha = os.environ["PRISM_MEMATTE_CHECKPOINT_SHA256"].lower()
if not relative_key or ".." in Path(relative_key).parts:
    raise SystemExit("Invalid MEMatte checkpoint key")
if re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None:
    raise SystemExit("PRISM_MEMATTE_CHECKPOINT_SHA256 must be a full SHA-256")

def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

if not destination.is_file() or checksum(destination) != expected_sha:
    volume = _get_volume_with_federate(storage, volume_name, federation_type="read")
    if not hasattr(volume, "s3_client"):
        raise SystemExit("MEMatte checkpoint volume is not S3-backed")
    key = "/".join(part for part in (volume.base_path.strip("/"), relative_key) if part)
    partial = destination.with_name(destination.name + ".part")
    try:
        volume.s3_client.download_file(volume.bucket_name, key, str(partial))
        actual_sha = checksum(partial)
        if actual_sha != expected_sha:
            raise SystemExit(f"MEMatte checkpoint SHA-256 mismatch: {actual_sha}")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
print(f"MEMATTE_CHECKPOINT_SHA256_OK {expected_sha}")
PY
elif [[ ! -s "$checkpoint" ]]; then
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
