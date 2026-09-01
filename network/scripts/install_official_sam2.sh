#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
network_root="$(cd -- "$script_dir/.." && pwd)"
destination="${1:-$network_root/third_party/sam2}"
sam2_ref="${2:-2b90b9f5ceec907a1c18123530e92e794ad901a4}"
model="${3:-sam2.1_hiera_large}"
python_bin="${PYTHON_BIN:-python}"

case "$model" in
  sam2.1_hiera_tiny|sam2.1_hiera_small|sam2.1_hiera_base_plus|sam2.1_hiera_large)
    ;;
  none)
    ;;
  *)
    echo "Unsupported model: $model" >&2
    exit 2
    ;;
esac

mkdir -p "$(dirname -- "$destination")"
if [[ ! -d "$destination/.git" ]]; then
  if [[ -e "$destination" ]]; then
    echo "Destination exists but is not a Git checkout: $destination" >&2
    exit 2
  fi
  git clone https://github.com/facebookresearch/sam2.git "$destination"
fi

current_ref="$(git -C "$destination" rev-parse HEAD)"
if [[ "$current_ref" != "$sam2_ref" ]]; then
  git -C "$destination" fetch origin "$sam2_ref"
  git -C "$destination" checkout --detach "$sam2_ref"
fi

if [[ "$(uname -s)" == "Darwin" ]]; then
  export SAM2_BUILD_CUDA=0
fi
if ! PRISM_EXPECTED_SAM2_ROOT="$destination" "$python_bin" - <<'PY'
import os
from pathlib import Path

import cv2  # noqa: F401
import refractive_mam2  # noqa: F401
import sam2

expected = Path(os.environ["PRISM_EXPECTED_SAM2_ROOT"]).resolve()
actual = Path(sam2.__file__).resolve().parents[1]
if actual != expected:
    raise SystemExit(f"installed SAM2 resolves to {actual}, expected {expected}")
PY
then
  "$python_bin" -m pip install -e "$destination"
  "$python_bin" -m pip install -e "$network_root[sam2,data,test]"
else
  echo "SAM2 and PRISM Python packages are already installed."
fi

checkpoint_path=""
if [[ "$model" != "none" ]]; then
  checkpoint_dir="$network_root/checkpoints"
  checkpoint_path="$checkpoint_dir/$model.pt"
  checkpoint_url="https://dl.fbaipublicfiles.com/segment_anything_2/092824/$model.pt"
  expected_sha256=""
  if [[ "$model" == "sam2.1_hiera_large" ]]; then
    expected_sha256="2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"
  fi
  mkdir -p "$checkpoint_dir"
  if [[ ! -s "$checkpoint_path" ]]; then
    temporary="$(mktemp "$checkpoint_dir/$model.pt.part.XXXXXX")"
    trap 'rm -f "$temporary"' EXIT
    curl --fail --location --retry 3 --output "$temporary" "$checkpoint_url"
    downloaded_sha256="$(shasum -a 256 "$temporary" | awk '{print $1}')"
    if [[ -n "$expected_sha256" && "$downloaded_sha256" != "$expected_sha256" ]]; then
      echo "Checkpoint SHA-256 mismatch for $model" >&2
      exit 3
    fi
    mv "$temporary" "$checkpoint_path"
    trap - EXIT
  fi
  actual_sha256="$(shasum -a 256 "$checkpoint_path" | awk '{print $1}')"
  if [[ -n "$expected_sha256" && "$actual_sha256" != "$expected_sha256" ]]; then
    echo "Existing checkpoint SHA-256 mismatch for $model: $checkpoint_path" >&2
    exit 3
  fi
  printf '%s  %s\n' "$actual_sha256" "$(basename -- "$checkpoint_path")" \
    > "$checkpoint_path.sha256"
fi

resolved_ref="$(git -C "$destination" rev-parse HEAD)"
echo "Official SAM2 source: $destination ($resolved_ref)"
if [[ -n "$checkpoint_path" ]]; then
  echo "Official SAM2 checkpoint: $checkpoint_path"
fi
echo "PRISM now discovers these local assets automatically."
