#!/usr/bin/env bash
set -euo pipefail

destination="${1:-third_party/sam2}"
sam2_ref="${2:-}"
if [[ -e "$destination" ]]; then
  echo "Destination already exists: $destination" >&2
  exit 2
fi

git clone https://github.com/facebookresearch/sam2.git "$destination"
if [[ -n "$sam2_ref" ]]; then
  git -C "$destination" checkout --detach "$sam2_ref"
fi
python -m pip install -e "$destination"
python -m pip install -e ".[sam2,test]"

resolved_ref="$(git -C "$destination" rev-parse HEAD)"
echo "Installed official SAM2 at $resolved_ref and refractive-mam2. Download a SAM2.1 checkpoint from Meta before inference."
