#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${PRISM_128_WORKSPACE:-/home/sukhun/prism-generation-128-20261103}"
dataset_root="${PRISM_128_OUTPUT_ROOT:-/home/sukhun/prism-datasets/128-20261103}"
python="$workspace/.venv/bin/python"
cd "$workspace/RCDatasetCreation"
export CUDA_VISIBLE_DEVICES=1
export OPENCV_IO_ENABLE_OPENEXR=1
export PYTHONUNBUFFERED=1
export PRISM_128_OUTPUT_ROOT="$dataset_root"

test -x "$python"
test -d dataset_resources_research/background
test -d dataset_resources_research/shape
mkdir -p "$dataset_root"
echo "PRISM_128_START gpu_visible=$CUDA_VISIBLE_DEVICES output=$dataset_root"
sha256sum configs/dataset_prism_main_128.yaml asset_manifests/prism_research_assets_v1.json
"$python" tools/prepare_prism_assets.py validate
"$python" - <<'PY'
import mitsuba as mi
import cv2
import numpy as np
from pathlib import Path
import tempfile
mi.set_variant('cuda_ad_rgb')
print('PRISM_128_CUDA_OK', mi.__version__, mi.variant(), flush=True)
with tempfile.TemporaryDirectory() as root:
    exr = Path(root) / 'write_test.exr'
    assert cv2.imwrite(str(exr), np.zeros((2, 2, 3), dtype=np.float32))
    assert cv2.imread(str(exr), cv2.IMREAD_UNCHANGED).shape == (2, 2, 3)
print('PRISM_128_EXR_OK', cv2.__version__, flush=True)
PY

"$python" - <<'PY'
from pathlib import Path
import yaml

source = Path('configs/dataset_prism_main_128.yaml')
target = Path('configs/dataset_prism_main_128_smoke_local.yaml')
config = yaml.safe_load(source.read_text())
config['project_name'] = 'prism_main_128_smoke_20261103'
config['backgrounds_per_shape'] = 2
config['sequence_num_per_background'] = 1
config['frames_per_sequence'] = 2
config['ValidationSplit']['fraction'] = 0.25
config['Camera']['sampler']['sample_count'] = 16
config['ResourceSubset'] = {
    'enabled': True,
    'seed': 20261103,
    'train_shapes': 4,
    'test_shapes': 1,
    'train_backgrounds': 4,
    'test_backgrounds': 1,
}
target.write_text(yaml.safe_dump(config, sort_keys=False))
print('PRISM_128_SMOKE_CONFIG', target, flush=True)
PY

smoke_name=prism_main_128_smoke_20261103
"$python" render_dataset.py \
  --conf configs/dataset_prism_main_128_smoke_local.yaml \
  --device gpu --output-folder "$dataset_root" \
  --project_name "$smoke_name" --splits train validation test
for split in train validation test; do
  "$python" tools/validate_prism_contract.py "$dataset_root/$smoke_name/$split"
done
"$python" tools/freeze_prism_manifest.py "$dataset_root/$smoke_name"
"$python" tools/freeze_prism_manifest.py "$dataset_root/$smoke_name" --verify
touch "$dataset_root/$smoke_name/.generation_complete"
echo PRISM_128_SMOKE_COMPLETE

config=configs/dataset_prism_main_128.yaml
for index in 0 1 2 3 4; do
  printf -v shard '%02d' "$index"
  name="prism_main_128_20261103_train_s${shard}of05"
  if [[ -f "$dataset_root/$name/.generation_complete" ]]; then
    echo "PRISM_128_SKIP_COMPLETE $name"
    continue
  fi
  echo "PRISM_128_GENERATE_START $name"
  "$python" render_dataset.py --conf "$config" --device gpu \
    --output-folder "$dataset_root" --project_name "$name" \
    --splits train --shard-count 5 --shard-index "$index"
  "$python" tools/validate_prism_contract.py "$dataset_root/$name/train"
  "$python" tools/freeze_prism_manifest.py "$dataset_root/$name"
  "$python" tools/freeze_prism_manifest.py "$dataset_root/$name" --verify
  touch "$dataset_root/$name/.generation_complete"
  echo "PRISM_128_GENERATE_COMPLETE $name"
done

for split in validation test; do
  name="prism_main_128_20261103_$split"
  if [[ -f "$dataset_root/$name/.generation_complete" ]]; then
    echo "PRISM_128_SKIP_COMPLETE $name"
    continue
  fi
  echo "PRISM_128_GENERATE_START $name"
  "$python" render_dataset.py --conf "$config" --device gpu \
    --output-folder "$dataset_root" --project_name "$name" \
    --splits "$split"
  "$python" tools/validate_prism_contract.py "$dataset_root/$name/$split"
  "$python" tools/freeze_prism_manifest.py "$dataset_root/$name"
  "$python" tools/freeze_prism_manifest.py "$dataset_root/$name" --verify
  touch "$dataset_root/$name/.generation_complete"
  echo "PRISM_128_GENERATE_COMPLETE $name"
done
echo PRISM_128_ALL_SPLITS_COMPLETE
