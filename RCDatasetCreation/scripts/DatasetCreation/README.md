# Dataset Resource Preparation

This folder contains scripts for preparing the shape and environment-map resources used by the rendering pipeline.

The final prepared resources are expected to be organized as:

```text
dataset_resources/
├── shape/
│   ├── train_shape.txt
│   ├── test_shape.txt
│   └── ...
└── env_map/
    ├── train_env.txt
    ├── test_env.txt
    └── ...
```

## Environment Maps

Two sources are supported:

- PolyHeaven or any custom HDR / EXR collection
- Laval Indoor HDR Dataset

### PolyHeaven or custom HDR collections

```bash
python scripts/DatasetCreation/EnvmapPrepare/prepare_polyheaven.py \
  --input_folders /path/to/polyheaven_1k /path/to/custom_envmaps \
  --output_folder /path/to/dataset_resources/env_map
```

This script:

- collects `.hdr` and `.exr` files,
- performs a random train/test split,
- copies them into the output folder,
- writes `train_env.txt` and `test_env.txt`.

### Laval Indoor HDR Dataset

```bash
python scripts/DatasetCreation/EnvmapPrepare/prepare_laval.py \
  --input_folder /path/to/LavalDataset/IndoorHDRDataset2018 \
  --output_folder /path/to/dataset_resources/env_map
```

This script additionally applies exposure correction and downsampling before saving.

## Shapes

Two sources are supported:

- BasicShape
- OmniObject3D

### BasicShape

```bash
python scripts/DatasetCreation/ShapePrepare/prepare_basic_shape.py \
  --input_folder /path/to/RawBasicShape \
  --output_folder /path/to/dataset_resources/shape
```

This script copies meshes, renames them into a unified layout, and writes the corresponding train/test index files.

### OmniObject3D

The OmniObject3D pipeline uses three explicit stages.

#### Step 1: select and preprocess

```bash
python scripts/DatasetCreation/ShapePrepare/prepare_omniverse.py select \
  --raw_folder /path/to/OmniObject3D \
  --output_folder /path/to/omniverse_selected \
  --coarse_list scripts/DatasetCreation/ShapePrepare/coarse_selected_omniverse.txt \
  --fine_list scripts/DatasetCreation/ShapePrepare/fine_selected_omniverse.txt
```

#### Step 2: further simplify

```bash
python scripts/DatasetCreation/ShapePrepare/prepare_omniverse.py simplify \
  --input_folder /path/to/omniverse_selected \
  --output_folder /path/to/omniverse_simplified
```

#### Step 3: split into final dataset resources

```bash
python scripts/DatasetCreation/ShapePrepare/prepare_omniverse.py split \
  --input_folder /path/to/omniverse_simplified \
  --output_folder /path/to/dataset_resources/shape
```

## OmniObject3D Selection Lists

Selection uses two text files with OR logic:

- `coarse_selected_omniverse.txt`: select all objects in a category
- `fine_selected_omniverse.txt`: select specific object instances

The lists are complementary and intentionally non-overlapping.

## Watertightness Note

The mesh repair pipeline is best-effort only. It improves mesh quality but does not guarantee watertight meshes.

If strictly watertight meshes are required, an additional post-processing step with ManifoldPlus is recommended:

```bash
./ManifoldPlus --input input.obj --output output.obj --depth 8
```
