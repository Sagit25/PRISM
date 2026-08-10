# RCTrans Dataset Creation

This repository contains the dataset creation code for our SIGGRAPH Asia 2025 paper:

> **RCTrans: Transparent Object Reconstruction in Natural Scene via Refractive Correspondence Estimation**

It focuses on synthetic data generation for transparent-object research, including large-scale correspondence dataset rendering and per-object multi-view reconstruction data rendering.

## Related Repository

This repository only contains the dataset creation part of the paper.

- Full method repository: https://github.com/arkgao/RCTrans.git
- Rendering framework base: https://github.com/arkgao/RenderForge

## What This Repository Provides

- Batch rendering for a large-scale training dataset with refraction correspondence ground truth.
- Multi-view rendering for a single transparent object with RGB images, masks, normals, and correspondence maps.
- Resource preparation scripts for shapes and environment maps.
- A small utility for sampling a balanced validation subset.

## Installation

Create a clean Python environment first:

```bash
conda create -n renderer python=3.10
conda activate renderer
pip install -r requirements.txt
pip install flow_vis
```

Additional dependencies used by the preparation scripts or the optional legacy tracer:

```bash
pip install pyembree
```

Notes:

- `mitsuba==3.7.0` is listed in `requirements.txt`.
- GPU rendering uses `cuda_ad_rgb`; CPU rendering uses `llvm_ad_rgb`.
- The default tracing backend is `mitsuba`. The legacy `trimesh` backend is still available for reproduction.

## Repository Layout

```text
.
├── main.py                          # Single-object multi-view rendering entry point
├── render_dataset.py                # Batch dataset rendering entry point
├── sample_val_subset.py             # Balanced validation subset sampling
├── configs/
│   ├── recon_data.yaml              # Public single-object example config
│   ├── dataset.yaml                 # Public batch-render config template
│   └── templates/                   # Shared config fragments
├── projects/                        # Rendering task implementations
├── scene_builder/                   # Scene construction helpers
├── camera_poses/                    # Camera trajectory / sampling strategies
├── utils/                           # Tracing, logging, camera, config helpers
├── scripts/DatasetCreation/         # Shape and envmap preparation scripts
└── resources/examples/              # Lightweight example assets
```

## Resource Layout

Batch dataset generation expects a resource directory like this:

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

The public template config `configs/dataset.yaml` points to `./dataset_resources`. Update that path if your resources live elsewhere.

## Quick Start

### 1. Single-object multi-view example

This example uses the bundled lightweight assets under `resources/examples`.

```bash
python main.py --conf configs/recon_data.yaml --device gpu
```

The output is written to `./result/recon_example/TransCorresRecon/`.

### 2. Batch dataset rendering

First prepare your resource folder, then run:

```bash
python render_dataset.py --conf configs/dataset.yaml --device gpu
```

The default template writes to `./result/dataset_release_example/` and will automatically bootstrap scene construction from the first available indexed shape and environment map. Users do not need placeholder files such as `holder.ply` or `holder.exr`.

### 3. Sample a smaller validation subset

```bash
python sample_val_subset.py \
  --input /path/to/test_file.txt \
  --output /path/to/val_file.txt \
  --total_num 200
```

This script samples a balanced subset from the full validation split, with equal contributions from BasicShape and OmniObject3D-derived samples.

## Resource Preparation

Detailed resource preparation instructions are available in [scripts/DatasetCreation/README.md](scripts/DatasetCreation/README.md).

Supported sources:

- Shapes: BasicShape and OmniObject3D
- Environment maps: PolyHeaven-like HDR collections and the Laval Indoor HDR Dataset

## Tracing Backend

Tracer selection is controlled in the config:

```yaml
Tracer:
  backend: mitsuba  # or trimesh
```

- `mitsuba` is the default and recommended option.
- `trimesh` is kept for reproducing the older CPU-based pipeline.

The two backends differ in how ray-mesh intersection is performed:

- `trimesh` uses an external mesh intersection path based on `trimesh`, typically accelerated with `pyembree` on CPU. This matches the earlier dataset generation workflow more closely and is kept for reproduction.
- `mitsuba` reuses Mitsuba's built-in scene intersection interface directly inside the rendering stack. It is better integrated with the current pipeline and is typically faster, so it is the default choice in this release.

Both backends trace refraction through the transparent mesh and produce the same type of outputs, namely outgoing directions and background correspondence. The choice mainly affects the implementation path, runtime characteristics, and how closely the behavior matches the older rendering pipeline.

## Notes on Reproduction

The released code filters correspondences that project outside the image plane. But the original paper data accidentally omitted that filtering step. Users working with the released dataset do not have to re-render the data to handle this issue: at load time, they can manually reject correspondences whose coordinates satisfy `correspondence < 0` or `correspondence > 1`, which removes samples projected outside the normalized image plane. If exact reproduction of the original rendered data matters more than the corrected behavior, remove the `inside_mask` filtering in the dataset generation path.

The refractive residual inside the object region is also not expected to be exactly zero. This is mainly due to approximation sources in the current rendering and correspondence pipeline, including pixel-center handling, finite sampling, and the fact that the correspondence-based formulation does not explicitly model all reflection effects.

## Acknowledgements

This codebase is built on Mitsuba 3 and our RenderForge framework.

## Citation

If this repository helps your research, please cite:

```bibtex
@inproceedings{gao2025rctrans,
  title={RCTrans: Transparent Object Reconstruction in Natural Scene via Refractive Correspondence Estimation},
  author={Gao, Ark},
  booktitle={SIGGRAPH Asia},
  year={2025}
}
```
