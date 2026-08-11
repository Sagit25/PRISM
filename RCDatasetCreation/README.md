# Planar-background CPU smoke test

This package replaces the earlier environment-map smoke test. It uses an
ordinary 2D image on a finite textured plane behind the transparent object, so
the saved `*_background.exr` is an object-free clean plate in the same fixed
camera view.

## Install into RCDatasetCreation

Unzip this archive at the repository root. It contains:

```text
projects/RefractiveCorresDataset.py
configs/dataset_cpu_smoke_background.yaml
dataset_resources/shape/...
dataset_resources/background/...
```

The old `dataset_resources/env_map/` directory is not read by this version and
may remain in place.

The companion reflection-control changes in
`scene_builder/mitsuba_utils.py` and `scene_builder/elements/shape.py` are still
required when `reflection_scale` is between 0 and 1.

## Run

```bash
OPENCV_IO_ENABLE_OPENEXR=1 \
python render_dataset.py \
  --conf configs/dataset_cpu_smoke_background.yaml \
  --device cpu
```

The config creates one train frame and one test frame at 96 x 96 and 16 spp.
The main sequence-level clean plate is named `*_background.exr`.

## What changed

- Inputs now come from `dataset_resources/background/`, not `env_map/`.
- The full, reflection-off, object-free, and white-transmission passes share
  the same finite XY background plane.
- Refractive flow `Phi` is obtained by intersecting each exit ray with that
  plane and projecting the hit point into the clean-background image.
- A neutral constant environment is retained only as background-independent
  illumination for highlights/reflections.

For a larger dataset, keep cameras above the configured plane. With
`Background.z: -1.25` and an object around the origin, use a spherical camera
range such as `CamPose.theta_range: [20, 70]` rather than sampling below the
plane.
