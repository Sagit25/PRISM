# v26 checkpoint → depth/normal post head on GPU1

The v26 128×128 training remains untouched. This extension starts **only after**
`/home/sukhun/prism-training/v26-128-full/checkpoints/stage4/.training_complete`
and its nonempty `prism_stage4_best.pt` both exist. It loads that checkpoint as
the donor, initializes a separate geometry head, freezes every preexisting
PRISM parameter, and trains only the new head. Its output is stored in
`/home/sukhun/prism-training/v26-128-geometry-posthead`; it does not overwrite
v26 checkpoints or datasets.

The head receives the final jointly optimized foreground, alpha, color
transmission, refractive flow, residual, background, and reconstructed frame,
plus the upstream MAM2/SAM masks and trimap. It predicts the **first visible
surface** world-space normal and camera-distance depth within the object.
Supervision uses the existing `_N_object.npy`, `_D_object.npy`, and validity
maps. The separate-head result does not retroactively change v26's F/B/flow.

After v26 Stage 4 has completed and the new checkout is on GPU1, run:

```bash
PRISM_V26_REPO_ROOT=/home/sukhun/prism-training-v26-128-20261103 \
PRISM_TRAIN_PYTHON=/home/sukhun/prism-training-v26-128-20261103/.venv/bin/python \
bash network/scripts/run_prism_v26_128_geometry_posthead_gpu1.sh
```

Override `PRISM_GEOMETRY_DONOR_ROOT`, `PRISM_GEOMETRY_DONOR_CHECKPOINT`,
`PRISM_DATA_ROOT`, or `PRISM_OUTPUT_ROOT` only if the corresponding GPU1 paths
actually differ. The launcher checks the completed donor marker, checkpoint,
dataset manifest, Python environment, and existing HF/W&B credentials before
starting. It uses GPU1 (`CUDA_VISIBLE_DEVICES=1`) and runs training, validation,
and a final test with W&B online logging. `PRISM_GEOMETRY_EPOCHS` defaults to 12;
`PRISM_GEOMETRY_LR` defaults to `1e-4`.

The resulting checkpoint is a full model checkpoint containing the frozen v26
base plus the new geometry head. The original v26 checkpoint remains the
reference for the original method. Compare object-region depth RMSE/AbsRel and
normal mean angular error with the same test split, alongside v26's original
F/B/render metrics. Do not interpret a geometry-only gain as improved
foreground or background reconstruction.
