# Refractive MAM2 on official SAM2

This repository upgrades Meta's official SAM2.1 video predictor with a
paper-faithful reproduction of the two MAM2 components that are essential for
video matting, then attaches the counterfactual-background physical matting
model:

```text
official SAM2.1 image encoder + prompt/mask decoder + mask memory
  -> PDD pass 1 on memory-conditioned features: refined binary mask
  -> MSS / shared PDD pass 2 on clean non-memory features: 3-class trimap
  -> one sequence-level B_cf from directly exposed fixed-camera pixels
  -> reusable operator: alpha, RGB tau, G, refractive flow u, residual R
  -> inverse splat: (I-G-R)/tau at Phi(x)=x+u into the shared B_cf canvas
  -> repeat joint operator/background refinement
  -> I_hat = G + tau * sample(B_cf, x+u) + R
```

The official SAM2 files are not copied or modified. Hydra instantiates a real
`MAM2VideoPredictor(SAM2VideoPredictor)` subclass, the original checkpoint is
loaded with an explicit allow-list for only the new `mam2_mss.*` keys, and LoRA
is injected afterwards. No dynamic `__class__` replacement is used.

MAM2's paper and supplement are public, but its authors' implementation was not
public at the time this repository was written. PDD/MSS here are therefore a
clean-room reproduction of the published design, not official MAM2 code.

## Implemented modules

| Module | Implementation |
| --- | --- |
| Official SAM2.1 | Official Hydra config/modules/checkpoint, instantiated as a strict subclass |
| PDD | sparse-prompt cross attention, dense-prompt fusion, SAM mask residual, high-resolution trimap refinement |
| MSS | memory pass for mask, refined-mask pseudo-prompt, same PDD weights on clean feature for trimap |
| LoRA | post-checkpoint injection into Hiera attention `qkv` and `proj` linears |
| Background | one `[B,3,H,W]` reusable asset; robust direct observations, inverse-refracted evidence, completion only in true holes |
| Matter | predicts alpha, premultiplied G, RGB color transmission, refractive flow, bounded residual and confidence |
| Inverse solver | differentiable bilinear forward splatting of transparent-interior background observations |
| Renderer | `G + tau * sample(B_cf, x+u) + R`, with `tau=(1-alpha)*color_transmission` |
| RCTrans data | v15 sequence loader, linear RGB/BGR conversion, full GT mapping, contract checks and paired-background sampler |
| Training | direct alpha/G/C/tau/Phi/u/R/confidence supervision with validity masks, selective semantic stages and paired operator invariance |
| Inference | official first-frame prompt API and full-video propagation runner |
| Checkpoints | compact extension + physics checkpoint save/load |

## Installation

SAM2.1 requires Python 3.10+, PyTorch 2.5.1+ and a compatible CUDA toolchain.

```bash
git clone https://github.com/facebookresearch/sam2.git third_party/sam2
python -m pip install -e third_party/sam2
python -m pip install -e ".[sam2,data,test]"
```

The same commands are available in `scripts/install_official_sam2.sh`. Download
one of Meta's SAM2.1 checkpoints separately; model weights are intentionally not
included.

For a reproducible environment, pass a tested SAM2 commit/tag as the second
installer argument, for example `scripts/install_official_sam2.sh path REF`.

## End-to-end inference

Put JPEG frames in a directory using lexically sortable names such as
`00000.jpg`, `00001.jpg`, then run:

```bash
python examples/run_sam2_refractive.py \
  --video data/clip \
  --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --sam2-checkpoint checkpoints/sam2.1_hiera_large.pt \
  --checkpoint checkpoints/refractive_mam2.pt \
  --point 640 360 \
  --output outputs/clip.pt
```

The output file contains `mask_logits`, `trimap_logits`, `alpha`,
`straight_foreground`, `premultiplied_foreground`, `color_transmission`,
`transmittance`, `refractive_flow`, `residual`, the single
`counterfactual_background` asset, direct/inverse coverage, and `reconstruction`.

Programmatic use:

```python
import numpy as np
from refractive_mam2 import (
    SAM2RefractiveRunner,
    build_mam2_video_predictor,
    build_physics_pipeline_for_sam2,
)

predictor = build_mam2_video_predictor(
    "configs/sam2.1/sam2.1_hiera_l.yaml",
    "checkpoints/sam2.1_hiera_large.pt",
)
physics = build_physics_pipeline_for_sam2().cuda().eval()
state = predictor.init_state(video_path="data/clip")
predictor.add_new_points_or_box(
    inference_state=state,
    frame_idx=0,
    obj_id=1,
    points=np.array([[640, 360]], np.float32),
    labels=np.array([1], np.int32),
)
output = SAM2RefractiveRunner(predictor, physics).run(
    frames_rgb_0_to_1,  # [T,3,H,W]
    state,
    object_id=1,
)
```

PDD, trimap, completion and physical heads are newly initialized unless a
trained extension checkpoint is loaded. Shape-correct output from untrained
weights is not a meaningful matte.

The end-to-end runner currently decomposes one selected object per call. SAM2
can still track several objects; call the physical runner once for each object
when separate mattes are required.

## Joint image formation and inversion

The network predicts the premultiplied color directly and exposes straight
foreground only as a guarded derived diagnostic:

```text
G = alpha * F_std
tau = (1-alpha) * color_transmission
B_refracted(x) = sample(B_cf, x + u(x))
I_hat = G + tau * B_refracted + R
B_observation(Phi(x)) = (I(x) - G(x) - R(x)) / tau(x)
```

`F_std` is the foreground before background mixing. Predicting `G` prevents
division instability when alpha is near zero. RGB `tau` makes the same operator
reusable on new backgrounds for colored transparent materials. `R` is bounded
and sparsity-regularized; it must not absorb the full image. Structured HDR
reflection remains outside the main reflection-free scope.

The fixed-camera sequence owns exactly one background canvas. Directly exposed
pixels are preserved. For pixels never exposed, transparent-interior estimates
are bilinearly splatted through `Phi`; the completion network is used only when
both sources are absent. `PipelineConfig.joint_refinement_steps` controls the
unrolled fixed-point iterations. No detach is used in the final joint stage.

## Training

Stage 1 reproduces MAM2 selective supervision:

```python
from refractive_mam2.training import (
    SemanticTargets,
    configure_stage1,
    normalize_sam2_training_frames,
    selective_semantic_loss,
)

optimizer = torch.optim.AdamW(configure_stage1(predictor), lr=1e-4)
semantic = predictor.forward_mam2_clip(
    normalize_sam2_training_frames(resized_frames_rgb_0_to_1),
    first_frame_point_inputs={
        "point_coords": point_coords,
        "point_labels": point_labels,
    },
)
losses = selective_semantic_loss(
    semantic.mask_logits,
    semantic.trimap_logits,
    SemanticTargets(object_mask=mask_gt, trimap=trimap_gt),
    dataset_kind="synthetic_physics",
)
```

- VOS samples supervise mask.
- video/image matting samples supervise trimap.
- exact synthetic-physics samples may supervise both.
- the original SAM2 weights stay frozen; PDD/MSS and Hiera LoRA train.

Stage 2 freezes semantic tracking and trains the background/matter heads:

```python
from refractive_mam2.training import configure_stage2, physics_stage_loss

optimizer = torch.optim.AdamW(
    configure_stage2(predictor, physics),
    lr=2e-4,
)
losses = physics_stage_loss(prediction, ground_truth)
```

Recommended schedule:

1. Train physics matter with ground-truth counterfactual backgrounds.
2. Train the one-canvas direct background reconstruction and completion.
3. Mix ground-truth and predicted backgrounds with decaying teacher forcing.
4. Train PDD/MSS and encoder LoRA using selective supervision.
5. Jointly fine-tune PDD/MSS, inverse splatting, background and operator with
   `configure_joint` and `joint_stage_loss`.
6. Render each object trajectory on paired backgrounds and apply the
   cross-background operator-reuse loss.

Synthetic clips should store observed frames, geometric object mask, transparent
trimap, alpha, straight/premultiplied foreground, RGB transmittance, refractive
flow, residual (zero in the exactly modeled main split), and one object-free
counterfactual background. Paired backgrounds for identical pose trajectories
are required to supervise reusability.

## Scope: fixed camera

The joint global-canvas implementation intentionally accepts only a fixed
camera and fixed background. There is no RAFT/local-flow correspondence path:
transparent distortions must not be mistaken for camera/background motion.
Small camera jitter should be stabilized before inference. Supporting a
genuinely moving camera requires a different world-coordinate background
representation.

## RCTrans v15 dataset integration

The loader consumes the output of the modified `RCDatasetCreation` generator
without renaming fields. OpenCV reads EXR as BGR, so the loader explicitly
converts it back to **linear RGB**. It also builds the transparent-object
trimap and checks `Phi=x+u`, `tau=(1-alpha)C`, confidence range and the full
image-formation equation before returning a sample.

| Generator output | Network target |
| --- | --- |
| `*_I.exr` | `frames` |
| `*_object_mask.png` | `object_mask` and generated trimap |
| `*_alpha.npy` | `alpha` |
| `*_CF.exr` | premultiplied foreground `G` |
| `*_F.exr` | diagnostic straight foreground `F_std` |
| `*_T.exr` | RGB color transmission `C` |
| `*_A.exr` | RGB transmittance `tau` |
| `*_Phi.npy` | absolute source coordinate `Phi` |
| `*_u.npy` | pixel displacement `u=Phi-x` |
| `*_R.exr` | signed residual `R` |
| `*_confidence.npy` | correspondence/model-fit confidence |
| `*_phi_valid.png` | validity mask for `C`, `tau`, `Phi`, `u` and confidence losses |
| `*_background.exr` | one sequence-level `B_cf` |
| `*_sequence_meta.json` | pair group, split, color space and generator version |

Minimal contract/loading check:

```bash
OPENCV_IO_ENABLE_OPENEXR=1 python examples/check_rctrans_dataset.py \
  --root /path/to/generated/train \
  --clip-length 4 \
  --check-pairs
```

Programmatic paired loading:

```python
from refractive_mam2 import RCTransPRISMDataset, build_paired_prism_dataloader

dataset = RCTransPRISMDataset(
    "/path/to/generated/train",
    clip_length=4,
    strict_contract=True,
    require_split_kind="main",
)
loader = build_paired_prism_dataloader(
    dataset,
    backgrounds_per_group=2,
    num_workers=4,
)

for batch in loader:
    batch = batch.to("cuda", non_blocking=True)
    target = batch.ground_truth
    prediction = physics(
        target.frames,
        counterfactual_background_gt=target.counterfactual_background,
        use_ground_truth_background=teacher_forcing,
    )
    losses = joint_stage_loss(
        prediction,
        target,
        paired_background_group_ids=batch.paired_background_group_ids,
        operator_support=target.object_mask * target.refractive_validity,
    )
    losses["total"].backward()
```

The pair sampler only groups sequences that share
`paired_background_group_id` while having distinct `background_path` values.
This makes the reuse loss compare the same camera, material and object pose on
different backgrounds rather than accidentally comparing unrelated clips.

## Resolution and refractive-flow range

The default matter head now predicts a normalized displacement and converts it
to RCTrans pixel units using 25% of each image dimension. Consequently, the
approximate per-axis limits scale automatically:

| Resolution | Default maximum displacement |
| --- | --- |
| 256 px | 64 px |
| 512 px | 128 px |
| 1024 px | 256 px |

Use `MatterConfig(flow_parameterization="fixed_pixels",
max_refractive_flow=64)` only for old checkpoints that were trained with the
v0.4 fixed-pixel convention. The renderer now uses zero out-of-bounds padding,
matching RCTrans `cv2.remap(..., BORDER_CONSTANT, 0)`.

## Validation

Static checks and unit tests:

```bash
python -m compileall -q refractive_mam2 tests examples
python -m pytest
```

The complete data path additionally requires the data extra and a rendered
RCTrans v15 directory:

```bash
python -m pip install -e ".[data,test]"
OPENCV_IO_ENABLE_OPENEXR=1 python examples/check_rctrans_dataset.py \
  --root /path/to/generated/train --check-pairs
```

See `docs/MAM2_INTEGRATION.md` for exact official-SAM2 hook locations and tensor
contracts, and `docs/JOINT_DECOMPOSITION.md` for the shared-background inverse
solver, operator definitions, paired-background supervision, and the explicit
3-D geometry boundary.

## References

- [Official SAM2 repository](https://github.com/facebookresearch/sam2)
- [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714)
- [Matting Anything 2: Towards Video Matting for Anything](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d077bc9ea82a2998ca6b2d0158b5ac6e-Abstract-Conference.html)
