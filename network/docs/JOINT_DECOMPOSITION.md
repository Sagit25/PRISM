# Joint reusable background/operator decomposition

## Scope

The implemented inverse problem assumes one fixed camera, one static
background, and one moving transparent object. The sequence owns one reusable
counterfactual background asset `B_cf` with shape `[B,3,H,W]`. Frame-specific
copies are broadcast views only.

## Reusable colored operator

For every frame, `PhysicsAwareMatter` predicts:

- scalar geometric opacity `alpha`;
- premultiplied additive foreground `G`;
- RGB `color_transmission`;
- RGB total transmittance `tau = (1-alpha) * color_transmission`;
- normalized weights `w_k` and target-to-background displacements `u_k` for a
  deformable 3x3 local kernel;
- expected displacement `u=sum_k w_k u_k`, with `Phi(x)=x+u(x)`, retained as
  the RCTrans-supervised compatibility output;
- bounded additive residual `R`;
- inverse-evidence confidence `c`.

The renderer is

```text
I_hat(x) = G(x) + tau(x) * sum_k w_k(x) B_cf(x+u_k(x)) + R(x)
```

Outside the semantic object support, the identity operator is enforced:
`alpha=G=u=R=0` and `tau=1`.

`R` is not a general image-generation escape route. The default head bounds it
with `residual_scale`, and training adds direct residual supervision when
available plus an always-on sparsity penalty. Structured HDR/specular
reflection is still excluded from the main dataset.

## Background evidence

### Direct evidence

Mask and trimap support are dilated, pixels above the configured object
threshold are rejected, and every remaining object-free frame contributes at
the identical fixed-camera coordinate. A robust color residual weight rejects
segmentation leakage and sensor outliers. Directly observed pixels have strict
priority in the final asset.

### Transparent-interior evidence

The current operator turns each transparent-object pixel into a background
observation:

```text
B_observation(Phi(x)) = (I(x) - G(x) - R(x)) / tau(x)
```

Samples with insufficient transmission are rejected. Remaining values are
weighted by support, transmission, confidence and `w_k`, then bilinearly
forward-splatted through every `x+u_k(x)` on the shared canvas. The four
bilinear neighbors and all 3x3 taps are vectorized. The splat is differentiable with respect to
the recovered radiance, transmission, confidence and the fractional flow.

### Fusion priority

```text
direct observation > inverse-refracted observation > learned completion
```

Thus a generative completion result can never overwrite a real, directly
exposed background pixel.

## Joint refinement

The pipeline unrolls `PipelineConfig.joint_refinement_steps` iterations:

1. construct/fuse the current shared background;
2. predict the colored refractive operator conditioned on that background;
3. invert transparent-interior observations through the operator;
4. splat them into the shared canvas and fuse again.

The completion in steps 1/4 is PRISM-FFC by default: a deterministic
64-wide, nine-block LaMa/GLaMa-style encoder-decoder (up to 512 channels) whose bottleneck keeps local and global
feature streams and mixes the global stream in the Fourier domain. Its input
is `[evidence RGB, coverage, true-hole mask]`. It runs once per unrolled
iteration, remains in the autograd graph, and is supervised by spatial
true-hole and frequency-domain background objectives. The completed proposal
is hard-composited only at `true_hole`, preserving all physical evidence
exactly. The earlier dilated CNN is retained solely for a controlled ablation.

After the last fusion, the operator is evaluated once more on the final shared
background. The final render therefore uses a mutually aligned operator and
background instead of adjacent fixed-point iterates. No detach is used in the
default joint configuration, so render and component losses train both sides.
The curriculum assigns distinct responsibilities: Stage 2 trains only PAM with
an oracle background, Stage 3 trains PAM and background recovery with decaying
teacher forcing, and Stage 4 reconnects the MAM2 adapters/alpha decoder for
low-risk end-to-end fine-tuning. The official SAM2 base remains frozen while
its LoRA adapters and the configured MEMatte adaptive-token backbone/decoder
are trainable in the final full-quality preset.

## PRISM-PAM and PRISM-Background responsibilities

PRISM-PAM is the per-frame physics operator. Given the observed frame, MAM2
alpha/trimap/features, and the current counterfactual background, it predicts
the object's standard premultiplied foreground `G`, RGB transmission `C`,
transmittance `tau=(1-alpha)C`, deformable 3x3 background-sampling kernel
`{w_k,u_k}` and its expected displacement `u`, bounded
residual/reflection `R`, confidence, and an UNKNOWN-only bounded alpha
correction. It answers: *how does this transparent object transform whatever
background is placed behind it in this frame?*

PRISM-Background is the sequence-level counterfactual scene solver. It robustly
aggregates directly visible fixed-camera pixels, forward-splats the background
evidence recovered by inverting PAM's refractive operator, fuses both evidence
sources with coverage/confidence, and completes only pixels that remain true
holes. It emits one reusable `[B,3,H,W]` background plus direct/inverse coverage
and uncertainty. It answers: *what static scene would have been visible if the
object had never been present?*

The separation prevents a degenerate solution: PAM cannot store an entire
frame-specific background as “foreground,” and background completion cannot
overwrite directly observed evidence. Their fixed-point loop exchanges only
the current background estimate and physically interpretable operator fields.

## Reusability supervision

`reusable_operator_consistency` and `joint_stage_loss` accept predictions of
the same object trajectory rendered on two different backgrounds. They force
`alpha`, `G`, `C`, `tau`, `{w_k,u_k}`, expected `u`, `R` and confidence to remain invariant while
the background changes. `RCTransPRISMDataset` reads
`paired_background_group_id`, and `PairedBackgroundBatchSampler` rejects groups
that do not contain distinct background assets. This supervision is essential:
the output schema alone cannot prevent the operator from encoding
source-background texture.

## Dataset contract

The canonical RCTrans v15 main split stores:

```text
I, object mask, alpha, F_std, G, RGB C, RGB tau, Phi, u, R,
confidence, phi_valid, B_cf, sequence metadata
```

The loader constructs the transparency-aware trimap. It directly supervises C
and generator confidence, masks correspondence-dependent C/tau/Phi/u losses
with `phi_valid`, and checks `Phi=x+u` and `tau=(1-alpha)C`. `R` is a signed GT
term even in the reflection-free main split because it also records numerical
and boundary mismatch. Add a separate diagnostic split for structured
reflection. For reusability training, render the same camera,
geometry/material and pose trajectory on at least two independent backgrounds.

EXRs are linear RGB on disk. Because OpenCV decodes them as BGR, the dataset
loader reverses the channel order. Both renderer and generator use zero-valued
out-of-bounds samples. Flow is stored and supervised in pixels; the default
network parameterization scales its maximum displacement with resolution.

## Optional surface-geometry supervision

`MatterConfig.predict_geometry` adds three auxiliary PAM outputs without
changing the 2-D image-formation operator: a normalized signed world-space
object-surface normal, a positive pinhole-to-surface depth, and geometry
confidence. They are supervised by the renderer's `N_object`, `D_object`, and
validity masks using cosine normal loss, robust log-depth loss, and a masked
log-depth gradient loss. Cross-background operator consistency also constrains
these predictions because paired sequences share object geometry and pose.

This is a direct supervised surface prediction task, not a claim that
refractive flow alone uniquely determines 3-D geometry. Full refractive 3-D
recovery would additionally require calibrated cameras, known or estimated
IOR, surface/thickness parameterization, differentiable Snell-law ray tracing,
integrability, and temporal rigidity. The geometry head is therefore reported
with normal angular error and depth AbsRel/RMSE as an auxiliary multi-task
extension, while the principal PRISM compositor remains the 2-D reusable
colored refractive operator.
