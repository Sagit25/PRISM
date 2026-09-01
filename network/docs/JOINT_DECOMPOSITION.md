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
- target-to-background displacement `u`, with `Phi(x)=x+u(x)`;
- bounded additive residual `R`;
- inverse-evidence confidence `c`.

The renderer is

```text
I_hat(x) = G(x) + tau(x) * B_cf(Phi(x)) + R(x)
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
weighted by support, transmission and confidence and bilinearly forward-splat
to `Phi(x)` on the shared canvas. The splat is differentiable with respect to
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

After the last fusion, the operator is evaluated once more on the final shared
background. The final render therefore uses a mutually aligned operator and
background instead of adjacent fixed-point iterates. No detach is used in the
default joint configuration, so render and component losses train both sides.
Earlier oracle/frozen stages remain available for stable curriculum training.

## Reusability supervision

`reusable_operator_consistency` and `joint_stage_loss` accept predictions of
the same object trajectory rendered on two different backgrounds. They force
`alpha`, `G`, `C`, `tau`, `u`, `R` and confidence to remain invariant while
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

## Not implemented: 3-D geometry recovery

The package exposes refractive flow but does not claim that flow alone uniquely
determines 3-D geometry. A follow-up geometry module needs calibrated cameras,
rigid object pose across frames, known or estimated IOR, surface/thickness
parameterization, Snell-law ray tracing, and integrability/temporal rigidity
constraints. Keeping this boundary explicit prevents the current 2-D inverse
renderer from making an unsupported 3-D claim.
