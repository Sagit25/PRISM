# Flow-coupled released-model experiment

This optional PRISM backend uses **SAM2Matting** (ECCV 2026) for video
mask/alpha estimation and **ObjectClear** (CVPR 2026) for object-free
background completion. It does not load MEMatte or FLUX Fill weights in this mode.
The original PRISM backends remain available for controlled comparisons.

## Reproducible external assets

- SAM2Matting source: `FudanCVL/SAM2Matting`, tested at Git commit
  `73dd721d77b56749248aefe5e8824d7f61b9d13c`. The tested Tiny weights are
  from Hugging Face revision `4315db9c60d27fde396b09765748a0ca6c97bed5`,
  `checkpoints/SAM2Matting-SAM2.1Tiny.pt`, SHA-256
  `5b9321e3b51bc20f5b84c208746cc083dd3053dd701590f2e88dc8640afcc39d`.
- ObjectClear source: `zjx0101/ObjectClear`, tested for import at Git commit
  `c052d91ecd8772744a5ab97527441c813ab83009`. Use pretrained model
  `jixin0101/ObjectClear`, Hugging Face revision
  `c73af80888dbd519819d0a521b5c8d0f3cda6859`.
- Install PRISM with `pip install -e './network[data,sam2,experiment,evaluation,objectclear]'`
  and then the two official source checkouts. Do **not** combine the
  `objectclear` and FLUX-oriented `diffusion` extras in one environment: they
  require different `diffusers` API versions. An isolated import check passed
  with `diffusers==0.31.0`,
  `transformers==4.45.2`, `accelerate==1.0.1`, and
  `huggingface_hub==0.26.5` (but with a different PyTorch version from PRISM's
  pinned 2.5.x); this does **not** prove a full ObjectClear inference with the
  released weights or the combined environment. Respect each project's license.

Set `PRISM_SAM2_ROOT` to the SAM2Matting checkout **before Python starts**.
Its `sam2` fork must not be mixed with PRISM's default SAM2 checkout.
Pass the same checkout through `--official-sam2matting-root` and the released
weights through `--official-sam2matting-checkpoint`.

## Joint path

1. The released SAM2Matting tracker and first-frame prompt produce a video
   mask and initial alpha. The tracker is frozen. Its multi-scale alpha/ROI
   modules, however, can be fine-tuned in Stage 4.
2. PRISM-PAM predicts the first colored optical operator, including refractive
   flow `u`, RGB transmittance `tau`, foreground `G`, and residual `R`.
3. `B(x+u)`, normalized `u`, object support, and the current render error
   modulate SAM2Matting's released 1/16-resolution feature pyramid. Its
   original progressive alpha head is run again. This revised alpha changes
   the next PAM operator and the inverse-refracted background evidence.
4. ObjectClear runs on the direct-observation background canvas. The removal
   mask is filled from the observed first frame *only for ObjectClear's input*,
   so its official object encoder sees the actual target rather than a blank
   hole; known background pixels still come from direct observations. By default
   its complete attention-guided-fusion result is encoded back to its own
   SDXL VAE latent. A small, zero-initialized trainable latent adapter receives
   the inverse-refracted radiance, its confidence, the splatted flow, and the
   editable-hole mask. Its output is decoded by ObjectClear's frozen VAE.
5. The shared-background and render losses backpropagate through the alpha
   head/flow modulator, the optical operator, inverse splat, and background
   latent adapter. ObjectClear's expensive denoiser and SAM2Matting's video
   tracker remain frozen. “Joint” here refers to the connected trainable
   branches, **not** full fine-tuning of both generative backbones.

An **experimental** alternative injects the same trainable optical latent
adapter before ObjectClear's final `K` denoising steps. The released denoiser
weights remain frozen, but the render-loss gradient passes through those last
`K` denoising operations to the adapter and PRISM flow. This path uses raw
latents rather than ObjectClear's non-differentiable PIL attention-guided
fusion. It is slower and needs more GPU memory than the default post-fusion
adapter; it has only passed a fake-pipeline gradient test, **not** a released-
weight GPU run or a quality comparison.

The ObjectClear prior is cached for identical direct-evidence input, while
the flow-conditioned adapter is recomputed each iteration. Use
`--objectclear-dtype bfloat16` (or float32): the official SDXL VAE's fp16
force-upcast path is not safe to mutate between multiple autograd forwards.
At each scalar logging step, W&B records
`train/matte_flow_adapter_grad` and `train/background_flow_adapter_grad`
alongside the existing `train/pam_head_grad_flow`, so disconnected or
collapsed optical coupling is visible during Stage 4.

## Training flags

Use the existing PRISM Stage 2 -> Stage 3 -> Stage 4 progression and existing
dataset/archive arguments. Stage 2 may start directly from the released
SAM2Matting weights. Stage 3/4 use the previous PRISM checkpoint and include:

```text
PRISM_SAM2_ROOT=<SAM2Matting checkout>
--mam2-matter-backend official_sam2matting
--official-sam2matting-root <SAM2Matting checkout>
--official-sam2matting-checkpoint <SAM2Matting Tiny checkpoint>
--official-sam2matting-variant tiny
--completion-variant diffusion
--diffusion-backend objectclear
--objectclear-source-root <ObjectClear checkout>
--diffusion-model jixin0101/ObjectClear
--diffusion-revision c73af80888dbd519819d0a521b5c8d0f3cda6859
--objectclear-dtype bfloat16
--evidence-pretrained-only
--refinement-steps 1
```

For the experimental inside-denoiser variant, add
`--objectclear-late-denoise-steps 2 --no-objectclear-fused-output
--diffusion-cpu-offload off`. Keep `K` below the chosen inference step count. Do not
enable this in a full training run until the real-weight GPU smoke passes with
`--late-denoise-steps 2` and available VRAM is measured.

`--evidence-pretrained-only` is the clearer alias for the historical
`--evidence-flux-only` switch; it does not select FLUX when
`--diffusion-backend objectclear` is present. ObjectClear's released training
resolution is 512 on the short side, so a 64-pixel PRISM frame is upscaled for
its frozen prior and reduced again after latent adaptation. This may be slower
than the old 64-pixel completion network. Do not infer quality gains from the
model names: compare alpha, background-hole, flow, and final render metrics on
the same fixed split.

## Verification so far

- All PRISM `network/tests` pass, including a joint-graph backward test, a
  checkpoint round trip for both trainable branches, and a check that the
  ObjectClear prompt contains an observed object rather than a blank hole.
  The mock-model joint test uses one render loss and checks both the default
  post-fusion adapter and the experimental inside-denoiser path; it does not
  establish released-weight runtime or output quality.
- The released SAM2Matting Tiny checkpoint passed checksum, strict load
  (`missing_keys=[]`, `unexpected_keys=[]`), one-frame CPU inference, and
  released alpha-head backward on the official source API. On a synthetic
  16-pixel clip, one flow-adapter update made its released alpha head respond
  to a changed refractive flow with a nonzero flow gradient. This is a graph
  check, **not** a quality result.
- The released ObjectClear source imports with the tested dependency set.
  Its full multi-gigabyte checkpoint and CUDA execution have **not yet been
  validated** in this worktree. A GPU smoke run is required before claiming
  the two released models run together end to end.

The read-only-weight smoke entry point is
`network/scripts/smoke_joint_sota_flow.py`. On a CUDA worker with both official
sources and the released SAM2Matting checkpoint available, pass
`--sam2matting-root`, `--sam2matting-checkpoint`, and `--objectclear-root`.
It performs one synthetic two-frame forward/backward using the **single shared
render loss**, checks that ObjectClear actually ran, and fails if the
SAM2Matting alpha head, both flow adapters, PRISM-PAM, or PRISM's two-channel
flow head have zero gradient. It does not submit a cloud run or train epochs.
The isolated VESSL smoke configuration in
`network/scripts/prism_joint_sota_flow_smoke_v11_v1.yaml` runs both the
post-fusion adapter and one final denoising step with the released weights.
It reports peak reserved VRAM and trainable parameter counts, including the
expected zero trainable parameters in ObjectClear's original UNet. A pass
proves a connected optimization step, not superior image quality or full
fine-tuning of ObjectClear.
