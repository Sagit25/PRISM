# Official SAM2 integration details

## Hook location

The extension overrides only `SAM2VideoPredictor._track_step`. The parent
method already returns all four values required for MSS:

```text
current_out, sam_outputs, high_res_features, pix_feat_with_memory
```

The clean feature is reconstructed from the last entry of
`current_vision_feats` before memory attention:

```python
non_memory = current_vision_feats[-1].permute(1, 2, 0).reshape(B, C, H, W)
```

MSS then executes with the official sparse/dense prompt embeddings:

```text
mask = PDD.decode_mask(pix_feat_with_memory, official_sam_mask, user_prompt)
pseudo_prompt = SAM2PromptEncoder(sigmoid(mask))
trimap = PDD.decode_trimap(non_memory, mask, pseudo_prompt, high_res_features)
```

Both calls use one `PromptableDualModeDecoder` instance. A second decoder object
would not be the parameter-sharing siamese design described by MAM2.

The PDD mask head predicts a residual. Its final convolution is initialized to
zero, so an untrained extension exactly preserves official SAM2 mask logits.
When `replace_sam_mask_for_memory=True`, the learned residual is added at both
SAM2 low and high resolution before the parent `track_step` runs the memory
encoder. Thus the refined mask, never the trimap, becomes future mask memory.

## Subclass and checkpoint boundary

Meta's `_load_checkpoint` intentionally rejects missing keys. The custom builder
uses Hydra's `_target_` override to instantiate a real subclass and then
performs these operations in order:

1. instantiate `MAM2VideoPredictor` from the official SAM2 config;
2. attach PDD/MSS in its constructor;
3. load the official checkpoint with `strict=False`, while accepting missing
   keys only under `mam2_mss.*` and accepting no unexpected keys;
4. inject LoRA into the already-loaded Hiera linear layers;
5. optionally load the compact MAM2/refractive checkpoint strictly.

No original SAM2 parameter is silently ignored or overwritten.

## Output contract

The semantic stage returns:

| Name | Shape | Source |
| --- | --- | --- |
| `mask_logits` | `[B,T,1,Hm,Wm]` | memory-conditioned PDD pass |
| `trimap_logits` | `[B,T,3,Ht,Wt]` | shared PDD on non-memory features |
| `non_memory_features` | `[B,T,C,h,w]` | image feature before memory attention |

During training these are inserted into `current_out` as
`mam2_mask_logits`, `mam2_trimap_logits`, and
`mam2_non_memory_features`, retaining gradients. The public video inference API
compacts unknown fields, so detached copies are held in a short-lived side cache
and consumed frame-by-frame by `propagate_mam2_backbone`.

`MAM2VideoPredictor.forward_mam2_clip` is the differentiable training path. It
runs `forward_image`, slices the time-major backbone features, and repeatedly
calls the official `track_step`, inserting each result into the same
conditioning/non-conditioning memory dictionaries used by SAM2. Inputs must be
resized to `predictor.image_size` and normalized with
`normalize_sam2_training_frames`.

## Trimap classes

Class indices are `BG=0`, `UNKNOWN=1`, `FG=2`. For transparent objects the
whole visible object interior may be UNKNOWN. `build_transparency_trimap`
creates this label from geometric support and alpha; it does not use a narrow
morphological boundary as the only unknown region.

## Fixed-camera and refractive-flow convention

The physical pipeline builds one image-coordinate background asset for the
complete clip, so there is no optical-flow correspondence input. Refractive
flow is target-to-source within this fixed canvas:

```text
Phi(x) = x + refractive_flow(x)
B_refracted(x) = B_cf(Phi(x))
```

The inverse solver recovers `(I-G-R)/tau` at `Phi(x)` and bilinearly
forward-splats it into the same canvas. Directly exposed observations retain
priority; inverse evidence fills unseen positions before learned completion.

## Compatibility boundary

The integration targets the official `facebookresearch/sam2` SAM2.1 predictor
API in which `_track_step` returns
`(current_out, sam_outputs, high_res_features, pix_feat)`. The builder disables
the VOS fully-compiled predictor because Python-level PDD/MSS output collection
requires the ordinary predictor path. Image-encoder compilation may be enabled
upstream after validating the selected environment.
