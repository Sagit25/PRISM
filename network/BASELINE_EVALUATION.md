# PRISM held-out comparison protocol

The unit of comparison is a held-out `test` sequence from
`/data/datasets/prism-full-64-materialized-v7r1/test`. Each model receives the
same ordered sequence and, for still-image outputs, frame 4 of the first eight
frames. The generated predictions, sequence IDs, metrics, and first twelve
preview images are saved under `/data/comparisons/prism-test64-v1/`.

## What each baseline may use

| Method | Input privilege | Output scored | Status |
| --- | --- | --- | --- |
| Telea | Frame 4 and oracle frame-4 object mask | Background | Runner implemented |
| Temporal visible median + Telea | Eight frames and oracle object masks | Background | Runner implemented |
| Official SAM2Matting Base+ | Eight frames and oracle first-frame mask | Alpha and object mask | Runner implemented |
| Official MAM2 | Eight frames and oracle first-frame mask | Alpha and object mask | Official code and weights now available; 256px internal-input smoke prepared |
| Official ObjectClear | Frame 4 and oracle frame-4 mask | Background | 16-sequence smoke passed; full test submitted |
| Official ProPainter | Eight frames and oracle per-frame masks | Background at frame 4 | Runner implemented |
| PRISM v25 | Eight frames and first-frame point prompt | Background, alpha, foreground, flow, recomposition | Eight saved test predictions aligned; full evaluation runner prepared |
| ROSE | Full video and oracle per-frame masks | Background | Official inference code and weights are available; a one-sequence smoke is prepared. The 8-frame test clips are extended to 17 frames by repeating the last frame nine times, and only original frame 4 is scored. This is a modified input contract, so it is reported separately. |
| Omnimatte family | Per-video optimization and masks | Alpha/layer; variant-dependent background | Not yet run; substantial per-test-video fitting |
| Official MatAnyone | Eight frames and oracle first-frame mask | Alpha at frame 4 | Runner implemented with 256px internal input; human-focused training domain |
| Official MatAnyone 2 | Eight frames and oracle first-frame mask | Alpha at frame 4 | 256px internal-input runner prepared; smoke pending |
| VideoMaMa / MEMatte / TransMatting | Method-specific video/image inputs and prompts | Alpha | Candidate matting controls; adapters and weights to verify |
| STTN / DiffuEraser / Object-WIPER | Video and per-frame masks | Background | Candidate video-removal controls; adapters and weights to verify |
| TOM-Net / CTOM-Net | Method-specific images and backgrounds | Refraction decomposition | Candidate optical controls; released weights and compatible inputs to verify |

The table records input privileges because a model receiving a GT mask has
more information than a fully automatic model. Methods are compared only on
outputs they actually produce. For example, an alpha-only matting model has no
background PSNR or refractive-flow EPE. An inpainting model has no alpha SAD.
The PRISM v25 saved-prediction comparison currently covers only the first
eight ordered test sequences. Its full runner reuses the completed Stage-4
checkpoint and exports every test prediction by sequence ID before the
common evaluator assigns numeric test indices. The official MatAnyone
weights require an internal input larger than 64 pixels because their
top-k memory attention otherwise has fewer keys than its configured k; we
upsample the observed frames and first mask to 256 pixels and return the
predicted alpha to the 64-pixel evaluation grid.

## Metrics

`evaluate_comparison_predictions.py` verifies the prediction's sequence ID and
shape against the dataset contract. For RGB outputs it computes linear-RGB
L1, MSE, RMSE and PSNR, and sRGB SSIM and AlexNet LPIPS. For scalar alpha it
computes L1, MSE, RMSE, PSNR and SAD. Binary masks receive IoU and F1.
Refractive flow, if present, receives EPE and bad-1/bad-3-pixel rates on
valid GT object pixels. Regions are the whole frame, object, a two-pixel
boundary band, outside the object, object pixels never directly visible in
any of the eight frames, and object pixels visible in another frame. SSIM's
window can include pixels outside the scored region. Region LPIPS copies GT
outside the scored region before measuring full-frame LPIPS; it is explicitly
reported as `lpips_srgb_region_composited`, not native masked LPIPS.

All metric definitions and target fields must be fixed before comparing the
methods. The paper should report a prespecified primary metric and complete
supplementary results, rather than choosing the metric on which PRISM happens
to rank highest after looking at the held-out test scores.

Before the full PRISM test evaluation finishes, the primary direct-background
comparison is fixed as mean per-sequence linear-RGB PSNR on the GT object
region (`background/object/psnr_db`). The never-directly-visible region PSNR,
whole-frame LPIPS, and SSIM are secondary. For alpha-producing methods, report
whole-frame alpha SAD per frame and object-region alpha L1 together with mask
IoU when available. PRISM's flow EPE, foreground object-region PSNR and
recomposition PSNR are separate diagnostic outputs, not fabricated scores for
methods that do not produce those fields. Report all measured columns and
per-sequence values, including unfavorable results.
