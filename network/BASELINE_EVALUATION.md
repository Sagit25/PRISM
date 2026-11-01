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
| Navier–Stokes inpainting | Frame 4 and oracle frame-4 object mask | Background | Full 960-sequence evaluation complete |
| Biharmonic inpainting | Frame 4 and oracle frame-4 object mask | Background | Full 960-sequence evaluation complete |
| Nearest visible frame + Telea fallback | Eight frames and oracle per-frame object masks | Background | Full 960-sequence evaluation complete; no geometric alignment |
| GrabCut | Frame 4 and oracle object bounding box | Binary alpha and object mask | Full 960-sequence evaluation complete; cannot represent fractional transparency |
| Official SAM2Matting Base+ | Eight frames and oracle first-frame mask | Alpha and object mask | Runner implemented |
| Official MAM2 | Eight frames and oracle first-frame mask | Alpha and object mask | Official code and weights now available; 256px internal-input smoke prepared |
| Official ObjectClear | Frame 4 and oracle frame-4 mask | Background | 16-sequence smoke passed; full test submitted |
| Official ProPainter | Eight frames and oracle per-frame masks | Background at frame 4 | Runner implemented |
| PRISM v26 (fixed MAM2-style alpha) | Eight frames and first-frame point prompt | Background, alpha, foreground, flow, recomposition | Stage-4 training in progress; full 960-sequence evaluation queued |
| PRISM joint v15 (SAM2Matting Base+ + ObjectClear + flow bridge) | Eight frames and first-frame point prompt | Background, alpha, foreground, flow, recomposition | Final completed VESSL checkpoint preserved on SNU PVC; same-protocol smoke and full evaluation submitted as one sequential Job. The checkpoint was fine-tuned with four-frame clips, so eight-frame evaluation is temporal extrapolation. |
| ROSE | Full video and oracle per-frame masks | Background | Official inference code and weights are available; a one-sequence smoke is prepared. The 8-frame test clips are extended to 17 frames by repeating the last frame nine times, and only original frame 4 is scored. This is a modified input contract, so it is reported separately. |
| Omnimatte family | Per-video optimization and masks | Alpha/layer; variant-dependent background | Not yet run; substantial per-test-video fitting |
| Official MatAnyone | Eight frames and oracle first-frame mask | Alpha at frame 4 | Runner implemented with 256px internal input; human-focused training domain |
| Official MatAnyone 2 | Eight frames and oracle first-frame mask | Alpha at frame 4 | 256px internal-input runner prepared; smoke pending |
| VideoMaMa / MEMatte / TransMatting | Method-specific video/image inputs and prompts | Alpha | Candidate matting controls; adapters and weights to verify |
| STTN / DiffuEraser / Object-WIPER | Video and per-frame masks | Background | Candidate video-removal controls; adapters and weights to verify |
| E2FGVI-HQ | Eight frames and oracle per-frame masks | Background | Official code and weights available; 64-pixel inference adapter and smoke still required |
| LaMa | Frame 4 and oracle frame-4 mask | Background | Official code and weights available; single-frame inpainting control to add |
| FBA Matting | Frame 4 and oracle trimap | Alpha and straight foreground | Official code and weights available; its predicted background is an ordinary matting layer, not necessarily the refractive counterfactual background |
| ViTMatte | Frame 4 and oracle trimap | Alpha | Official code and weights available; internal upsampling and trimap privilege must be disclosed |
| TOM-Net / CTOM-Net | Method-specific images and backgrounds | Refraction decomposition | Candidate optical controls; released weights and compatible inputs to verify |

The table records input privileges because a model receiving a GT mask has
more information than a fully automatic model. Methods are compared only on
outputs they actually produce. For example, an alpha-only matting model has no
background PSNR or refractive-flow EPE. An inpainting model has no alpha SAD.
PRISM v26 will export each test prediction by sequence ID before the common
evaluator assigns numeric test indices. The official MatAnyone
weights require an internal input larger than 64 pixels because their
top-k memory attention otherwise has fewer keys than its configured k; we
upsample the observed frames and first mask to 256 pixels and return the
predicted alpha to the 64-pixel evaluation grid.

## Metrics

`evaluate_comparison_predictions.py` verifies the prediction's sequence ID and
shape against the dataset contract. For RGB outputs it computes linear-RGB
L1, MSE, RMSE and PSNR, and sRGB SSIM and AlexNet LPIPS. For scalar alpha it
computes L1, MSE, RMSE, PSNR and SAD. Binary masks receive IoU and F1.
`extend_comparison_metrics.py` additionally reports sRGB L1/MSE/RMSE/PSNR,
CIEDE2000 color difference, Lab chroma error, linear luminance L1/PSNR,
Sobel-gradient L1, and the 95th-percentile linear absolute error for every
available RGB output and region. For straight foreground it separately scores
GT-alpha-valid object and boundary pixels (`alpha >= 0.05`). For alpha output
it adds binary precision, recall and F1 at thresholds 0.1, 0.5 and 0.9,
plus two-pixel-tolerant boundary F1 at each threshold. For predicted object
masks it adds precision, recall and boundary F1; the original mask F1 stays
in the common evaluator. An empty prediction and empty GT support score 1.
RGB intensities themselves have no threshold-independent F1; no F1 is
fabricated for an inpainting method that does not output alpha or a mask.

The additional classical controls passed a 16-sequence smoke and completed
all 960 test sequences in `prism-kubectl-classical-additional-v1-full`.
The resulting ten-method, 135-metric table is stored at
`/data/comparisons/prism-test64-v1/prism-kubectl-classical-additional-v1-full/classical_plus_existing_full960.csv`.
Matched test images are in that run's `matched_images` directory. Its
direct-background object PSNR is 18.193 dB (Navier–Stokes), 12.220 dB
(biharmonic), and 18.379 dB (nearest visible). GrabCut produces only binary
alpha/mask, with alpha F1 at threshold 0.5 of 0.269 and mask F1 of 0.363.
These are mean per-sequence test values. The privileged masks and box are
included in the protocol and must be disclosed when comparing methods.
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

## VESSL asset preservation and joint evaluation

The complete 64-pixel source archive already exists at
`/data/datasets/prism-full-64-singleflow-pack-v2-369367271785-output` on the
SNU PVC. Its completion manifest and all eight tar shards were rehashed on
the PVC and exactly matched the VESSL source manifest. The materialized test
split contains 960 held-out sequences. The VESSL `dataset_manifest.json` and
`models/MEMatte_AIM500.pth` also match the copies already on the PVC.

The final completed joint-model VESSL run is
`prism-joint-sota-v15-physical-alpha-flow-v2` (run 369367275412, code
`694539ecfa8acac573d76c71a27ec9d01ce047f0`). Its Stage-4 model-only
best checkpoint, the completed v15 visible-background alternative, and the
v12 Stage-2/Stage-4 best and resumable donor checkpoints are preserved under
`/data/checkpoints/vessl-final-migration`. The SHA-256 manifest is stored in
that directory and locally at
`/Users/yangsukhun/PRISM/comparison_results/prism_test64_v1/vessl_final_migration_manifest.json`.
The later v16 run failed and produced no checkpoint in the source volume.

`prism-kubectl-joint-v15-comparison-v1-full` first evaluates eight sequences.
Only if that succeeds does the same Job evaluate all 960 sequences. Both
phases use the pinned v15 model code and original official SAM2Matting Base+
and ObjectClear source revisions. Predictions are converted to the same
frame-4 `.npz` contract and passed through the common metric evaluator.
Matched images include GT and available baseline rows for the same ordered
test IDs. The checkpoint was trained on four-frame clips; the eight-frame
evaluation context is stated in the comparison table and must be considered
when interpreting its rank.

The one-time CPU Job `prism-kubectl-comparison-finalize-v2` waits for both
960-sequence summaries. It then writes a merged metric CSV and matched
images (including both PRISM variants) to
`/data/comparisons/prism-test64-v1/final-two-prism-methods`.
