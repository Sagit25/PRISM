# PRISM: Physics-guided Refraction-aware Inverse Scene Matting

PRISM jointly estimates a reusable counterfactual background and a per-frame
colored refractive foreground operator from one fixed-camera video containing
a moving transparent object. It does not require a clean plate at inference.

The repository contains:

- `RCDatasetCreation/`: an RCTrans-derived Mitsuba generator for fixed-camera,
  moving-object sequences with complete physical supervision.
- `network/`: official SAM2.1 integration, a clean-room executable MAM2
  mask/trimap/alpha pipeline, shared-background inversion, physics operator
  heads, training, inference, metrics, and tests.

## Dataset generation

Use Python 3.10+ and Mitsuba 3.7.0. Start with the included CPU contract test:

```bash
cd RCDatasetCreation
python render_dataset.py --conf configs/dataset_prism_main_smoke.yaml
python tools/validate_prism_contract.py result/prism_main_smoke/train --require-pairs
```

Generate the full main split on a CUDA-capable renderer:

```bash
cd RCDatasetCreation
python -m pip install -r requirements-assets.txt
python tools/prepare_prism_assets.py all
python tools/prepare_prism_assets.py validate
python render_dataset.py --conf configs/dataset_prism_main.yaml
python tools/validate_prism_contract.py result/prism_main/train --require-pairs
python tools/validate_prism_contract.py result/prism_main/validation --require-pairs
python tools/validate_prism_contract.py result/prism_main/test --require-pairs
python tools/freeze_prism_manifest.py result/prism_main
python tools/freeze_prism_manifest.py result/prism_main --verify
```

The main generator renders disjoint train/validation/test resource partitions.
Meshes and backgrounds reserved for validation are selected deterministically
from the training resource lists; the test lists remain untouched. The main
and reflection-diagnostic configurations render at 512x512 with 128 samples
per pixel. `dataset_manifest.json` records the partitions and
`artifact_manifest.json` content-hashes every generated file. Run `--verify`
before each training/evaluation campaign to reject changed or missing files.
The smoke assets are contract-test fixtures only. The separately generated
research pack contains 120/30 train-pool/test meshes and 800/100 DIV2K
train/test backgrounds; its source, license, processing metrics, rejection
audit, and hashes are recorded in
`RCDatasetCreation/asset_manifests/prism_research_assets_v1.json`. A full run
deliberately refuses to start until each selected research index contains at
least two distinct assets. The default main configuration produces 4,800
sequences and 38,400 frames, including validation and test, so provision large
scratch storage for its HDR and debug passes.

The reflection/caustic/shadow diagnostic split is separate from the
refraction-dominant main split:

```bash
python render_dataset.py --conf configs/dataset_prism_diagnostic_reflection.yaml
```

## Network installation and validation

Use Python 3.10+ and PyTorch 2.5.1+. W&B is optional and disabled by default.
The bootstrap command installs PRISM, checks out the official SAM2 source at
the research-pinned revision, downloads the official SAM2.1 Large checkpoint,
and verifies its SHA-256 digest:

```bash
network/scripts/install_official_sam2.sh
python -m pytest network/tests
OPENCV_IO_ENABLE_OPENEXR=1 python network/examples/check_rctrans_dataset.py \
  --root RCDatasetCreation/result/prism_main/train --check-pairs
```

The local checkout is placed at `network/third_party/sam2` and the checkpoint
at `network/checkpoints/sam2.1_hiera_large.pt`. Both are discovered
automatically and ignored by Git; their pinned source, URL, and digest are
tracked in `network/third_party/SAM2_ASSETS.json`. Set `PRISM_SAM2_ROOT` or
`PRISM_SAM2_CHECKPOINT` only to override these defaults.

## Stage-wise training and evaluation

The packaged `prism-train` command runs one stage at a time. The primary
protocol uses one deterministic positive point; `--prompt-mode mask` is an
oracle and must be reported separately. Validation selects the best checkpoint
and the held-out test split is read only after training.

```bash
prism-train \
  --train-data RCDatasetCreation/result/prism_main/train \
  --val-data RCDatasetCreation/result/prism_main/validation \
  --test-data RCDatasetCreation/result/prism_main/test \
  --checkpoint checkpoints/stage1b/prism_stage1b_best.pt \
  --stage 2 --prompt-mode point \
  --epochs 10 \
  --batch-size 2 \
  --mode both \
  --save-dir checkpoints/stage2
```

Run stages `1a`, `1b`, `2`, `3`, and `4` separately, passing the preceding
format-v6 checkpoint through `--checkpoint`. Stage 1A learns mask/trimap
semantics; Stage 1B learns alpha with the in-tree matter or the official
MEMatte decoder. Stage 2 learns PRISM-PAM against the ground-truth background.
Stage 3 enables deterministic background recovery and decays teacher forcing.
Stage 4 reconnects all differentiable paths while keeping the original SAM2
and MEMatte ViT encoders frozen. Stages 3/4 require
`--paired-backgrounds --batch-size 2`; bypasses are explicit ablation flags.
`--resume` restores model, optimizer, scheduler, Python/Torch RNG and
best-validation state exactly.

Stages 1A/1B can mix VOS and image/video-matting records through repeatable
`--stage1-manifest` JSONL arguments. Every row contains `dataset_kind`,
`frames`, and either `object_masks`, `trimaps`, or `alpha`; paths are relative
to the manifest. VOS supervises masks; matting records supervise the predicted
trimap and MAM2 alpha; synthetic-physics records may supervise all three. Stage
1B automatically filters out manifest rows without alpha labels.

Evaluation writes region-separated direct/inverse/true-hole background metrics,
SSIM and optional LPIPS, alpha SAD/MSE/gradient/connectivity/boundary scores,
flow EPE/bad-pixel rate, render metrics, evidence-preservation error and paired
unseen-background recomposition. It also saves qualitative montages and tensor
assets with checkpoint, dataset-manifest, SAM2 and diffusion provenance.

## PRISM-Base and PRISM-Diffusion

`--completion-variant base` is the default research model. It performs the
fixed-point decomposition with direct evidence, inverse-refracted evidence,
and a trainable deterministic LaMa/GLaMa-style FFC network for remaining true
holes. The FFC network runs at every fixed-point update, mixes global context
in the Fourier domain, and stays connected to PAM through autograd. Use
`--completion-backbone dilated` only for the legacy CNN ablation.

PRISM-Diffusion reuses a trained PRISM checkpoint and replaces only the final
true-hole candidate with a frozen image-inpainting diffusion pipeline. The
diffusion model runs once after the final inverse-splat update; hard compositing
preserves every direct and inverse-supported pixel exactly.

```bash
python -m pip install -e "./network[diffusion]"
prism-train \
  --test-data RCDatasetCreation/result/prism_main/test \
  --checkpoint checkpoints/stage4/prism_stage4_epoch020.pt \
  --stage 4 --mode test \
  --completion-variant diffusion \
  --diffusion-model MODEL_ID_OR_LOCAL_PATH \
  --diffusion-revision PINNED_REVISION \
  --paired-eval --batch-size 2 --compute-lpips \
  --save-dir results/prism-diffusion
```

Pass `--diffusion-adapter LORA_ID_OR_LOCAL_PATH` to load an adapted inpainting
LoRA. Diffusion weights and adapters remain external frozen assets and are not
duplicated in the compact PRISM checkpoint. Linear-RGB evidence is converted
to sRGB for diffusion and converted back before true-hole compositing. A
A dilated generation mask supplies seam context, while hard compositing still
edits only the exact true-hole mask. Diffusion training is rejected by the CLI.

Compare matched metric files only after checking checkpoint, dataset, prompt
and evidence-preservation invariants:

```bash
prism-compare results/prism-base/prism_stage4_metrics.json \
  results/prism-diffusion/prism_stage4_metrics.json \
  --output results/base-vs-diffusion.json
```

## Implemented physical contract

PRISM predicts premultiplied foreground `G`, opacity `alpha`, RGB color
transmission `C`, effective transmission `tau=(1-alpha)C`, refractive flow `u`,
bounded residual `R`, and confidence. It reconstructs

```text
Phi(x) = x + u(x)
I_hat(x) = G(x) + tau(x) * B_cf(Phi(x)) + R(x)
```

Transparent-interior evidence is inverted and forward-splatted into one shared
background canvas. Directly observed pixels have priority, inverse evidence
fills remaining supported locations, and learned completion edits only true
holes. See [network/README.md](network/README.md) for module and tensor details.

## Research completion status

The implementation and local physical-contract validation are complete. The
repository does not include trained extension weights or claim benchmark
numbers that have not been run. Publication completion still requires external
GPU compute, the official SAM2 checkpoint, full generated data, and baseline
implementations:

1. Render and validate the high-resolution main and diagnostic splits.
2. Pre-train PDD/MSS and Hiera LoRA on VOS and image/video matting data.
3. Train physics heads, then run the joint stage and the planned loss/iteration
   ablations.
4. Compare foreground operators with TOM-Net, CTOM-Net, and TransMatting;
   background recovery with median, STTN, ProPainter, and DiffuEraser; and
   joint decomposition with Omnimatte and OmnimatteRF.
5. Treat Snell-law depth/normal recovery as an extension after the 2D
   refractive-flow estimate is quantitatively stable.

The acceptance criteria and exact command sequence are recorded in
[network/RESEARCH_COMPLETION.md](network/RESEARCH_COMPLETION.md).
