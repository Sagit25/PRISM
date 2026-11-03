# v26 native 128px GPU1 run

This branch preserves the v26 MAM2-style PRISM architecture. It trains Stage
1A, 1B, 2, 3, and 4 from scratch using all train sequences. It does not reuse
the short 64px Stage2 checkpoint or the official MAM2-PAM2 experiment.

## Server prerequisites

- Checkout this branch in `/home/sukhun/prism-training-v26-128-20261103` and
  install a Python environment at `.venv`. Install
  `./network[data,sam2,experiment,evaluation,diffusion,test]`,
  `peft>=0.17.0`, official SAM2, and official MEMatte/Detectron2.
- Put the official `sam2.1_hiera_large.pt` and `MEMatte_AIM500.pth` weights
  under `network/checkpoints/`, or set `PRISM_SAM2_CHECKPOINT` and
  `PRISM_MEMATTE_CHECKPOINT` to their actual paths. Install the corresponding
  pinned source trees under `network/third_party/`.
- Provide `HF_TOKEN` and `WANDB_API_KEY` as server environment variables,
  or put the approved base64 JSON credential file at
  `/home/sukhun/.config/prism/credentials.json` with owner-only mode `600`.
  Never put either value in this repository or a command line argument.
- The seven native 128px generation projects must have
  `.generation_complete` under `/home/sukhun/prism-datasets/128-20261103`.
  The launcher waits for them and verifies every source artifact, split
  contract, and merged manifest before training.

For a fresh checkout, the Python and official source setup is:

```bash
python3 -m venv .venv
/home/sukhun/.local/bin/uv pip install --python .venv/bin/python \
  torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
/home/sukhun/.local/bin/uv pip install --python .venv/bin/python \
  -e './network[data,sam2,experiment,evaluation,diffusion,test]' 'peft>=0.17.0,<1'
PYTHON_BIN="$PWD/.venv/bin/python" bash network/scripts/install_official_sam2.sh
PRISM_MEMATTE_CHECKPOINT="$PWD/network/checkpoints/MEMatte_AIM500.pth" \
  PYTHON_BIN="$PWD/.venv/bin/python" bash network/scripts/install_official_mematte.sh
```

The MEMatte command requires the AIM500 checkpoint to be present first.

From the checkout, start with:

```bash
export PRISM_EXPECTED_COMMIT="$(git rev-parse HEAD)"
bash network/scripts/run_prism_v26_128_gpu1.sh
```

The launcher selects physical GPU 1, checks the 96GB-class GPU, CUDA/PyTorch
operators, FLUX gated access, W&B online authentication, official model assets,
code tests, data manifests/contracts, and free disk. It does not start training
if any prerequisite fails. Training logs and checkpoints are under
`/home/sukhun/prism-training/v26-128-full/`.

To use GPU memory and overlap CPU decoding with training, the GPU1 launcher
starts with Stage 1 batch 2, paired Stage 2–4 batch 4, test batch 4, and eight
training data workers. It skips activation recomputation for Stage 1–2, where
the one-frame graph is smaller. Stage 3–4 keep activation checkpointing; FLUX
stays on GPU if its automatic memory check finds enough space. A CUDA OOM
automatically retries from completed/resumable checkpoints, first restoring
early-stage checkpointing when relevant, then reducing batches to 1/2/2. If
the minimum paired batch still OOMs in Stage 3/4, it enables FLUX CPU offload.
The launcher verifies all data once before training, so repeat per-sample
runtime contract checks are disabled for throughput.

Full test evaluation writes the iterative trace under
`results/prism-diffusion-flux-fill/refinement_trace/<sequence_id>/`. Each
sequence has a `progression.png` comparison; each joint iteration has a
detailed PNG, an all-frame contact sheet, and a lossless NPZ containing all
frames of B, alpha, F, tau, flow, R, and re-rendered I. `steps.json` records
the frame-specific render error and object PSNR. The file order reflects
iterations, but improvement is an empirical result to inspect rather than a
property guaranteed by the model.
