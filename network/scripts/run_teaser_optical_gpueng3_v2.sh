#!/usr/bin/env bash
set -Eeuo pipefail

base=/home/ruai-u6151/prism-inverse-ab
root="$base/eval/comparisons/prism-test128-v1/prism-kubectl-128-v26-common-v1"
code="$root/network"
run_name="prism-gpueng3-128-teaser-optical-four-v2"
out="$base/output/$run_name"
data="$base/data"
mkdir -p "$out" "$out/tmp"

export PYTHONUNBUFFERED=1 OPENCV_IO_ENABLE_OPENEXR=1
export TMPDIR=/tmp/prism3 HF_HOME="$base/hf-cache"
mkdir -p "$TMPDIR"
export WANDB_BASE_URL=https://api.forge.coreweave.com
export PYTHONPATH="$base/experiment:$code:$base/third_party/sam2:${PYTHONPATH:-}"
export PRISM_TRACE_ROOT="$out/trace"
export PRISM_TRACE_EXPECTED_SEQUENCES=4
export PRISM_TRACE_WANDB_NAME="$run_name-trace"
export PRISM_TEASER_OPTICAL_ROOT="$out/optical"
export PRISM_TEASER_SELECTED_IDS="procedural_test_0107_solid_asymmetric_ornament_div2k_0890_bg006_seq0002,procedural_test_0108_hollow_bottle_div2k_0812_bg003_seq0000,objaverse_07f3c0290fb040bc9e31ec894e50b1dc_div2k_0803_bg006_seq0002,procedural_test_0105_solid_lens_or_ellipsoid_div2k_0881_bg003_seq0000"
export PRISM_MEMATTE_ROOT="$base/third_party/MEMatte"
export PRISM_MEMATTE_CHECKPOINT="$base/models/MEMatte_AIM500.pth"
export PRISM_SAM2_ROOT="$base/third_party/sam2"
export PRISM_SAM2_CHECKPOINT="$base/models/sam2.1_hiera_large.pt"

test -f "$root/.runtime-ready"
test -f "$data/dataset_manifest.json"
test "$(find "$data/test" -maxdepth 1 -name '*_sequence_meta.json' | wc -l)" -ge "$PRISM_TRACE_EXPECTED_SEQUENCES"
test "$(sha256sum "$base/models/prism_stage4_best.pt" | cut -d' ' -f1)" = fa1071de93de9adabb76822452a3d642236b8044b3036638495338f84c0bafcf
source "$base/venv/bin/activate"
export WANDB_API_KEY="$(python -c 'import json; import base64; print(base64.b64decode(json.load(open("/home/ruai-u6151/.config/prism/credentials.json"))["WANDB_API_KEY"]).decode())')"
export HF_TOKEN="$(python -c 'import json; import base64; print(base64.b64decode(json.load(open("/home/ruai-u6151/.config/prism/credentials.json"))["HF_TOKEN"]).decode())')"
python -c 'import lpips' || python -m pip install --no-cache-dir 'lpips==0.1.4'
python - <<'PY'
import torch
assert torch.cuda.is_available()
print('PRISM_GPUENG3_INVERSE_PRIORITY_RUNTIME_OK', torch.cuda.get_device_name(0), flush=True)
PY

python "$base/experiment/export_teaser_optical_gpueng3_v2.py" \
  --test-data "$data/test" \
  --checkpoint "$base/models/prism_stage4_best.pt" \
  --save-dir "$out/native" \
  --stage 4 --mode test \
  --batch-size 1 --clip-length 8 --workers 0 --eval-workers 0 \
  --prompt-mode point --prompt-seed 7 --seed 7 \
  --sam2-checkpoint "$PRISM_SAM2_CHECKPOINT" \
  --mam2-matter-backend external_mematte \
  --mematte-root "$PRISM_MEMATTE_ROOT" \
  --mematte-config "$PRISM_MEMATTE_ROOT/configs/MEMatte_S_topk0.25_win_global_long.py" \
  --mematte-checkpoint "$PRISM_MEMATTE_CHECKPOINT" \
  --mematte-max-tokens 12000 --mematte-patch-decoder \
  --mematte-train-decoder --mematte-train-backbone \
  --matte-frame-chunk-size 4 --sam2-temporal-checkpoint-chunk-size 4 \
  --fixed-mam2-alpha --flow-refinement \
  --foreground-refinement --foreground-detail-refinement \
  --flow-guided-latent-bridge --use-mam2-foreground-prior \
  --evidence-flux-only --flux-spatial-singleton-evidence \
  --refine-inverse-evidence \
  --background-exclusion-dilation 1 \
  --flux-direct-observation-threshold 0.1 \
  --flux-min-consistent-observations 2 \
  --refinement-steps 3 --refractive-kernel-size 1 \
  --refractive-kernel-radius-fraction 0.015625 \
  --refractive-kernel-center-bias 4.0 \
  --completion-variant diffusion --completion-backbone ffc \
  --diffusion-model "$base/FLUX.1-Fill-dev" \
  --diffusion-steps 30 --diffusion-guidance-scale 30 \
  --diffusion-mask-dilation 0 --diffusion-min-working-side 1024 \
  --diffusion-pixel-refinement --diffusion-dtype bfloat16 \
  --diffusion-cpu-offload auto --amp-dtype bfloat16 \
  --loss-profile balanced_detail_recovery \
  --qualitative-limit 0 --max-eval-batches 960 \
  --compute-lpips \
  --wandb-mode online --wandb-project PRISM \
  --wandb-entity sukhuny-seoul-national-university \
  --wandb-group "$run_name" \
  --wandb-run-name "$run_name" \
  --wandb-image-limit 4

test -f "$PRISM_TRACE_ROOT/summary.json"
test "$(find "$PRISM_TEASER_OPTICAL_ROOT" -maxdepth 1 -name '*.npz' | wc -l)" -eq 4
export PRISM_RENDERER_SOURCE="$code/src/renderer.py"
cd "$base/experiment/teaser_figure"
# The paper figure builder reads a fixed figure-asset directory. Populate it
# only after all four independent optical exports pass the completion check.
mkdir -p teaser_four_rows_128/optical
cp "$PRISM_TEASER_OPTICAL_ROOT"/*.npz teaser_four_rows_128/optical/
python build_teaser_four_rows_optical_128.py
test -s teaser_four_rows_128/prism_teaser_four_rows_optical_128.pdf
cp teaser_four_rows_128/prism_teaser_four_rows_optical_128.pdf "$out/"
cp teaser_four_rows_128/prism_teaser_four_rows_optical_128.svg "$out/"
echo "PRISM_TEASER_OPTICAL_EXPORT_COMPLETE sequences=4"
