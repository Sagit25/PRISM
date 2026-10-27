"""Compare frozen FLUX Fill canvas sizes on one saved PRISM W&B montage.

This is a bounded diagnostic, not a training entry point. The montage comes
from the first Stage-4 image, so its displayed 64px direct canvas and GT can
be compared without loading a trainable checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from refractive_mam2.completion import (
    DiffusionCompletionSettings,
    FrozenDiffusionBackgroundCompleter,
    _linear_to_srgb,
    _srgb_to_linear,
)


def _panel(montage: Image.Image, index: int) -> Image.Image:
    column, row = index % 4, index // 4
    # _supervised_prediction_montages uses 160px cells and 64px images.
    box = (column * 160 + 48, row * 84 + 20,
           column * 160 + 112, row * 84 + 84)
    return montage.crop(box)


def _rgb_tensor(image: Image.Image, device: str) -> torch.Tensor:
    srgb = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32).copy())
    srgb = srgb.permute(2, 0, 1)[None].to(device) / 255.0
    return _srgb_to_linear(srgb)


def _save_rgb(value: torch.Tensor, path: Path) -> None:
    srgb = _linear_to_srgb(value.detach().float().cpu())[0]
    pixels = (srgb.permute(1, 2, 0).mul(255).round().byte().numpy())
    Image.fromarray(pixels, mode="RGB").save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--montage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sides", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    montage = Image.open(args.montage).convert("RGB")
    if montage.size != (640, 924):
        raise ValueError(f"unexpected Stage-4 montage size: {montage.size}")
    # Stable Stage-4 panel order: B pred=33, B GT=34, direct support=35,
    # direct B pred=36. This diagnostic uses exactly the saved sample.
    direct = _rgb_tensor(_panel(montage, 36), "cuda")
    target = _rgb_tensor(_panel(montage, 34), "cuda")
    support = np.asarray(_panel(montage, 35).convert("L")) > 127
    hole = torch.from_numpy((~support).copy())[None, None].to("cuda")
    if not bool(hole.any()) or not bool((~hole).any()):
        raise ValueError("montage has no mixed observed and missing background")
    _save_rgb(direct, args.output / "direct.png")
    _save_rgb(target, args.output / "target.png")
    measurements = {"hole_fraction": float(hole.float().mean())}
    prior = _rgb_tensor(_panel(montage, 33), "cuda")
    measurements["saved_flux128_hole_mse"] = float((prior - target).square()[hole.expand_as(prior)].mean())

    import diffusers

    pipe = diffusers.FluxFillPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16,
    )
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_model_cpu_offload(device="cuda")
    for side in args.sides:
        try:
            completer = FrozenDiffusionBackgroundCompleter(
                pipe,
                settings=DiffusionCompletionSettings(
                    inference_steps=args.steps,
                    guidance_scale=30.0,
                    min_working_side=side,
                    image_cache_entries=0,
                ),
                execution_device="cuda",
                cache_prompt_embeddings=True,
            )
            candidate = completer(direct, (~hole).float(), hole)
            candidate = torch.where(hole, candidate, direct)
            mse = float((candidate - target).square()[hole.expand_as(candidate)].mean())
            measurements[f"flux{side}_hole_mse"] = mse
            _save_rgb(candidate, args.output / f"flux{side}.png")
            print(f"PRISM_FLUX_SIZE_PROBE side={side} hole_mse={mse:.6f}", flush=True)
        except Exception as exc:
            measurements[f"flux{side}_error"] = f"{type(exc).__name__}: {exc}"
            print(f"PRISM_FLUX_SIZE_PROBE_FAILED side={side} error={exc}", flush=True)
        finally:
            torch.cuda.empty_cache()
    (args.output / "metrics.json").write_text(json.dumps(measurements, indent=2))
    print("PRISM_FLUX_SIZE_PROBE_COMPLETE " + json.dumps(measurements), flush=True)


if __name__ == "__main__":
    main()
