"""Precompute the fixed FLUX Fill prompt without occupying the training GPU."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from refractive_mam2.completion import DiffusionCompletionSettings
from refractive_mam2.config import BackgroundConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from diffusers import FluxFillPipeline

    model = FluxFillPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    with torch.inference_mode():
        prompt, pooled, _ = model.encode_prompt(
            prompt=BackgroundConfig.diffusion_prompt,
            device=torch.device("cpu"),
            num_images_per_prompt=1,
        )
    if prompt.shape[0] != 1 or pooled.shape[0] != 1:
        raise RuntimeError("Expected one prompt embedding")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "prompt": BackgroundConfig.diffusion_prompt,
            "prompt_embeds": prompt.detach().cpu(),
            "pooled_prompt_embeds": pooled.detach().cpu(),
        },
        args.output,
    )
    print("PRISM_FLUX_PROMPT_CPU_CACHE_COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()
