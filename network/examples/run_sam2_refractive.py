"""Single-object end-to-end inference with the official SAM2.1 predictor."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from refractive_mam2 import (
    SAM2RefractiveRunner,
    build_mam2_video_predictor,
    build_physics_pipeline_for_sam2,
    load_refractive_checkpoint,
)
from refractive_mam2.vendor import SAM2_CHECKPOINT, SAM2_CONFIG


def load_rgb_frames(video_directory: Path) -> torch.Tensor:
    paths = sorted(
        [*video_directory.glob("*.jpg"), *video_directory.glob("*.jpeg")],
        key=lambda path: path.name,
    )
    if not paths:
        raise FileNotFoundError("video directory must contain JPEG frames")
    arrays = [np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0 for path in paths]
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("all frames must have the same resolution")
    return torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True, help="directory of JPEG frames")
    parser.add_argument("--sam2-config", default=SAM2_CONFIG)
    parser.add_argument("--sam2-checkpoint", type=Path, default=SAM2_CHECKPOINT)
    parser.add_argument("--checkpoint", type=Path, help="trained refractive MAM2 checkpoint")
    parser.add_argument("--point", nargs=2, type=float, metavar=("X", "Y"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("refractive_output.pt"))
    args = parser.parse_args()

    predictor = build_mam2_video_predictor(
        args.sam2_config,
        args.sam2_checkpoint,
        device=args.device,
    )
    physics = build_physics_pipeline_for_sam2().to(args.device).eval()
    if args.checkpoint:
        load_refractive_checkpoint(args.checkpoint, predictor, physics)

    state = predictor.init_state(video_path=str(args.video))
    predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=0,
        obj_id=1,
        points=np.asarray([args.point], dtype=np.float32),
        labels=np.asarray([1], dtype=np.int32),
    )
    frames = load_rgb_frames(args.video)
    runner = SAM2RefractiveRunner(predictor, physics)
    with torch.inference_mode():
        output = runner.run(frames, state, object_id=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "alpha": output.matter.alpha.cpu(),
            "straight_foreground": output.matter.straight_foreground.cpu(),
            "premultiplied_foreground": output.matter.premultiplied_foreground.cpu(),
            "color_transmission": output.matter.color_transmission.cpu(),
            "transmittance": output.matter.transmittance.cpu(),
            "refractive_flow": output.matter.refractive_flow.cpu(),
            "residual": output.matter.residual.cpu(),
            "counterfactual_background": output.background.background.cpu(),
            "direct_background_coverage": output.background.direct_coverage.cpu(),
            "inverse_background_coverage": output.background.inverse_coverage.cpu(),
            "background_uncertainty": output.background.uncertainty.cpu(),
            "matter_confidence": output.matter.confidence.cpu(),
            "reconstruction": output.reconstructed_frames.cpu(),
            "trimap_logits": output.backbone.trimap_logits.cpu(),
            "mask_logits": output.backbone.mask_logits.cpu(),
        },
        args.output,
    )


if __name__ == "__main__":
    main()
