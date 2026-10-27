"""Reproducible stage-wise training and evaluation entry point for PRISM."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import platform
import random
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.checkpoint import checkpoint

from .background import BackgroundOutput, MaskedTemporalBackground
from .completion import (
    DiffusionCompletionSettings,
    FrozenDiffusionBackgroundCompleter,
)
from .color import linear_to_srgb
from .config import (
    BackgroundConfig,
    LossWeights,
    MAM2MatteConfig,
    MatterConfig,
    PipelineConfig,
    SAM2IntegrationConfig,
)
from .dataset import (
    RCTransBatch,
    RCTransPRISMDataset,
    EpochShuffleSampler,
    build_paired_prism_dataloader,
    prism_collate,
)
from .fba_prior import FBAPretrainedForegroundPrior
from .logger import WandbLogger
from .losses import (
    RefractiveGroundTruth,
    RefractiveLoss,
    _far_background_support,
    _identifiable_color_transmission,
    _leave_one_out_background,
    _past_visible_refracted_background,
    _straight_foreground_support,
    reusable_operator_consistency_in_batch,
)
from .matter import PhysicsMatterOutput
from .pipeline import RefractiveMAM2, RefractiveMAM2Output
from .pdd import PromptableDualModeDecoder
from .renderer import recompose, warp_background
from .runner import (
    load_refractive_checkpoint,
    load_stage1_mam2_checkpoint,
    load_refractive_training_state,
    save_refractive_checkpoint,
)
from .sam2_integration import MAM2VideoPredictor, build_mam2_video_predictor
from .semantic_dataset import ManifestSemanticDataset, semantic_collate
from .streaming_dataset import VesslShardCyclingDataset
from .training import (
    SemanticTargets,
    configure_stage1a,
    configure_stage1b,
    configure_stage2,
    configure_stage3,
    configure_stage4,
    joint_stage_loss,
    normalize_sam2_training_frames,
    physics_stage_loss,
    selective_semantic_loss,
)
from .types import MAM2BackboneOutput
from .vendor import (
    MEMATTE_CHECKPOINT,
    MEMATTE_CONFIG,
    MEMATTE_ROOT,
    SAM2_CHECKPOINT,
    SAM2_CONFIG,
    mematte_setup_hint,
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _autocast_context(device: torch.device, amp_dtype: str):
    if amp_dtype == "float32":
        return contextlib.nullcontext()
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[amp_dtype]
    return torch.autocast(device_type=device.type, dtype=dtype)


def _training_state(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    *,
    epoch: int,
    global_step: int,
    best_value: float,
    batch_in_epoch: int = 0,
) -> dict[str, object]:
    return {
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "global_step": global_step,
        "best_value": best_value,
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "python_random_state": random.getstate(),
        "torch_rng_state": torch.random.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }


def _restore_training_state(
    state: dict[str, object],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
) -> tuple[int, int, int, float]:
    if not state:
        raise RuntimeError("resume checkpoint contains no training state")
    try:
        optimizer.load_state_dict(state["optimizer"])
    except ValueError as exc:
        raise RuntimeError(
            "resume optimizer layout does not match the current trainable modules; "
            "load the previous model with --checkpoint and start a fresh optimizer "
            "instead of using --resume across a training-design change"
        ) from exc
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    random.setstate(state["python_random_state"])
    torch.random.set_rng_state(state["torch_rng_state"])
    cuda_state = state.get("cuda_rng_state", [])
    if torch.cuda.is_available() and cuda_state:
        torch.cuda.set_rng_state_all(cuda_state)
    return (
        int(state["epoch"]),
        int(state.get("batch_in_epoch", 0)),
        int(state["global_step"]),
        float(state["best_value"]),
    )


def _device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def _sha256_path(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    digest = hashlib.sha256()
    paths = (
        [path]
        if path.is_file()
        else sorted(item for item in path.rglob("*") if item.is_file())
    )
    for item in paths:
        relative = item.name if path.is_file() else str(item.relative_to(path))
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _reproducibility_record(args: argparse.Namespace) -> dict[str, object]:
    manifest = None
    if args.test_data is not None:
        candidate = args.test_data.parent / "artifact_manifest.json"
        if not candidate.is_file():
            candidate = args.test_data.parent / "dataset_manifest.json"
        manifest = candidate if candidate.is_file() else None
    local_diffusion = Path(args.diffusion_model) if args.diffusion_model else None
    local_adapter = Path(args.diffusion_adapter) if args.diffusion_adapter else None
    mematte_checkpoint = args.mematte_checkpoint
    semantic_checkpoint = (
        args.official_sam2matting_checkpoint
        if args.mam2_matter_backend == "official_sam2matting"
        else args.sam2_checkpoint
    )
    semantic_architecture = {
        "official_sam2matting": "official-sam2matting-eccv2026-flow-v1",
        "official_mam2": "official-mam2-iclr2026-flow-v1",
    }.get(args.mam2_matter_backend, PromptableDualModeDecoder.ARCHITECTURE)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "seed": args.seed,
        "prompt_mode": args.prompt_mode,
        "checkpoint_sha256": _sha256_path(args.checkpoint),
        "sam2_checkpoint_sha256": _sha256_path(semantic_checkpoint),
        "mam2_architecture": semantic_architecture,
        "mam2_matter_backend": args.mam2_matter_backend,
        "official_sam2matting_checkpoint_sha256": _sha256_path(args.official_sam2matting_checkpoint),
        "official_sam2matting_root": (
            None if args.official_sam2matting_root is None
            else str(args.official_sam2matting_root)
        ),
        "mematte_train_backbone": args.mematte_train_backbone,
        "mematte_train_decoder": args.mematte_train_decoder,
        "mematte_root": None if args.mematte_root is None else str(args.mematte_root),
        "mematte_config": (
            None if args.mematte_config is None else str(args.mematte_config)
        ),
        "mematte_checkpoint_sha256": _sha256_path(mematte_checkpoint),
        "dataset_artifact_manifest_sha256": _sha256_path(manifest),
        "diffusion_model": args.diffusion_model,
        "diffusion_revision": args.diffusion_revision,
        "diffusion_backend": args.diffusion_backend,
        "objectclear_dtype": args.objectclear_dtype,
        "objectclear_fused_output": args.objectclear_fused_output,
        "objectclear_late_denoise_steps": args.objectclear_late_denoise_steps,
        "objectclear_pixel_refinement": args.objectclear_pixel_refinement,
        "diffusion_local_sha256": _sha256_path(local_diffusion),
        "diffusion_adapter": args.diffusion_adapter,
        "diffusion_adapter_local_sha256": _sha256_path(local_adapter),
    }


def _wandb_config(args: argparse.Namespace) -> dict[str, object]:
    """Recursively convert CLI values such as lists of Paths for W&B config."""

    def convert(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [convert(item) for item in value]
        return value

    return {key: convert(value) for key, value in vars(args).items()}


def _preview_rgb(value: Tensor):
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise ImportError("qualitative output requires numpy and Pillow") from exc
    value = value.detach().float().clamp(0, 1)
    value = torch.where(
        value <= 0.0031308,
        12.92 * value,
        1.055 * value.pow(1.0 / 2.4) - 0.055,
    )
    array = value.permute(1, 2, 0).mul(255).round().byte().cpu().numpy()
    return Image.fromarray(np.asarray(array), mode="RGB")


def _preview_straight_foreground(value: Tensor):
    """Compress HDR straight-F values consistently for paired W&B panels."""

    nonnegative = value.clamp_min(0)
    return _preview_rgb(nonnegative / (1.0 + nonnegative))


def _preview_mask(value: Tensor):
    value = value.detach().float().clamp(0, 1)
    if value.ndim == 3:
        value = value[0]
    return _preview_rgb(value.unsqueeze(0).expand(3, -1, -1))


def _preview_transmittance(value: Tensor):
    """Preserve colored tau; tolerate legacy single-channel fixtures."""

    if value.shape[0] == 1:
        value = value.expand(3, -1, -1)
    return _preview_rgb(value)


def _preview_mask_overlay(image: Tensor, mask: Tensor):
    """Draw a high-contrast GT support contour over a linear-RGB frame."""

    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise ImportError("qualitative output requires numpy and Pillow") from exc
    if mask.ndim == 3:
        mask = mask[0]
    support = (mask.detach().float() >= 0.5)[None, None]
    dilated = F.max_pool2d(support.float(), kernel_size=5, stride=1, padding=2) > 0
    eroded = (
        1.0 - F.max_pool2d((~support).float(), kernel_size=5, stride=1, padding=2) > 0.5
    )
    boundary = (dilated & ~eroded)[0, 0].cpu().numpy()
    array = np.asarray(_preview_rgb(image)).copy()
    array[boundary] = np.asarray((255, 48, 48), dtype=np.uint8)
    return Image.fromarray(array, mode="RGB")


def _preview_temporal_gt(frames: Tensor, masks: Tensor, *, limit: int = 4):
    """Show up to four aligned frame/GT-mask pairs in one square panel."""

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:  # pragma: no cover
        raise ImportError("qualitative output requires Pillow") from exc
    count = min(int(frames.shape[0]), int(masks.shape[0]), limit)
    if count < 1:
        raise ValueError("temporal preview requires at least one frame")
    width, height = _preview_rgb(frames[0]).size
    cell_width = max(width // count, 1)
    cell_height = max(height // 2, 1)
    canvas = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    resampling = getattr(Image, "Resampling", Image)
    for frame_index in range(count):
        x0 = frame_index * cell_width
        x1 = width if frame_index == count - 1 else (frame_index + 1) * cell_width
        size = (x1 - x0, cell_height)
        overlay = _preview_mask_overlay(frames[frame_index], masks[frame_index]).resize(
            size, resample=resampling.BILINEAR
        )
        mask = _preview_mask(masks[frame_index]).resize(
            size, resample=resampling.NEAREST
        )
        canvas.paste(overlay, (x0, 0))
        canvas.paste(mask, (x0, cell_height))
        draw.text((x0 + 2, 2), f"t{frame_index}", fill="white", font=font)
    return canvas


def _alpha_gt_values(
    target: RefractiveGroundTruth,
    *,
    max_values: int | None = 65_536,
) -> Tensor | None:
    """Return a deterministic bounded sample of alpha values inside GT support."""

    if target.alpha is None:
        return None
    alpha = target.alpha.detach().float().clamp(0, 1)
    support = (
        target.object_mask.detach() >= 0.5
        if target.object_mask is not None
        else torch.ones_like(alpha, dtype=torch.bool)
    )
    values = alpha[support].reshape(-1)
    if values.numel() == 0:
        return values
    if max_values is not None and values.numel() > max_values:
        stride = math.ceil(values.numel() / max_values)
        values = values[::stride][:max_values]
    return values


def _alpha_gt_statistics(target: RefractiveGroundTruth) -> dict[str, Tensor]:
    """Summarize GT opacity inside the object support for W&B monitoring."""

    if target.alpha is None:
        return {}
    alpha = target.alpha.detach().float().clamp(0, 1)
    support = (
        target.object_mask.detach() >= 0.5
        if target.object_mask is not None
        else torch.ones_like(alpha, dtype=torch.bool)
    )
    support_count = support.sum()
    if int(support_count) == 0:
        return {}
    denominator = support_count.to(dtype=alpha.dtype)
    statistics = {
        "alpha_gt_mean": (alpha * support).sum() / denominator,
        "alpha_gt_near_clear_fraction": (support & (alpha <= 0.05))
        .sum()
        .to(alpha.dtype)
        / denominator,
        "alpha_gt_translucent_fraction": (support & (alpha > 0.05) & (alpha < 0.95))
        .sum()
        .to(alpha.dtype)
        / denominator,
        "alpha_gt_near_opaque_fraction": (support & (alpha >= 0.95))
        .sum()
        .to(alpha.dtype)
        / denominator,
    }
    if target.object_mask is not None:
        statistics["object_mask_coverage"] = (
            (target.object_mask.detach() >= 0.5).float().mean()
        )
    return statistics


def _preview_trimap(value: Tensor):
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise ImportError("qualitative output requires numpy and Pillow") from exc
    if value.ndim == 3:
        value = value[0]
    classes = value.detach().long().clamp(0, 2).cpu().numpy()
    palette = np.asarray(
        ((32, 64, 180), (235, 170, 35), (235, 235, 235)),
        dtype=np.uint8,
    )
    return Image.fromarray(palette[classes], mode="RGB")


def _preview_flow(value: Tensor, *, scale: Tensor | None = None):
    """Encode dx, dy, and magnitude for a dependency-free W&B preview."""

    if value.ndim != 3 or value.shape[0] != 2:
        raise ValueError("flow preview requires [2,H,W]")
    flow = value.detach().float()
    magnitude = flow.square().sum(dim=0).sqrt()
    if scale is None:
        scale = torch.quantile(magnitude.flatten(), 0.95).clamp_min(1e-6)
    dx = (0.5 + 0.5 * flow[0] / scale).clamp(0, 1)
    dy = (0.5 + 0.5 * flow[1] / scale).clamp(0, 1)
    return _preview_rgb(torch.stack((dx, dy, (magnitude / scale).clamp(0, 1))))


def _preview_signed_rgb(value: Tensor, *, scale: Tensor):
    """Use the same signed range for a prediction and its residual GT."""

    return _preview_rgb((0.5 + 0.5 * value.detach().float() / scale).clamp(0, 1))


def _preview_normal(value: Tensor):
    if value.ndim != 3 or value.shape[0] != 3:
        raise ValueError("normal preview requires [3,H,W]")
    normal = F.normalize(value.detach().float(), dim=0, eps=1e-6)
    return _preview_rgb((normal + 1.0) * 0.5)


def _preview_depth(value: Tensor, validity: Tensor | None = None):
    if value.ndim == 3:
        value = value[0]
    depth = value.detach().float()
    valid = torch.isfinite(depth) & (depth > 0)
    if validity is not None:
        if validity.ndim == 3:
            validity = validity[0]
        valid = valid & (validity.detach() >= 0.5)
    display = torch.zeros_like(depth)
    if bool(valid.any()):
        valid_depth = depth[valid]
        low = torch.quantile(valid_depth, 0.02)
        high = torch.quantile(valid_depth, 0.98)
        display[valid] = ((depth[valid] - low) / (high - low).clamp_min(1e-6)).clamp(
            0, 1
        )
    return _preview_rgb(torch.stack((display, display, display)))


def _labeled_grid(panels, *, columns: int = 4, min_cell_width: int = 0):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:  # pragma: no cover
        raise ImportError("qualitative output requires Pillow") from exc
    if not panels:
        raise ValueError("at least one panel is required")
    width, height = panels[0][1].size
    cell_width = max(width, min_cell_width)
    title_height = 20
    rows = math.ceil(len(panels) / columns)
    montage = Image.new(
        "RGB", (columns * cell_width, rows * (height + title_height)), "black"
    )
    draw = ImageDraw.Draw(montage)
    font = ImageFont.load_default()
    for index, (label, panel) in enumerate(panels):
        if panel.size != (width, height):
            panel = panel.resize((width, height))
        x = (index % columns) * cell_width
        y = (index // columns) * (height + title_height)
        montage.paste(panel, (x + (cell_width - width) // 2, y + title_height))
        draw.text((x + 4, y + 4), label, fill="white", font=font)
    return montage


def _prediction_montages(
    prediction: RefractiveMAM2Output,
    target: RefractiveGroundTruth,
    sample_ids: list[str],
    *,
    limit: int,
):
    count = min(limit, prediction.background.background.shape[0])
    output = []
    for index in range(count):
        background_gt = target.counterfactual_background
        if background_gt is not None:
            background_gt = background_gt[index]
            if background_gt.ndim == 4:
                background_gt = background_gt[0]
        error = (
            (prediction.reconstructed_frames[index, 0] - target.frames[index, 0])
            .abs()
            .mul(4.0)
            .clamp(0, 1)
        )
        panels = [
            ("input I", _preview_rgb(target.frames[index, 0])),
            ("reconstruction", _preview_rgb(prediction.reconstructed_frames[index, 0])),
            ("|I-I_hat| x4", _preview_rgb(error)),
            ("background pred", _preview_rgb(prediction.background.background[index])),
            (
                "evidence bg",
                _preview_rgb(prediction.background.evidence_background[index]),
            ),
            ("MAM2 alpha", _preview_mask(prediction.backbone.alpha_matte[index, 0])),
            ("alpha pred", _preview_mask(prediction.matter.alpha[index, 0])),
            (
                "direct support",
                _preview_mask(prediction.background.direct_support[index]),
            ),
            (
                "inverse support",
                _preview_mask(prediction.background.inverse_support[index]),
            ),
            ("true hole", _preview_mask(prediction.background.true_hole[index])),
            ("flow pred", _preview_flow(prediction.matter.refractive_flow[index, 0])),
        ]
        coarse_flow = getattr(prediction.matter, "coarse_refractive_flow", None)
        if coarse_flow is not None:
            panels.append(
                (
                    "flow coarse",
                    _preview_flow(coarse_flow[index, 0]),
                )
            )
        if target.object_mask is not None:
            panels.insert(
                1,
                (
                    "input + mask GT",
                    _preview_mask_overlay(
                        target.frames[index, 0], target.object_mask[index, 0]
                    ),
                ),
            )
            panels.insert(
                2,
                (
                    "temporal input/mask GT",
                    _preview_temporal_gt(
                        target.frames[index], target.object_mask[index]
                    ),
                ),
            )
        if background_gt is not None:
            panels.insert(5, ("background GT", _preview_rgb(background_gt)))
        if target.alpha is not None:
            panels.insert(7, ("alpha GT", _preview_mask(target.alpha[index, 0])))
        if prediction.matter.surface_normal is not None:
            panels.append(
                (
                    "normal pred",
                    _preview_normal(prediction.matter.surface_normal[index, 0]),
                )
            )
        if target.surface_normal is not None:
            panels.append(
                ("normal GT", _preview_normal(target.surface_normal[index, 0]))
            )
        if prediction.matter.depth is not None:
            panels.append(
                (
                    "depth pred",
                    _preview_depth(
                        prediction.matter.depth[index, 0],
                        None
                        if target.depth_validity is None
                        else target.depth_validity[index, 0],
                    ),
                )
            )
        if target.depth is not None:
            panels.append(
                (
                    "depth GT",
                    _preview_depth(
                        target.depth[index, 0],
                        None
                        if target.depth_validity is None
                        else target.depth_validity[index, 0],
                    ),
                )
            )
        if prediction.matter.geometry_confidence is not None:
            panels.append(
                (
                    "geometry confidence",
                    _preview_mask(prediction.matter.geometry_confidence[index, 0]),
                )
            )
        caption = sample_ids[index] if index < len(sample_ids) else f"sample_{index}"
        output.append((_labeled_grid(panels), caption))
    return output


def _supervised_prediction_montages(
    prediction: RefractiveMAM2Output,
    target: RefractiveGroundTruth,
    sample_ids: list[str],
    *,
    limit: int,
    stage: str,
    teacher_forcing: bool = False,
):
    """W&B panels for the outputs that can actually learn in this stage."""

    if stage not in {"2", "3", "4"}:
        raise ValueError(f"physics montage requires Stage 2, 3, or 4, got {stage!r}")
    count = min(limit, prediction.reconstructed_frames.shape[0])
    output = []
    color_validity = (
        _identifiable_color_transmission(target)
        if target.color_transmission is not None else None
    )
    if color_validity is not None and not isinstance(color_validity, Tensor):
        color_validity = torch.ones_like(target.color_transmission[:, :, :1])
    past_evidence = _past_visible_refracted_background(target)
    for index in range(count):
        frame = target.frames[index, 0]
        rendered = prediction.reconstructed_frames[index, 0]
        frame_count = target.frames.shape[1]
        panels = [
            (f"input I t0 (T{frame_count})", _preview_rgb(frame)),
            ("recomposed I pred", _preview_rgb(rendered)),
            ("render error x4", _preview_rgb(((rendered - frame).abs() * 4).clamp(0, 1))),
        ]

        def pair(name: str, predicted: Tensor, truth: Tensor | None, preview):
            if truth is not None:
                panels.extend(
                    ((f"{name} pred", preview(predicted)),
                     (f"{name} GT", preview(truth)))
                )

        size = target.frames.shape[-2:]
        corrected_support_logits = getattr(
            prediction.matter, "object_support_logits", None
        )
        foreground_candidate = getattr(
            prediction.matter, "foreground_candidate", None
        )
        if target.object_mask is not None:
            frozen_mask = F.interpolate(
                prediction.backbone.mask_logits[index, 0][None],
                size=size, mode="bilinear", align_corners=False,
            ).sigmoid()[0]
            mask = (
                corrected_support_logits[index, 0].sigmoid()
                if corrected_support_logits is not None
                else frozen_mask
            )
            pair(
                "object support", (mask >= 0.5).to(mask.dtype),
                target.object_mask[index, 0], _preview_mask,
            )
        if stage == "4":
            if target.object_mask is not None:
                mask_name = "SAM2 mask" if corrected_support_logits is not None else "mask"
                pair(mask_name, frozen_mask, target.object_mask[index, 0], _preview_mask)
            if target.trimap is not None:
                trimap = F.interpolate(
                    prediction.backbone.trimap_logits[index, 0][None],
                    size=size, mode="bilinear", align_corners=False,
                ).argmax(dim=1)[0]
                pair("trimap", trimap, target.trimap[index, 0], _preview_trimap)
            if target.alpha is not None:
                mam2_alpha = F.interpolate(
                    prediction.backbone.alpha_matte[index, 0][None],
                    size=size, mode="bilinear", align_corners=False,
                )[0]
                pair(
                    "MAM2 alpha", mam2_alpha,
                    target.alpha[index, 0], _preview_mask,
                )

        pair(
            "PAM alpha", prediction.matter.alpha[index, 0],
            None if target.alpha is None else target.alpha[index, 0], _preview_mask,
        )
        foreground_support = _straight_foreground_support(target)
        if target.straight_foreground is not None and foreground_support is not None:
            valid = foreground_support[index, 0]
            panels.append(("foreground F GT-alpha valid", _preview_mask(valid)))
            pair(
                "foreground F (valid, tone-mapped)",
                prediction.matter.straight_foreground[index, 0] * valid,
                target.straight_foreground[index, 0] * valid,
                _preview_straight_foreground,
            )
            if foreground_candidate is not None:
                pair(
                    "foreground F candidate (GT-valid)",
                    foreground_candidate[index, 0] * valid,
                    target.straight_foreground[index, 0] * valid,
                    _preview_straight_foreground,
                )
        foreground_gt = target.premultiplied_foreground
        if foreground_gt is None and target.straight_foreground is not None and target.alpha is not None:
            foreground_gt = target.straight_foreground * target.alpha
        pair(
            "foreground G", prediction.matter.premultiplied_foreground[index, 0],
            None if foreground_gt is None else foreground_gt[index, 0], _preview_rgb,
        )
        if target.color_transmission is not None:
            valid = color_validity[index, 0]
            panels.append(("color T identifiable", _preview_mask(valid)))
            pair(
                "color T", prediction.matter.color_transmission[index, 0] * valid,
                target.color_transmission[index, 0] * valid,
                _preview_rgb,
            )
        pair(
            "transmittance tau", prediction.matter.transmittance[index, 0],
            None if target.transmittance is None else target.transmittance[index, 0],
            _preview_transmittance,
        )
        if past_evidence is not None:
            past_background, past_support = past_evidence
            source_counts = past_support[index].sum(dim=(1, 2, 3))
            source_time = int(source_counts.argmax().item())
            if source_counts[source_time] > 0:
                panels.append((
                    f"past-visible source support t{source_time}",
                    _preview_mask(past_support[index, source_time]),
                ))
                panels.append((
                    f"past-visible B at Phi t{source_time}",
                    _preview_rgb(
                        past_background[index, source_time]
                        * past_support[index, source_time]
                    ),
                ))
        if target.residual is not None:
            residual_pred = prediction.matter.residual[index, 0]
            residual_gt = target.residual[index, 0]
            residual_scale = torch.maximum(
                torch.quantile(residual_pred.detach().float().abs().flatten(), 0.98),
                torch.quantile(residual_gt.detach().float().abs().flatten(), 0.98),
            ).clamp_min(1e-6)
            pair(
                "residual R", residual_pred, residual_gt,
                lambda value: _preview_signed_rgb(value, scale=residual_scale),
            )
        if target.refractive_flow is not None:
            flow_pred = prediction.matter.refractive_flow[index, 0]
            flow_gt = target.refractive_flow[index, 0]
            flow_scale = torch.maximum(
                torch.quantile(flow_pred.detach().float().norm(dim=0).flatten(), 0.95),
                torch.quantile(flow_gt.detach().float().norm(dim=0).flatten(), 0.95),
            ).clamp_min(1e-6)
            # Phi=x+u: one displacement image represents both supervised terms.
            pair(
                "flow u / Phi", flow_pred, flow_gt,
                lambda value: _preview_flow(value, scale=flow_scale),
            )
            coarse_flow = getattr(prediction.matter, "coarse_refractive_flow", None)
            if coarse_flow is not None:
                pair(
                    "flow coarse / Phi",
                    coarse_flow[index, 0],
                    flow_gt,
                    lambda value: _preview_flow(value, scale=flow_scale),
                )
        pair(
            "confidence", prediction.matter.confidence[index, 0],
            None if target.confidence is None else target.confidence[index, 0],
            _preview_mask,
        )

        background_gt = target.counterfactual_background
        if background_gt is not None:
            background_gt = background_gt[index]
            if background_gt.ndim == 4:
                background_gt = background_gt[0]
            if stage == "2":
                panels.append(("oracle B input", _preview_rgb(background_gt)))
            else:
                if teacher_forcing:
                    panels.append(("oracle B render input", _preview_rgb(background_gt)))
                pair(
                    "background B", prediction.background.background[index],
                    background_gt, _preview_rgb,
                )
                for name, estimate, support in (
                    ("direct B", prediction.background.observed_background[index],
                     prediction.background.direct_support[index]),
                    ("inverse B", prediction.background.inverse_background[index],
                     prediction.background.inverse_support[index]),
                    ("hole B", prediction.background.background[index],
                     prediction.background.true_hole[index]),
                ):
                    panels.append((f"{name} support", _preview_mask(support)))
                    if bool(support.any()):
                        pair(name, estimate * support, background_gt * support, _preview_rgb)

        if prediction.matter.surface_normal is not None and target.surface_normal is not None:
            pair(
                "normal", prediction.matter.surface_normal[index, 0],
                target.surface_normal[index, 0], _preview_normal,
            )
        if prediction.matter.depth is not None and target.depth is not None:
            validity = None if target.depth_validity is None else target.depth_validity[index, 0]
            pair(
                "depth", prediction.matter.depth[index, 0], target.depth[index, 0],
                lambda value: _preview_depth(value, validity),
            )
        caption = sample_ids[index] if index < len(sample_ids) else f"sample_{index}"
        if stage == "2" or teacher_forcing:
            caption += " | recomposition uses oracle background"
        output.append((_labeled_grid(panels, min_cell_width=160), caption))
    return output


def _semantic_montages(
    semantics: MAM2BackboneOutput,
    target: RefractiveGroundTruth,
    sample_ids: list[str],
    *,
    limit: int,
    stage: str,
):
    """Show only the semantic outputs supervised in this Stage-1 phase."""

    if stage not in {"1a", "1b"}:
        raise ValueError(f"semantic montage requires Stage 1A or 1B, got {stage!r}")
    count = min(limit, semantics.mask_logits.shape[0])
    size = target.frames.shape[-2:]
    if stage == "1a":
        masks = F.interpolate(
            semantics.mask_logits[:, 0], size=size, mode="bilinear", align_corners=False
        ).sigmoid()
        trimaps = F.interpolate(
            semantics.trimap_logits[:, 0], size=size, mode="bilinear", align_corners=False
        ).argmax(dim=1)
    else:
        alpha = F.interpolate(
            semantics.alpha_matte[:, 0], size=size, mode="bilinear", align_corners=False
        )
    output = []
    for index in range(count):
        panels = [("input I", _preview_rgb(target.frames[index, 0]))]
        if stage == "1a":
            panels.append(("mask pred", _preview_mask(masks[index])))
            if target.object_mask is not None:
                panels.append(("mask GT", _preview_mask(target.object_mask[index, 0])))
            panels.append(("trimap pred", _preview_trimap(trimaps[index])))
            if target.trimap is not None:
                panels.append(("trimap GT", _preview_trimap(target.trimap[index, 0])))
        else:
            panels.append(("alpha pred", _preview_mask(alpha[index])))
            if target.alpha is not None:
                panels.append(("alpha GT", _preview_mask(target.alpha[index, 0])))
        caption = sample_ids[index] if index < len(sample_ids) else f"sample_{index}"
        output.append((_labeled_grid(panels, min_cell_width=140), caption))
    return output


def _save_qualitative(
    prediction: RefractiveMAM2Output,
    target: RefractiveGroundTruth,
    output_dir: Path,
    sample_ids: list[str],
    *,
    remaining: int,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = min(remaining, prediction.background.background.shape[0])
    montages = _prediction_montages(
        prediction,
        target,
        sample_ids,
        limit=count,
    )
    for index, (montage, caption) in enumerate(montages):
        safe_id = "_".join(Path(caption).parts)[-120:]
        sample_dir = output_dir / safe_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        montage.save(sample_dir / "input_reconstruction_bgpred_bggt_alpha.png")
        saved_prediction = {
                "background": prediction.background.background[index].detach().cpu(),
                "evidence_background": prediction.background.evidence_background[index]
                .detach()
                .cpu(),
                "direct_support": prediction.background.direct_support[index]
                .detach()
                .cpu(),
                "inverse_support": prediction.background.inverse_support[index]
                .detach()
                .cpu(),
                "true_hole": prediction.background.true_hole[index].detach().cpu(),
                "mam2_alpha": prediction.backbone.alpha_matte[index].detach().cpu(),
                "alpha": prediction.matter.alpha[index].detach().cpu(),
                "straight_foreground": prediction.matter.straight_foreground[
                    index
                ].detach().cpu(),
                "premultiplied_foreground": prediction.matter.premultiplied_foreground[
                    index
                ]
                .detach()
                .cpu(),
                "transmittance": prediction.matter.transmittance[index].detach().cpu(),
                "refractive_flow": prediction.matter.refractive_flow[index]
                .detach()
                .cpu(),
                "residual": prediction.matter.residual[index].detach().cpu(),
                "reconstruction": prediction.reconstructed_frames[index].detach().cpu(),
            }
        coarse_flow = getattr(prediction.matter, "coarse_refractive_flow", None)
        if coarse_flow is not None:
            saved_prediction["coarse_refractive_flow"] = (
                coarse_flow[index].detach().cpu()
            )
        for name in ("surface_normal", "depth", "geometry_confidence"):
            value = getattr(prediction.matter, name)
            if value is not None:
                saved_prediction[name] = value[index].detach().cpu()
        torch.save(saved_prediction, sample_dir / "prediction.pt")
    return count


def _semantic_forward(
    predictor: MAM2VideoPredictor,
    target: RefractiveGroundTruth,
    *,
    prompt_mode: str,
    prompt_seed: int = 0,
    prompt_jitter_pixels: float = 0.0,
    frames_are_linear: bool = True,
    compute_alpha: bool = True,
) -> MAM2BackboneOutput:
    """Run differentiable SAM2/PDD/MSS with a reproducible prompt protocol."""

    if target.object_mask is None:
        raise ValueError(
            "RCTrans training requires object_mask for the first-frame prompt"
        )
    size = (predictor.image_size, predictor.image_size)
    batch, frames, _, height, width = target.frames.shape
    semantic_frames = (
        linear_to_srgb(target.frames)
        if frames_are_linear
        else target.frames.clamp(0.0, 1.0)
    )
    resized = F.interpolate(
        semantic_frames.reshape(batch * frames, 3, height, width),
        size=size,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).reshape(batch, frames, 3, *size)
    normalized = normalize_sam2_training_frames(resized.clamp(0.0, 1.0))
    first_mask = target.object_mask[:, 0]
    official_kwargs = (
        {"original_size": (height, width)}
        if getattr(predictor, "official_mam2", False)
        else {}
    )
    if prompt_mode == "mask":
        return predictor.forward_mam2_clip(
            normalized,
            first_frame_mask_inputs=first_mask,
            compute_alpha=compute_alpha,
            **official_kwargs,
        )
    point_coords, point_labels = _prompt_points_from_mask(
        first_mask,
        size=predictor.image_size,
        mode=prompt_mode,
        seed=prompt_seed,
        jitter_pixels=prompt_jitter_pixels,
    )
    return predictor.forward_mam2_clip(
        normalized,
        first_frame_point_inputs={
            "point_coords": point_coords,
            "point_labels": point_labels,
        },
        compute_alpha=compute_alpha,
        **official_kwargs,
    )


def _prompt_points_from_mask(
    mask: Tensor,
    *,
    size: int,
    mode: str,
    seed: int = 0,
    jitter_pixels: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Create a deterministic positive point or SAM box-corner prompt."""

    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("prompt mask must have shape [B,1,H,W]")
    if mode not in ("point", "box"):
        raise ValueError("prompt mode must be point, box, or mask")
    batch, _, height, width = mask.shape
    coordinates: list[Tensor] = []
    labels: list[Tensor] = []
    for batch_index in range(batch):
        foreground = torch.nonzero(mask[batch_index, 0] >= 0.5, as_tuple=False)
        if foreground.numel() == 0:
            raise ValueError("cannot create a prompt from an empty first-frame mask")
        if mode == "point":
            center = foreground.float().mean(dim=0)
            nearest = torch.argmin((foreground.float() - center).square().sum(dim=1))
            selected = nearest
            if jitter_pixels > 0:
                distance = torch.linalg.vector_norm(
                    foreground.float() - foreground[nearest].float(),
                    dim=1,
                )
                candidates = torch.nonzero(
                    distance <= jitter_pixels,
                    as_tuple=False,
                ).flatten()
                rng = random.Random(seed + batch_index * 1_000_003)
                selected = candidates[rng.randrange(candidates.numel())]
            yx = foreground[selected].to(mask.dtype)
            xy = torch.stack((yx[1], yx[0]))
            coordinates.append(xy[None])
            labels.append(torch.ones(1, dtype=torch.int32, device=mask.device))
        else:
            minimum = foreground.amin(dim=0).to(mask.dtype)
            maximum = foreground.amax(dim=0).to(mask.dtype)
            if jitter_pixels > 0:
                rng = random.Random(seed + batch_index * 1_000_003)
                offsets = torch.tensor(
                    [rng.uniform(-jitter_pixels, jitter_pixels) for _ in range(4)],
                    device=mask.device,
                    dtype=mask.dtype,
                )
                minimum = minimum + offsets[:2]
                maximum = maximum + offsets[2:]
                minimum[0].clamp_(0, height - 1)
                minimum[1].clamp_(0, width - 1)
                maximum[0].clamp_(minimum[0], height - 1)
                maximum[1].clamp_(minimum[1], width - 1)
            coordinates.append(
                torch.stack(
                    (
                        torch.stack((minimum[1], minimum[0])),
                        torch.stack((maximum[1], maximum[0])),
                    )
                )
            )
            labels.append(torch.tensor((2, 3), dtype=torch.int32, device=mask.device))
    coords = torch.stack(coordinates).to(device=mask.device, dtype=torch.float32)
    scale = coords.new_tensor((size / max(width, 1), size / max(height, 1))).view(
        1, 1, 2
    )
    return coords * scale, torch.stack(labels)


def _predict(
    predictor: MAM2VideoPredictor,
    pipeline: RefractiveMAM2,
    target: RefractiveGroundTruth,
    *,
    teacher_forcing: bool,
    prompt_mode: str,
    prompt_seed: int = 0,
    prompt_jitter_pixels: float = 0.0,
    skip_background_estimation: bool = False,
) -> RefractiveMAM2Output:
    semantics = _semantic_forward(
        predictor,
        target,
        prompt_mode=prompt_mode,
        prompt_seed=prompt_seed,
        prompt_jitter_pixels=prompt_jitter_pixels,
    )
    return pipeline.forward_from_backbone(
        target.frames,
        semantics,
        counterfactual_background_gt=target.counterfactual_background,
        use_ground_truth_background=teacher_forcing,
        skip_background_estimation=skip_background_estimation,
    )


def _concatenate_predictions(
    predictions: list[RefractiveMAM2Output],
) -> RefractiveMAM2Output:
    """Join independently evaluated batch elements without changing losses.

    Stage 3/4 paired supervision needs both operator predictions in one logical
    batch. Keeping the forwards independent allows activation checkpointing to
    discard each sample's large SAM2/FFC/PAM intermediates before evaluating
    its partner; concatenating only the public outputs preserves the original
    batch-level loss and paired-consistency calculation.
    """

    if not predictions:
        raise ValueError("predictions must not be empty")

    def cat_fields(output_type, values):
        def concatenate(name):
            fields = [getattr(value, name) for value in values]
            if fields[0] is None:
                if any(field is not None for field in fields[1:]):
                    raise ValueError(f"inconsistent optional output field: {name}")
                return None
            if any(field is None for field in fields[1:]):
                raise ValueError(f"inconsistent optional output field: {name}")
            return torch.cat(fields, dim=0)

        return output_type(
            **{name: concatenate(name) for name in output_type.__dataclass_fields__}
        )

    return RefractiveMAM2Output(
        backbone=cat_fields(
            MAM2BackboneOutput,
            [prediction.backbone for prediction in predictions],
        ),
        background=cat_fields(
            BackgroundOutput,
            [prediction.background for prediction in predictions],
        ),
        matter=cat_fields(
            PhysicsMatterOutput,
            [prediction.matter for prediction in predictions],
        ),
        reconstructed_frames=torch.cat(
            [prediction.reconstructed_frames for prediction in predictions], dim=0
        ),
        refracted_background=torch.cat(
            [prediction.refracted_background for prediction in predictions], dim=0
        ),
    )


def _checkpointed_paired_predict(
    predictor: MAM2VideoPredictor,
    pipeline: RefractiveMAM2,
    target: RefractiveGroundTruth,
    *,
    teacher_forcing: bool,
    prompt_mode: str,
    prompt_seed: int = 0,
    prompt_jitter_pixels: float = 0.0,
    skip_background_estimation: bool = False,
) -> RefractiveMAM2Output:
    """Evaluate a paired batch one sample at a time with exact joint losses.

    The returned tensors are concatenated into the same shape as a conventional
    batch forward. Therefore ordinary losses and the cross-background
    operator-reuse loss remain unchanged. During backward, PyTorch recomputes
    each sample's forward instead of retaining both full activation graphs.
    """

    if target.object_mask is None:
        raise ValueError("paired training requires object_mask prompts")
    if teacher_forcing and target.counterfactual_background is None:
        raise ValueError("teacher forcing requires counterfactual_background")

    predictions: list[RefractiveMAM2Output] = []
    for index in range(target.frames.shape[0]):
        frames = target.frames[index : index + 1]
        object_mask = target.object_mask[index : index + 1]
        background = (
            None
            if target.counterfactual_background is None
            else target.counterfactual_background[index : index + 1]
        )
        # _prompt_points_from_mask uses this same offset for elements of a
        # conventional batch. Preserve it when every microbatch has index 0.
        sample_prompt_seed = prompt_seed + index * 1_000_003

        def forward_sample(
            sample_frames: Tensor,
            sample_mask: Tensor,
            sample_background: Tensor | None = background,
            seed: int = sample_prompt_seed,
        ) -> RefractiveMAM2Output:
            sample_target = RefractiveGroundTruth(
                frames=sample_frames,
                object_mask=sample_mask,
                counterfactual_background=sample_background,
            )
            return _predict(
                predictor,
                pipeline,
                sample_target,
                teacher_forcing=teacher_forcing,
                prompt_mode=prompt_mode,
                prompt_seed=seed,
                prompt_jitter_pixels=prompt_jitter_pixels,
                skip_background_estimation=skip_background_estimation,
            )

        # Non-reentrant checkpointing supports the nested dataclass output and
        # parameter gradients even though image inputs themselves need no grad.
        predictions.append(
            checkpoint(
                forward_sample,
                frames,
                object_mask,
                use_reentrant=False,
            )
        )
    return _concatenate_predictions(predictions)


def _semantic_loss(
    semantics: MAM2BackboneOutput,
    target: RefractiveGroundTruth,
    dataset_kinds: list[str] | None = None,
    *,
    stage: str = "1a",
) -> dict[str, Tensor]:
    alpha_target = None if stage == "1a" else target.alpha
    alpha_validity = None if stage == "1a" else target.alpha_validity
    if dataset_kinds is not None:
        if len(dataset_kinds) != semantics.mask_logits.shape[0]:
            raise ValueError("dataset kind count must match semantic batch size")
        terms: dict[str, Tensor] = {}
        for kind in sorted(set(dataset_kinds)):
            indices = torch.tensor(
                [index for index, value in enumerate(dataset_kinds) if value == kind],
                device=semantics.mask_logits.device,
            )
            selected = MAM2BackboneOutput(
                mask_logits=semantics.mask_logits.index_select(0, indices),
                trimap_logits=semantics.trimap_logits.index_select(0, indices),
                alpha_matte=semantics.alpha_matte.index_select(0, indices),
                non_memory_features=semantics.non_memory_features.index_select(
                    0, indices
                ),
            )
            selected_target = SemanticTargets(
                object_mask=(
                    None
                    if target.object_mask is None
                    else target.object_mask.index_select(0, indices)
                ),
                trimap=(
                    None
                    if target.trimap is None
                    else target.trimap.index_select(0, indices)
                ),
                alpha=(
                    None
                    if alpha_target is None
                    else alpha_target.index_select(0, indices)
                ),
                alpha_validity=(
                    None
                    if alpha_validity is None
                    else alpha_validity.index_select(0, indices)
                ),
            )
            selected_terms = selective_semantic_loss(
                selected.mask_logits,
                selected.trimap_logits,
                selected.alpha_matte,
                selected_target,
                dataset_kind=kind,
            )
            for name, value in selected_terms.items():
                if name != "total":
                    terms[name] = terms.get(name, value.new_zeros(())) + value
    else:
        terms = selective_semantic_loss(
            semantics.mask_logits,
            semantics.trimap_logits,
            semantics.alpha_matte,
            SemanticTargets(
                object_mask=target.object_mask,
                trimap=target.trimap,
                alpha=alpha_target,
                alpha_validity=alpha_validity,
            ),
            dataset_kind="synthetic_physics",
        )
    allowed = (
        {"mask", "trimap"}
        if stage == "1a"
        else {
            "mam2_alpha_unknown_l1",
            "mam2_alpha_known_l1",
            "mam2_alpha_l2",
            "mam2_alpha_laplacian",
            "mam2_alpha_gradient",
        }
    )
    filtered = {name: value for name, value in terms.items() if name in allowed}
    if not filtered:
        required = "mask/trimap" if stage == "1a" else "alpha"
        raise ValueError(f"Stage {stage} batch contains no {required} supervision")
    filtered["total"] = sum(filtered.values(), semantics.mask_logits.new_zeros(()))
    return filtered


def _semantic_metrics(
    semantics: MAM2BackboneOutput,
    target: RefractiveGroundTruth,
    *,
    include_alpha: bool = True,
) -> dict[str, tuple[float, int]]:
    metrics: dict[str, tuple[float, int]] = {}
    if target.object_mask is not None:
        logits = F.interpolate(
            semantics.mask_logits.flatten(0, 1),
            size=target.object_mask.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).reshape_as(target.object_mask)
        prediction = logits >= 0
        truth = target.object_mask >= 0.5
        intersection = (prediction & truth).sum(dim=(2, 3, 4)).float()
        union = (prediction | truth).sum(dim=(2, 3, 4)).float()
        iou = torch.where(union > 0, intersection / union, torch.ones_like(union))
        metrics["mask_iou"] = (float(iou.sum().cpu()), iou.numel())
    if target.trimap is not None:
        batch, frames = target.trimap.shape[:2]
        logits = F.interpolate(
            semantics.trimap_logits.flatten(0, 1),
            size=target.trimap.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, frames, 3, *target.trimap.shape[-2:])
        prediction = logits.argmax(dim=2)
        truth = target.trimap.squeeze(2).long()
        class_f1 = []
        for class_index in range(3):
            predicted_class = prediction == class_index
            true_class = truth == class_index
            true_positive = (predicted_class & true_class).sum().float()
            denominator = predicted_class.sum() + true_class.sum()
            class_f1.append(
                torch.where(
                    denominator > 0,
                    2.0 * true_positive / denominator,
                    denominator.new_ones(()),
                )
            )
        macro_f1 = torch.stack(class_f1).mean()
        metrics["trimap_macro_f1"] = (float(macro_f1.cpu()), 1)
    if include_alpha and target.alpha is not None:
        alpha = F.interpolate(
            semantics.alpha_matte.flatten(0, 1),
            size=target.alpha.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).reshape_as(target.alpha)
        validity = (
            torch.ones_like(target.alpha)
            if target.alpha_validity is None
            else target.alpha_validity.to(alpha.dtype)
        )
        absolute = (alpha - target.alpha).abs() * validity
        pixels = validity.sum().clamp_min(1.0)
        metrics["mam2_alpha_mae"] = (float((absolute.sum() / pixels).cpu()), 1)
        # Standard matting SAD is the summed absolute error divided by 1000.
        metrics["mam2_alpha_sad"] = (float((absolute.sum() / 1000.0).cpu()), 1)
    return metrics


def _loss_terms(
    stage: str,
    prediction: RefractiveMAM2Output,
    batch: RCTransBatch,
    *,
    loss_profile: str = "standard",
    oracle_background_scale: float = 1.0,
) -> dict[str, Tensor]:
    target = batch.ground_truth
    paired_ids = batch.paired_background_group_ids
    has_pair = len(paired_ids) != len(set(paired_ids))
    support = None
    if has_pair and target.object_mask is not None:
        support = target.object_mask
        if target.refractive_validity is not None:
            support = support * target.refractive_validity
    if oracle_background_scale <= 0 or not math.isfinite(oracle_background_scale):
        raise ValueError("oracle_background_scale must be finite and positive")
    weights = _loss_weights_for_profile(loss_profile)
    if oracle_background_scale != 1.0:
        weights = replace(
            weights,
            render_oracle_background=(
                weights.render_oracle_background * oracle_background_scale
            ),
        )
    criterion = RefractiveLoss(weights)
    if stage == "2" and not has_pair:
        return physics_stage_loss(prediction, target, loss=criterion)
    return joint_stage_loss(
        prediction,
        target,
        paired_background_group_ids=paired_ids if has_pair else None,
        operator_support=support,
        loss=criterion,
    )


def _loss_weights_for_profile(profile: str) -> LossWeights:
    if profile == "standard":
        return LossWeights()
    if profile == "optical_priority":
        return replace(
            LossWeights(),
            # Pixel-unit flow loss otherwise dominates all RGB operator terms
            # at 64px resolution, while still retaining direct GT flow loss.
            refractive_flow=0.25,
            source_coordinates=0.05,
            color_transmission=2.5,
            color_transmission_chroma=1.0,
            transmittance=2.0,
            straight_foreground=1.0,
            straight_foreground_gradient=0.2,
            straight_foreground_laplacian=0.15,
            residual=1.0,
            residual_sparsity=0.15,
            residual_excess=0.75,
            render_object=3.0,
            render_oracle_background=2.0,
            temporal_visible_transmission=0.5,
            temporal_visible_joint_recomposition=0.5,
            temporal_holdout_render=0.5,
        )
    if profile == "joint_sharp":
        return replace(
            _loss_weights_for_profile("optical_priority"),
            render_object_gradient=0.4,
            refracted_background_gradient=0.2,
            background_true_hole_gradient=0.2,
        )
    if profile == "foreground_recovery":
        return replace(
            _loss_weights_for_profile("joint_sharp"),
            matter_object_support=3.0,
            foreground_candidate=1.5,
            straight_foreground=1.5,
            premultiplied_foreground=1.5,
            alpha=2.5,
        )
    if profile == "detail_recovery":
        return replace(
            _loss_weights_for_profile("foreground_recovery"),
            matter_object_support_recall=0.75,
            foreground_candidate_gradient=0.4,
            foreground_candidate_laplacian=0.2,
            straight_foreground_gradient=0.35,
            straight_foreground_laplacian=0.25,
            premultiplied_foreground_gradient=0.45,
            background_true_hole=3.0,
            background_true_hole_gradient=0.5,
            background_true_hole_laplacian=0.2,
            render_object_gradient=0.6,
        )
    if profile == "flow_recovery":
        return replace(
            _loss_weights_for_profile("detail_recovery"),
            # Retain the RGB/foreground objective; the gain term addresses
            # the measured failure of refined flow to beat coarse flow.
            flow_refinement_gain=0.3,
        )
    raise ValueError(f"unknown loss profile: {profile}")


def _default_checkpoint_selection(
    stage: str,
    *,
    evidence_flux_only: bool,
    learned_object_support: bool = False,
) -> tuple[str, str]:
    """Select the component this stage can actually improve.

    In the FLUX-only variant, Stage 3 freezes semantics and the background
    prior while training PAM. Selecting its checkpoint by background PSNR can
    therefore discard the strongest optical operator. An independently
    rendered GT background isolates PAM quality without an extra model pass;
    Stage 4 still selects by the full predicted-background object render.
    """

    if stage == "1a":
        return "trimap_macro_f1", "max"
    if stage == "1b":
        return "mam2_alpha_sad", "min"
    if stage == "2":
        return "render_object_psnr", "max"
    if stage == "3":
        return (
            "render_oracle_background_object_psnr"
            if evidence_flux_only else "background_occluded_psnr",
            "max",
        )
    if stage == "4":
        return (
            "render_support_aware_score" if learned_object_support
            else "render_object_psnr",
            "max",
        )
    raise ValueError(f"unknown training stage: {stage}")


def _render_support_aware_score(render_psnr: float, support_iou: float) -> float:
    """Reward reconstruction only when the predicted object extent stays valid."""

    if not math.isfinite(render_psnr) or not math.isfinite(support_iou):
        raise ValueError("render PSNR and support IoU must be finite")
    if not 0.0 <= support_iou <= 1.0:
        raise ValueError("support IoU must be in [0, 1]")
    return render_psnr + 20.0 * math.log10(max(support_iou, 1e-4))


def _stage4_oracle_background_scale(progress: float) -> float:
    """Bridge Stage-2 oracle B to joint FLUX B without skipping any FLUX pass."""

    return 1.0 + max(0.0, 1.0 - 2.0 * max(0.0, min(progress, 1.0)))


def _merge_oracle_component_metrics(
    metrics: dict[str, tuple[float, int]],
    prediction: RefractiveMAM2Output,
    target: RefractiveGroundTruth,
    object_support: Tensor,
) -> None:
    """Measure which predicted operator term limits object reconstruction.

    Every intervention starts from the same predicted operator and an
    independently rendered GT background. It is a diagnostic, not a training
    target or an additive attribution: changing one term can expose errors
    that another predicted term was previously compensating for.
    """

    background = target.counterfactual_background
    if background is None:
        return
    if background.ndim == 4:
        background = background[:, None].expand(-1, target.frames.shape[1], -1, -1, -1)
    matter = prediction.matter
    defaults = dict(
        alpha=matter.alpha,
        premultiplied_foreground=matter.premultiplied_foreground,
        refractive_flow=matter.refractive_flow,
        transmittance=matter.transmittance,
        residual=matter.residual,
        refractive_kernel_weights=matter.refractive_kernel_weights,
        refractive_kernel_flows=matter.refractive_kernel_flows,
    )

    def record(
        name: str,
        *,
        render_background: Tensor | None = None,
        metric_prefix: str = "render_oracle",
        **overrides: Tensor | None,
    ) -> None:
        parts = defaults | overrides
        selected_background = background if render_background is None else render_background
        if selected_background.ndim == 4:
            selected_background = selected_background[:, None].expand(
                -1, target.frames.shape[1], -1, -1, -1,
            )
        rendered, _ = recompose(
            parts["alpha"], parts["premultiplied_foreground"],
            selected_background,
            parts["refractive_flow"], transmittance=parts["transmittance"],
            residual=parts["residual"],
            refractive_kernel_weights=parts["refractive_kernel_weights"],
            refractive_kernel_flows=parts["refractive_kernel_flows"],
        )
        region = _region_mse(
            rendered.flatten(0, 1), target.frames.flatten(0, 1),
            object_support.flatten(0, 1),
        )
        if region is not None:
            mse, count = region
            metrics[f"{metric_prefix}_{name}_object_psnr"] = (
                float((-10.0 * torch.log10(mse.detach().clamp_min(1e-12))).sum().cpu()),
                count,
            )

    record("background")
    if target.alpha is not None:
        record(
            "alpha",
            alpha=target.alpha,
            premultiplied_foreground=target.alpha * matter.straight_foreground,
            transmittance=(1.0 - target.alpha) * matter.color_transmission,
        )
    if target.color_transmission is not None:
        record(
            "color_T",
            transmittance=(1.0 - matter.alpha) * target.color_transmission,
        )
    if target.transmittance is not None:
        record("tau", transmittance=target.transmittance)
    if target.straight_foreground is not None and target.alpha is not None:
        valid_f = target.alpha >= 0.05
        if target.alpha_validity is not None:
            valid_f = valid_f & (target.alpha_validity > 0.5)
        straight_f = torch.where(
            valid_f, target.straight_foreground, matter.straight_foreground,
        )
        record("foreground_F", premultiplied_foreground=matter.alpha * straight_f)
        if target.residual is not None:
            # F and R can compensate for each other. Replacing F alone can
            # lower PSNR even when its direct GT error is large.
            record(
                "foreground_F_residual_R",
                premultiplied_foreground=matter.alpha * straight_f,
                residual=target.residual,
            )
        record(
            "alpha_foreground_F",
            alpha=target.alpha,
            premultiplied_foreground=target.alpha * straight_f,
            transmittance=(1.0 - target.alpha) * matter.color_transmission,
        )
    if target.refractive_flow is not None:
        flow = target.refractive_flow
        if target.refractive_validity is not None:
            flow = torch.where(
                target.refractive_validity.bool(), flow, matter.refractive_flow,
            )
        # GT is a single source coordinate; disable an optional predicted
        # multi-tap kernel for this intervention only.
        record(
            "flow", refractive_flow=flow,
            refractive_kernel_weights=None, refractive_kernel_flows=None,
        )
        predicted_background = prediction.background.background
        if predicted_background.ndim == 4:
            predicted_background = predicted_background[:, None].expand_as(background)
        record(
            "background_gt_flow", render_background=predicted_background,
            metric_prefix="render_predicted",
            refractive_flow=flow,
            refractive_kernel_weights=None, refractive_kernel_flows=None,
        )
        if target.color_transmission is not None:
            record(
                "gt_flow_color_T",
                refractive_flow=flow,
                transmittance=(1.0 - matter.alpha) * target.color_transmission,
                refractive_kernel_weights=None, refractive_kernel_flows=None,
            )
    if target.residual is not None:
        record("residual_R", residual=target.residual)
    if (
        target.alpha is not None
        and target.premultiplied_foreground is not None
        and target.transmittance is not None
        and target.refractive_flow is not None
        and target.residual is not None
    ):
        # Test the dataset's physical equation before interpreting any
        # one-component oracle swap. An inconsistent GT decomposition makes
        # all such attribution ambiguous.
        record(
            "all_gt",
            alpha=target.alpha,
            premultiplied_foreground=target.premultiplied_foreground,
            transmittance=target.transmittance,
            refractive_flow=target.refractive_flow,
            residual=target.residual,
            refractive_kernel_weights=None,
            refractive_kernel_flows=None,
        )
        record(
            "all_gt_operator",
            render_background=prediction.background.background,
            metric_prefix="render_predicted",
            alpha=target.alpha,
            premultiplied_foreground=target.premultiplied_foreground,
            transmittance=target.transmittance,
            refractive_flow=target.refractive_flow,
            residual=target.residual,
            refractive_kernel_weights=None,
            refractive_kernel_flows=None,
        )


def _batch_metrics(
    prediction: RefractiveMAM2Output,
    target: RefractiveGroundTruth,
    *,
    compute_lpips: bool = False,
    compute_connectivity: bool = False,
    matter_config: MatterConfig | None = None,
) -> dict[str, tuple[float, int]]:
    """Return sums and denominators so metrics are batch-size independent."""

    metrics: dict[str, tuple[float, int]] = {}

    def add(name: str, value: Tensor, count: int) -> None:
        metrics[name] = (float(value.detach().sum().cpu()), count)

    batch, frames = target.frames.shape[:2]
    add(
        "temporal_gt_available",
        target.frames.new_tensor(float(target.temporal_flow_to_next is not None)),
        1,
    )
    if target.temporal_flow_validity is not None:
        temporal_candidates = (
            target.object_mask[:, :-1].sum()
            if target.object_mask is not None
            else target.frames.new_tensor(
                float(target.temporal_flow_validity.numel())
            )
        )
        add(
            "temporal_surface_valid_fraction",
            target.temporal_flow_validity.sum()
            / temporal_candidates.clamp_min(1.0),
            1,
        )
    holdout = _leave_one_out_background(target)
    if holdout is not None and target.refractive_flow is not None:
        _, coverage = holdout
        observable = (
            warp_background(coverage, target.refractive_flow) > 1.0 - 1e-4
        ).to(target.frames.dtype)
        if target.object_mask is not None:
            observable = observable * target.object_mask
        if target.refractive_validity is not None:
            observable = observable * target.refractive_validity
        identifiable = _identifiable_color_transmission(target)
        if isinstance(identifiable, Tensor):
            observable = observable * identifiable
        candidate = (
            target.object_mask.sum()
            if target.object_mask is not None
            else target.frames.new_tensor(float(observable.numel()))
        )
        add(
            "temporal_holdout_observed_fraction",
            observable.sum() / candidate.clamp_min(1.0),
            1,
        )
    operator_support = target.object_mask
    if operator_support is not None:
        far_background = _far_background_support(operator_support)
        far_count = int(far_background.sum().item())
        if far_count:
            for name, value in (
                (
                    "foreground_far_background_abs_mean",
                    prediction.matter.premultiplied_foreground.abs().mean(
                        dim=2, keepdim=True
                    ),
                ),
                (
                    "flow_far_background_epe",
                    torch.linalg.vector_norm(
                        prediction.matter.refractive_flow, dim=2, keepdim=True
                    ),
                ),
                (
                    "residual_far_background_abs_mean",
                    prediction.matter.residual.abs().mean(dim=2, keepdim=True),
                ),
                ("alpha_far_background_mean", prediction.matter.alpha),
            ):
                add(name, (value * far_background).sum(), far_count)
    color_support = (
        _identifiable_color_transmission(target)
        if target.color_transmission is not None else None
    )
    if isinstance(color_support, Tensor) and operator_support is not None:
        add(
            "color_transmission_identifiable_fraction",
            color_support.sum() / operator_support.sum().clamp_min(1.0),
            1,
        )
        # A low RGB MAE alone can conceal collapse to neutral white.  Report
        # color separation only on pixels whose GT T is actually colored.
        truth_t = target.color_transmission
        predicted_t = prediction.matter.color_transmission
        truth_chroma = truth_t.amax(dim=2, keepdim=True) - truth_t.amin(
            dim=2, keepdim=True
        )
        predicted_chroma = predicted_t.amax(dim=2, keepdim=True) - predicted_t.amin(
            dim=2, keepdim=True
        )
        colored_support = color_support * (truth_chroma >= 0.1).to(
            color_support.dtype
        )
        colored_count = int(colored_support.sum().item())
        if colored_count:
            add(
                "colored_T_gt_chroma",
                (truth_chroma * colored_support).sum(),
                colored_count,
            )
            add(
                "colored_T_pred_chroma",
                (predicted_chroma * colored_support).sum(),
                colored_count,
            )
            add(
                "colored_T_neutral_collapse_fraction",
                (
                    (predicted_chroma < 0.05).to(colored_support.dtype)
                    * colored_support
                ).sum(),
                colored_count,
            )
        identifiable_count = int(color_support.sum().item())
        if identifiable_count:
            maximum_error = (
                predicted_t.amax(dim=2, keepdim=True) - 1.0
            ).abs()
            add(
                "color_transmission_max_channel_error",
                (maximum_error * color_support).sum(),
                identifiable_count,
            )

    def operator_mae(
        predicted: Tensor, truth: Tensor, support: Tensor | None = None
    ) -> Tensor:
        error = (predicted - truth).abs().mean(dim=2)
        if support is None:
            support = operator_support
        if support is None:
            return error.mean(dim=(2, 3))
        weight = support.squeeze(2).to(error.dtype)
        return (error * weight).sum(dim=(2, 3)) / weight.sum(dim=(2, 3)).clamp_min(1.0)

    render_mse = (
        (prediction.reconstructed_frames - target.frames).square().mean(dim=(2, 3, 4))
    )
    add("render_mse", render_mse, batch * frames)
    add("render_psnr", -10.0 * torch.log10(render_mse.clamp_min(1e-12)), batch * frames)
    if target.object_mask is not None or target.alpha is not None:
        object_support = (
            target.object_mask >= 0.5
            if target.object_mask is not None
            else torch.zeros_like(target.alpha, dtype=torch.bool)
        )
        if target.alpha is not None:
            object_support = object_support | (target.alpha > 1e-3)
        if prediction.matter.object_support_logits is not None:
            predicted_support = prediction.matter.object_support_logits >= 0
            intersection = (predicted_support & object_support).sum(
                dim=(2, 3, 4)
            ).float()
            union = (predicted_support | object_support).sum(
                dim=(2, 3, 4)
            ).float()
            mask_iou = torch.where(
                union > 0, intersection / union, torch.ones_like(union)
            )
            add("matter_object_support_iou", mask_iou, batch * frames)
            truth_area = object_support.sum(dim=(2, 3, 4)).float()
            mask_recall = torch.where(
                truth_area > 0,
                intersection / truth_area.clamp_min(1.0),
                torch.ones_like(truth_area),
            )
            add("matter_object_support_recall", mask_recall, batch * frames)
            predicted_area = predicted_support.sum(dim=(2, 3, 4)).float()
            mask_precision = torch.where(
                predicted_area > 0,
                intersection / predicted_area.clamp_min(1.0),
                torch.where(truth_area > 0, torch.zeros_like(truth_area), torch.ones_like(truth_area)),
            )
            add("matter_object_support_precision", mask_precision, batch * frames)
            add(
                "matter_object_support_area_ratio",
                predicted_area / truth_area.clamp_min(1.0),
                batch * frames,
            )
        object_render = _region_mse(
            prediction.reconstructed_frames.flatten(0, 1),
            target.frames.flatten(0, 1),
            object_support.flatten(0, 1),
        )
        if object_render is not None:
            object_mse, valid_frames = object_render
            add("render_object_mse", object_mse, valid_frames)
            add(
                "render_object_psnr",
                -10.0 * torch.log10(object_mse.clamp_min(1e-12)),
                valid_frames,
            )
        if target.counterfactual_background is not None:
            _merge_oracle_component_metrics(
                metrics, prediction, target, object_support,
            )

    if target.alpha is not None:
        if target.object_mask is not None:
            alpha_support = target.alpha > 1e-3
            add(
                "alpha_gt_outside_binary_mask_fraction",
                (alpha_support & (target.object_mask < 0.5)).sum(),
                max(int(alpha_support.sum().item()), 1),
            )
        mam2_alpha = F.interpolate(
            prediction.backbone.alpha_matte.flatten(0, 1),
            size=target.alpha.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).reshape_as(target.alpha)
        mam2_alpha_error = mam2_alpha - target.alpha
        add(
            "mam2_alpha_mse",
            mam2_alpha_error.square().mean(dim=(2, 3, 4)),
            batch * frames,
        )
        add(
            "mam2_alpha_sad",
            mam2_alpha_error.abs().sum(dim=(2, 3, 4)) / 1000.0,
            batch * frames,
        )
        alpha_error = prediction.matter.alpha - target.alpha
        add("alpha_mse", alpha_error.square().mean(dim=(2, 3, 4)), batch * frames)
        add("alpha_sad", alpha_error.abs().sum(dim=(2, 3, 4)) / 1000.0, batch * frames)
        pred_dx = (
            prediction.matter.alpha[..., :, 1:] - prediction.matter.alpha[..., :, :-1]
        )
        true_dx = target.alpha[..., :, 1:] - target.alpha[..., :, :-1]
        pred_dy = (
            prediction.matter.alpha[..., 1:, :] - prediction.matter.alpha[..., :-1, :]
        )
        true_dy = target.alpha[..., 1:, :] - target.alpha[..., :-1, :]
        gradient_mse = 0.5 * (
            (pred_dx - true_dx).square().mean(dim=(2, 3, 4))
            + (pred_dy - true_dy).square().mean(dim=(2, 3, 4))
        )
        add("alpha_gradient_mse", gradient_mse, batch * frames)
        boundary_f1 = _alpha_boundary_f1(prediction.matter.alpha, target.alpha)
        add("alpha_boundary_f1", boundary_f1, batch * frames)
        if compute_connectivity:
            add(
                "alpha_connectivity_error",
                _alpha_connectivity_error(prediction.matter.alpha, target.alpha),
                batch * frames,
            )
    if target.counterfactual_background is not None:
        background = target.counterfactual_background
        if background.ndim == 5:
            background = background[:, 0]
        bg_mse = (
            (prediction.background.background - background).square().mean(dim=(1, 2, 3))
        )
        add("background_mse", bg_mse, batch)
        add("background_psnr", -10.0 * torch.log10(bg_mse.clamp_min(1e-12)), batch)
        add(
            "background_gt_above_unit_fraction",
            (background > 1.0).sum(),
            background.numel(),
        )
        add(
            "background_ssim",
            _ssim(prediction.background.background, background),
            batch,
        )
        if compute_lpips:
            add(
                "background_lpips",
                _lpips(prediction.background.background, background),
                batch,
            )
        direct = prediction.background.direct_support
        direct_pixels = int(direct.sum().item())
        if direct_pixels:
            direct_error = prediction.background.background - background
            add(
                "background_direct_pixel_mse",
                (direct_error.square() * direct).sum(),
                direct_pixels * background.shape[1],
            )
            add(
                "background_direct_large_error_fraction",
                ((direct_error.abs().mean(dim=1, keepdim=True) > 0.1) & direct).sum(),
                direct_pixels,
            )
        overlap = direct & prediction.background.inverse_support
        overlap_error = _region_mse(
            prediction.background.inverse_background,
            prediction.background.observed_background,
            overlap,
        )
        if overlap_error is not None:
            overlap_mse, valid_samples = overlap_error
            add("background_inverse_direct_overlap_rmse", overlap_mse.sqrt(), valid_samples)
        add(
            "background_inverse_direct_overlap_fraction",
            overlap.float().mean(dim=(1, 2, 3)),
            batch,
        )
        inverse = prediction.background.inverse_support & ~direct
        true_hole = prediction.background.true_hole
        for name, mask in (
            ("direct", direct),
            ("inverse_only", inverse),
            ("true_hole", true_hole),
            ("occluded", ~direct),
        ):
            region = _region_mse(
                prediction.background.background,
                background,
                mask,
            )
            if region is not None:
                region_mse, valid_samples = region
                add(f"background_{name}_mse", region_mse, valid_samples)
                add(
                    f"background_{name}_psnr",
                    -10.0 * torch.log10(region_mse.clamp_min(1e-12)),
                    valid_samples,
                )
        if target.object_mask is not None or target.alpha is not None:
            occluded_each_frame = (
                target.object_mask >= 0.5
                if target.object_mask is not None
                else torch.zeros_like(target.alpha, dtype=torch.bool)
            )
            if target.alpha is not None:
                occluded_each_frame = occluded_each_frame | (target.alpha > 1e-3)
            gt_always_occluded = occluded_each_frame.all(dim=1)
            false_direct = direct & gt_always_occluded
            add(
                "background_false_direct_fraction",
                false_direct.sum(),
                max(int(gt_always_occluded.sum().item()), 1),
            )
            false_direct_region = _region_mse(
                prediction.background.background, background, false_direct,
            )
            if false_direct_region is not None:
                false_direct_mse, false_direct_samples = false_direct_region
                add(
                    "background_false_direct_psnr",
                    -10.0 * torch.log10(false_direct_mse.clamp_min(1e-12)),
                    false_direct_samples,
                )
            gt_region = _region_mse(
                prediction.background.background,
                background,
                gt_always_occluded,
            )
            if gt_region is not None:
                gt_mse, valid_samples = gt_region
                add("background_gt_occluded_mse", gt_mse, valid_samples)
                add(
                    "background_gt_occluded_psnr",
                    -10.0 * torch.log10(gt_mse.clamp_min(1e-12)),
                    valid_samples,
                )
        raw_inverse_region = _region_mse(
            prediction.background.inverse_background,
            background,
            inverse,
        )
        if raw_inverse_region is not None:
            inverse_mse, valid_samples = raw_inverse_region
            add(
                "inverse_evidence_psnr",
                -10.0 * torch.log10(inverse_mse.clamp_min(1e-12)),
                valid_samples,
            )
        add(
            "background_true_hole_fraction",
            true_hole.float().mean(dim=(1, 2, 3)),
            batch,
        )
        # Only direct observations are immutable. Inverse evidence is a
        # correctable hint, so changing it is not a preservation failure.
        supported = direct
        preservation = (
            prediction.background.background - prediction.background.evidence_background
        ).abs() * supported
        add(
            "evidence_preservation_l1",
            preservation.sum(dim=(1, 2, 3))
            / (supported.sum(dim=(1, 2, 3)).clamp_min(1) * 3),
            batch,
        )
        inverse_change = (
            prediction.background.background - prediction.background.inverse_background
        ).abs() * inverse
        add(
            "inverse_refinement_l1",
            inverse_change.sum(dim=(1, 2, 3))
            / (inverse.sum(dim=(1, 2, 3)).clamp_min(1) * 3),
            batch,
        )
    if target.refractive_flow is not None:
        if matter_config is not None:
            height, width = target.frames.shape[-2:]
            if matter_config.flow_parameterization == "resolution_fraction":
                cap_x = max(width - 1, 1) * matter_config.max_refractive_flow_fraction
                cap_y = max(height - 1, 1) * matter_config.max_refractive_flow_fraction
            else:
                cap_x = cap_y = matter_config.max_refractive_flow
            if matter_config.refine_refractive_flow:
                cap_x += max(width - 1, 1) * matter_config.flow_refinement_max_fraction
                cap_y += max(height - 1, 1) * matter_config.flow_refinement_max_fraction
            if matter_config.refractive_kernel_size > 1:
                cap_x += max(width - 1, 1) * matter_config.refractive_kernel_radius_fraction
                cap_y += max(height - 1, 1) * matter_config.refractive_kernel_radius_fraction
            target_outside_capacity = (
                (target.refractive_flow[:, :, 0:1].abs() > cap_x + 1e-4)
                | (target.refractive_flow[:, :, 1:2].abs() > cap_y + 1e-4)
            )
            valid = (
                torch.ones_like(target_outside_capacity)
                if target.refractive_validity is None
                else target.refractive_validity.bool()
            )
            add(
                "flow_gt_outside_capacity_fraction",
                (target_outside_capacity & valid).sum(),
                max(int(valid.sum().item()), 1),
            )
        endpoint = torch.linalg.vector_norm(
            prediction.matter.refractive_flow - target.refractive_flow,
            dim=2,
            keepdim=True,
        )
        validity = target.refractive_validity
        if validity is not None:
            endpoint_sum = (endpoint * validity).sum()
            valid_count = int(validity.sum().item())
        else:
            endpoint_sum = endpoint.sum()
            valid_count = endpoint.numel()
        add("flow_epe", endpoint_sum, max(valid_count, 1))
        coarse_flow = getattr(prediction.matter, "coarse_refractive_flow", None)
        if coarse_flow is not None:
            coarse_endpoint = torch.linalg.vector_norm(
                coarse_flow - target.refractive_flow,
                dim=2,
                keepdim=True,
            )
            coarse_sum = (
                (coarse_endpoint * validity).sum()
                if validity is not None
                else coarse_endpoint.sum()
            )
            add("coarse_flow_epe", coarse_sum, max(valid_count, 1))
        ground_truth_magnitude = torch.linalg.vector_norm(
            target.refractive_flow,
            dim=2,
            keepdim=True,
        )
        bad = (endpoint > 3.0) & (
            endpoint / ground_truth_magnitude.clamp_min(1e-6) > 0.05
        )
        if validity is not None:
            bad_sum = (bad * validity.bool()).sum()
        else:
            bad_sum = bad.sum()
        add("flow_bad_pixel", bad_sum, max(valid_count, 1))
    if (
        target.surface_normal is not None
        and prediction.matter.surface_normal is not None
    ):
        predicted_normal = F.normalize(
            prediction.matter.surface_normal, dim=2, eps=1e-6
        )
        target_normal = F.normalize(target.surface_normal, dim=2, eps=1e-6)
        cosine = (predicted_normal * target_normal).sum(dim=2, keepdim=True)
        angular = torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0)))
        normal_validity = (
            torch.ones_like(angular)
            if target.normal_validity is None
            else target.normal_validity.to(angular.dtype)
        )
        if operator_support is not None:
            normal_validity = normal_validity * operator_support
        normal_count = int(normal_validity.sum().item())
        add(
            "surface_normal_mean_angular_error",
            (angular * normal_validity).sum(),
            max(normal_count, 1),
        )
    if target.depth is not None and prediction.matter.depth is not None:
        depth_validity = (
            torch.ones_like(target.depth)
            if target.depth_validity is None
            else target.depth_validity.to(target.depth.dtype)
        )
        if operator_support is not None:
            depth_validity = depth_validity * operator_support
        depth_count = int(depth_validity.sum().item())
        depth_error = prediction.matter.depth - target.depth
        add(
            "depth_abs_rel",
            (depth_error.abs() / target.depth.clamp_min(1e-6) * depth_validity).sum(),
            max(depth_count, 1),
        )
        add(
            "depth_rmse",
            torch.sqrt(
                (depth_error.square() * depth_validity).sum()
                / depth_validity.sum().clamp_min(1.0)
            ),
            1,
        )
    for name, predicted, truth in (
        (
            "premultiplied_foreground_mae",
            prediction.matter.premultiplied_foreground,
            target.premultiplied_foreground,
        ),
        (
            "color_transmission_mae",
            prediction.matter.color_transmission,
            target.color_transmission,
        ),
        ("transmittance_mae", prediction.matter.transmittance, target.transmittance),
        ("residual_mae", prediction.matter.residual, target.residual),
        ("confidence_mae", prediction.matter.confidence, target.confidence),
    ):
        if truth is not None:
            if name == "color_transmission_mae":
                support = color_support
                if isinstance(support, Tensor):
                    channel_support = support.expand_as(truth)
                    count = int(channel_support.sum().item())
                    if count:
                        add(name, ((predicted - truth).abs() * channel_support).sum(), count)
                    continue
            add(name, operator_mae(predicted, truth), batch * frames)
    if target.residual is not None and matter_config is not None:
        add("residual_gt_abs_mean", target.residual.abs().sum(), target.residual.numel())
        residual_support = (
            torch.ones_like(target.residual[:, :, :1], dtype=torch.bool)
            if target.object_mask is None
            else target.object_mask >= 0.5
        )
        outside = (
            target.residual.abs().amax(dim=2, keepdim=True)
            > matter_config.residual_scale + 1e-4
        )
        add(
            "residual_gt_outside_capacity_fraction",
            (outside & residual_support).sum(),
            max(int(residual_support.sum().item()), 1),
        )
    foreground_support = _straight_foreground_support(target)
    if target.straight_foreground is not None and foreground_support is not None:
        foreground_pixels = int(foreground_support.sum().item())
        if foreground_pixels:
            if target.alpha is not None:
                # Compare all three G errors on identical GT-alpha-valid
                # pixels. Replacing only alpha or only F identifies which
                # branch is responsible for a blurred premultiplied result,
                # without a second model or background-completion forward.
                target_g = (
                    target.premultiplied_foreground
                    if target.premultiplied_foreground is not None
                    else target.alpha * target.straight_foreground
                )
                for name, candidate in (
                    ("foreground_joint_valid_mae", prediction.matter.premultiplied_foreground),
                    (
                        "foreground_gt_alpha_probe_mae",
                        target.alpha * prediction.matter.straight_foreground,
                    ),
                    (
                        "foreground_gt_f_probe_mae",
                        prediction.matter.alpha * target.straight_foreground,
                    ),
                ):
                    error = (candidate - target_g).abs().mean(dim=2, keepdim=True)
                    add(name, (error * foreground_support).sum(), foreground_pixels)
            foreground_error = (
                prediction.matter.straight_foreground - target.straight_foreground
            ).abs().mean(dim=2, keepdim=True)
            add(
                "straight_foreground_valid_mae",
                (foreground_error * foreground_support).sum(),
                foreground_pixels,
            )
            if prediction.matter.foreground_candidate is not None:
                candidate_error = (
                    prediction.matter.foreground_candidate
                    - target.straight_foreground
                ).abs().mean(dim=2, keepdim=True)
                add(
                    "foreground_candidate_valid_mae",
                    (candidate_error * foreground_support).sum(),
                    foreground_pixels,
                )
            predicted_f = prediction.matter.straight_foreground
            target_f = target.straight_foreground
            horizontal_valid = (
                foreground_support[..., 1:] * foreground_support[..., :-1]
            )
            vertical_valid = (
                foreground_support[..., 1:, :] * foreground_support[..., :-1, :]
            )
            gradient_pixels = int(
                horizontal_valid.sum().item() + vertical_valid.sum().item()
            )
            if gradient_pixels:
                horizontal_error = (
                    predicted_f[..., 1:] - predicted_f[..., :-1]
                    - target_f[..., 1:] + target_f[..., :-1]
                ).abs().mean(dim=2, keepdim=True)
                vertical_error = (
                    predicted_f[..., 1:, :] - predicted_f[..., :-1, :]
                    - target_f[..., 1:, :] + target_f[..., :-1, :]
                ).abs().mean(dim=2, keepdim=True)
                add(
                    "straight_foreground_gradient_mae",
                    (horizontal_error * horizontal_valid).sum()
                    + (vertical_error * vertical_valid).sum(),
                    gradient_pixels,
                )
                predicted_gradient_energy = (
                    (predicted_f[..., 1:] - predicted_f[..., :-1])
                    .abs().mean(dim=2, keepdim=True) * horizontal_valid
                ).sum() + (
                    (predicted_f[..., 1:, :] - predicted_f[..., :-1, :])
                    .abs().mean(dim=2, keepdim=True) * vertical_valid
                ).sum()
                target_gradient_energy = (
                    (target_f[..., 1:] - target_f[..., :-1])
                    .abs().mean(dim=2, keepdim=True) * horizontal_valid
                ).sum() + (
                    (target_f[..., 1:, :] - target_f[..., :-1, :])
                    .abs().mean(dim=2, keepdim=True) * vertical_valid
                ).sum()
                if target_gradient_energy > 1e-6:
                    add(
                        "straight_foreground_gradient_energy_ratio",
                        predicted_gradient_energy / target_gradient_energy,
                        1,
                    )
    if target.transmittance is not None and operator_support is not None:
        zero_tau = operator_support * (
            target.transmittance.amax(dim=2, keepdim=True) <= 1e-3
        ).to(operator_support.dtype)
        zero_support = zero_tau.expand_as(target.transmittance)
        count = int(zero_support.sum().item())
        if count:
            add(
                "zero_transmittance_mae",
                (
                    (prediction.matter.transmittance - target.transmittance).abs()
                    * zero_support
                ).sum(),
                count,
            )
    past_evidence = _past_visible_refracted_background(target)
    if past_evidence is not None:
        _, past_support = past_evidence
        identifiable = _identifiable_color_transmission(target)
        if isinstance(identifiable, Tensor):
            add(
                "past_visible_source_fraction",
                past_support.sum() / identifiable.sum().clamp_min(1.0),
                1,
            )
        if target.transmittance is not None:
            channel_support = past_support.expand_as(target.transmittance)
            count = int(channel_support.sum().item())
            if count:
                add(
                    "past_visible_transmittance_mae",
                    (
                        (prediction.matter.transmittance - target.transmittance).abs()
                        * channel_support
                    ).sum(),
                    count,
                )
        if target.color_transmission is not None:
            channel_support = past_support.expand_as(target.color_transmission)
            count = int(channel_support.sum().item())
            if count:
                add(
                    "past_visible_color_transmission_mae",
                    (
                        (prediction.matter.color_transmission - target.color_transmission).abs()
                        * channel_support
                    ).sum(),
                    count,
                )
    residual_energy = prediction.matter.residual.abs().mean(dim=2)
    if operator_support is not None:
        residual_weight = operator_support.squeeze(2).to(residual_energy.dtype)
        residual_energy = (residual_energy * residual_weight).sum(dim=(2, 3)) / (
            residual_weight.sum(dim=(2, 3)).clamp_min(1.0)
        )
    else:
        residual_energy = residual_energy.mean(dim=(2, 3))
    add("predicted_residual_abs_mean", residual_energy, batch * frames)
    return metrics


def _paired_recomposition_metrics(
    prediction: RefractiveMAM2Output,
    target: RefractiveGroundTruth,
    group_ids: list[str],
) -> dict[str, tuple[float, int]]:
    if target.counterfactual_background is None:
        return {}
    groups: dict[str, list[int]] = {}
    for index, group_id in enumerate(group_ids):
        groups.setdefault(group_id, []).append(index)
    mse_values: list[Tensor] = []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        anchor = indices[0]
        for paired in indices[1:]:
            background = target.counterfactual_background[paired : paired + 1]
            if background.ndim == 4:
                background = background[:, None].expand(
                    -1, target.frames.shape[1], -1, -1, -1
                )
            rendered, _ = recompose(
                prediction.matter.alpha[anchor : anchor + 1],
                prediction.matter.premultiplied_foreground[anchor : anchor + 1],
                background,
                prediction.matter.refractive_flow[anchor : anchor + 1],
                transmittance=prediction.matter.transmittance[anchor : anchor + 1],
                residual=prediction.matter.residual[anchor : anchor + 1],
                refractive_kernel_weights=(
                    None
                    if prediction.matter.refractive_kernel_weights is None
                    else prediction.matter.refractive_kernel_weights[
                        anchor : anchor + 1
                    ]
                ),
                refractive_kernel_flows=(
                    None
                    if prediction.matter.refractive_kernel_flows is None
                    else prediction.matter.refractive_kernel_flows[anchor : anchor + 1]
                ),
            )
            mse_values.append(
                (rendered - target.frames[paired : paired + 1])
                .square()
                .mean(dim=(2, 3, 4))
                .squeeze(0)
            )
    if not mse_values:
        return {}
    mse = torch.cat(mse_values)
    support = target.object_mask
    if support is not None and target.refractive_validity is not None:
        support = support * target.refractive_validity
    consistency = reusable_operator_consistency_in_batch(
        prediction.matter,
        group_ids,
        support,
    )
    return {
        "unseen_recomposition_mse": (float(mse.sum().cpu()), mse.numel()),
        "unseen_recomposition_psnr": (
            float((-10.0 * torch.log10(mse.clamp_min(1e-12))).sum().cpu()),
            mse.numel(),
        ),
        "paired_operator_consistency": (float(consistency.cpu()), 1),
    }


def _region_mse(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
) -> tuple[Tensor, int] | None:
    weight = mask.to(prediction.dtype)
    denominator = weight.sum(dim=(1, 2, 3)) * prediction.shape[1]
    valid = denominator > 0
    if not bool(valid.any()):
        return None
    error = ((prediction - target).square() * weight).sum(dim=(1, 2, 3))
    return error[valid] / denominator[valid], int(valid.sum().item())


def _ssim(prediction: Tensor, target: Tensor) -> Tensor:
    """Windowed SSIM for linear RGB values in the nominal [0,1] range."""

    kernel = 11
    padding = kernel // 2
    mean_pred = F.avg_pool2d(prediction, kernel, stride=1, padding=padding)
    mean_true = F.avg_pool2d(target, kernel, stride=1, padding=padding)
    var_pred = (
        F.avg_pool2d(prediction.square(), kernel, 1, padding) - mean_pred.square()
    )
    var_true = F.avg_pool2d(target.square(), kernel, 1, padding) - mean_true.square()
    covariance = (
        F.avg_pool2d(prediction * target, kernel, 1, padding) - mean_pred * mean_true
    )
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mean_pred * mean_true + c1) * (2 * covariance + c2)) / (
        (mean_pred.square() + mean_true.square() + c1) * (var_pred + var_true + c2)
    ).clamp_min(1e-12)
    return score.mean(dim=(1, 2, 3))


_LPIPS_MODELS: dict[str, torch.nn.Module] = {}


def _lpips(prediction: Tensor, target: Tensor) -> Tensor:
    try:
        import lpips
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "--compute-lpips requires refractive-mam2[evaluation]"
        ) from exc
    key = str(prediction.device)
    model = _LPIPS_MODELS.get(key)
    if model is None:
        model = lpips.LPIPS(net="alex").to(prediction.device).eval()
        _LPIPS_MODELS[key] = model

    def srgb(value: Tensor) -> Tensor:
        value = value.clamp(0, 1)
        return torch.where(
            value <= 0.0031308,
            12.92 * value,
            1.055 * value.pow(1.0 / 2.4) - 0.055,
        )

    return model(2 * srgb(prediction) - 1, 2 * srgb(target) - 1).flatten()


def _alpha_boundary_f1(prediction: Tensor, target: Tensor) -> Tensor:
    batch, frames = prediction.shape[:2]
    predicted = (prediction >= 0.5).flatten(0, 1).float()
    truth = (target >= 0.5).flatten(0, 1).float()

    def boundary(value: Tensor) -> Tensor:
        maximum = F.max_pool2d(value, 3, stride=1, padding=1)
        minimum = -F.max_pool2d(-value, 3, stride=1, padding=1)
        return (maximum - minimum) > 0

    predicted_boundary = boundary(predicted)
    true_boundary = boundary(truth)
    predicted_tolerance = F.max_pool2d(predicted_boundary.float(), 5, 1, 2) > 0
    true_tolerance = F.max_pool2d(true_boundary.float(), 5, 1, 2) > 0
    precision = (predicted_boundary & true_tolerance).sum(dim=(1, 2, 3)).float() / (
        predicted_boundary.sum(dim=(1, 2, 3)).clamp_min(1)
    )
    recall = (true_boundary & predicted_tolerance).sum(dim=(1, 2, 3)).float() / (
        true_boundary.sum(dim=(1, 2, 3)).clamp_min(1)
    )
    score = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    empty = (predicted_boundary.sum(dim=(1, 2, 3)) == 0) & (
        true_boundary.sum(dim=(1, 2, 3)) == 0
    )
    return torch.where(empty, torch.ones_like(score), score).reshape(batch, frames)


def _alpha_connectivity_error(prediction: Tensor, target: Tensor) -> Tensor:
    """Rhemann-style connectivity error, reported as SAD/1000 per frame."""

    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover - data dependency
        raise ImportError(
            "connectivity evaluation requires refractive-mam2[data]"
        ) from exc
    predicted = prediction.detach().float().cpu().numpy()
    truth = target.detach().float().cpu().numpy()
    values: list[float] = []
    for pred_frame, true_frame in zip(
        predicted.reshape(-1, *predicted.shape[-2:]),
        truth.reshape(-1, *truth.shape[-2:]),
    ):
        level = np.full(pred_frame.shape, -1.0, dtype=np.float32)
        thresholds = np.arange(0.0, 1.01, 0.1, dtype=np.float32)
        for threshold_index, threshold in enumerate(thresholds):
            intersection = (
                (pred_frame >= threshold) & (true_frame >= threshold)
            ).astype(np.uint8)
            component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
                intersection,
                connectivity=4,
            )
            omega = np.zeros_like(intersection, dtype=bool)
            if component_count > 1:
                largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                omega = labels == largest
            newly_disconnected = (level < 0) & ~omega
            previous = thresholds[max(threshold_index - 1, 0)]
            level[newly_disconnected] = previous
        level[level < 0] = 1.0
        pred_delta = pred_frame - level
        true_delta = true_frame - level
        pred_phi = 1.0 - pred_delta * (pred_delta >= 0.15)
        true_phi = 1.0 - true_delta * (true_delta >= 0.15)
        values.append(float(np.abs(pred_phi - true_phi).sum() / 1000.0))
    return prediction.new_tensor(values).reshape(prediction.shape[:2])


def _merge_metrics(
    accumulator: dict[str, list[float]],
    values: dict[str, tuple[float, int]],
) -> None:
    for name, (value, count) in values.items():
        current = accumulator.setdefault(name, [0.0, 0.0])
        current[0] += value
        current[1] += count


def evaluate(
    predictor: MAM2VideoPredictor,
    pipeline: RefractiveMAM2,
    dataloader: Iterable[RCTransBatch],
    device: torch.device,
    *,
    stage: str,
    prompt_mode: str,
    prompt_seed: int = 0,
    prompt_jitter_pixels: float = 0.0,
    qualitative_dir: Path | None = None,
    qualitative_limit: int = 0,
    compute_lpips: bool = False,
    compute_connectivity: bool = False,
    wandb_logger: WandbLogger | None = None,
    wandb_prefix: str | None = None,
    wandb_step: int = 0,
    wandb_image_limit: int = 0,
    amp_dtype: str = "float32",
    max_batches: int | None = None,
    loss_profile: str = "standard",
    oracle_foreground_diagnostic_batches: int = 1,
) -> dict[str, float]:
    if oracle_foreground_diagnostic_batches < 0:
        raise ValueError("oracle_foreground_diagnostic_batches must be non-negative")
    predictor.eval()
    pipeline.eval()
    totals: dict[str, list[float]] = {}
    batches = 0
    background_probe_batches = 0
    qualitative_saved = 0
    wandb_images_logged = False
    runtime_seconds = 0.0
    runtime_sequences = 0
    runtime_frames = 0
    peak_memory_mb = 0.0
    with torch.inference_mode():
        for raw_batch in dataloader:
            if max_batches is not None and batches >= max_batches:
                break
            batch = raw_batch.to(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            start_time = time.perf_counter()
            if stage in {"1a", "1b"}:
                with _autocast_context(device, amp_dtype):
                    semantics = _semantic_forward(
                        predictor,
                        batch.ground_truth,
                        prompt_mode=prompt_mode,
                        prompt_seed=prompt_seed,
                        prompt_jitter_pixels=prompt_jitter_pixels,
                        frames_are_linear=not hasattr(batch, "dataset_kinds"),
                        compute_alpha=stage != "1a",
                    )
                    losses = _semantic_loss(
                        semantics,
                        batch.ground_truth,
                        stage=stage,
                    )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                runtime_seconds += time.perf_counter() - start_time
                if device.type == "cuda":
                    peak_memory_mb = max(
                        peak_memory_mb,
                        torch.cuda.max_memory_allocated(device) / (1024**2),
                    )
                runtime_sequences += batch.ground_truth.frames.shape[0]
                runtime_frames += (
                    batch.ground_truth.frames.shape[0]
                    * batch.ground_truth.frames.shape[1]
                )
                for name, value in losses.items():
                    _merge_metrics(totals, {f"loss/{name}": (float(value.cpu()), 1)})
                _merge_metrics(
                    totals,
                    _semantic_metrics(
                        semantics,
                        batch.ground_truth,
                        include_alpha=stage != "1a",
                    ),
                )
                if (
                    wandb_logger is not None
                    and wandb_logger.enabled
                    and wandb_prefix is not None
                    and wandb_image_limit > 0
                    and not wandb_images_logged
                ):
                    sample_ids = getattr(
                        batch,
                        "sequence_ids",
                        [
                            f"sample_{batches}_{index}"
                            for index in range(batch.ground_truth.frames.shape[0])
                        ],
                    )
                    montages = _semantic_montages(
                        semantics,
                        batch.ground_truth,
                        sample_ids,
                        limit=wandb_image_limit,
                        stage=stage,
                    )
                    wandb_logger.log_images(
                        f"{wandb_prefix}/qualitative",
                        [image for image, _ in montages],
                        wandb_step,
                        captions=[caption for _, caption in montages],
                        commit=False,
                    )
                    wandb_images_logged = True
                batches += 1
                predictor.clear_mam2_cache()
                del losses, semantics, batch, raw_batch
                continue
            try:
                with _autocast_context(device, amp_dtype):
                    prediction = _predict(
                        predictor,
                        pipeline,
                        batch.ground_truth,
                        teacher_forcing=stage == "2",
                        prompt_mode=prompt_mode,
                        prompt_seed=prompt_seed,
                        prompt_jitter_pixels=prompt_jitter_pixels,
                        skip_background_estimation=stage == "2",
                    )
                    losses = _loss_terms(
                        stage, prediction, batch, loss_profile=loss_profile
                    )
            except (FloatingPointError, ValueError) as exc:
                sequence_ids = getattr(batch, "sequence_ids", None)
                raise type(exc)(
                    f"evaluation stage={stage} sequences={sequence_ids}: {exc}"
                ) from exc
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            runtime_seconds += time.perf_counter() - start_time
            if device.type == "cuda":
                peak_memory_mb = max(
                    peak_memory_mb,
                    torch.cuda.max_memory_allocated(device) / (1024**2),
                )
            runtime_sequences += batch.ground_truth.frames.shape[0]
            runtime_frames += (
                batch.ground_truth.frames.shape[0] * batch.ground_truth.frames.shape[1]
            )
            for name, value in losses.items():
                _merge_metrics(totals, {f"loss/{name}": (float(value.cpu()), 1)})
            baseline_metrics = _batch_metrics(
                prediction,
                batch.ground_truth,
                compute_lpips=compute_lpips,
                compute_connectivity=compute_connectivity,
                matter_config=pipeline.config.matter,
            )
            _merge_metrics(totals, baseline_metrics)
            if (
                stage in {"3", "4"}
                and batches < oracle_foreground_diagnostic_batches
                and batch.ground_truth.counterfactual_background is not None
                and batch.ground_truth.straight_foreground is not None
            ):
                # Re-use the same semantic features and substitute exact B.
                # Compare predictions on the same small probe subset, rather
                # than comparing a probe result to the full validation mean.
                # This adds no second SAM2 or FLUX call.
                with _autocast_context(device, amp_dtype):
                    oracle_prediction = pipeline.forward_from_backbone(
                        batch.ground_truth.frames,
                        prediction.backbone,
                        counterfactual_background_gt=(
                            batch.ground_truth.counterfactual_background
                        ),
                        use_ground_truth_background=True,
                        skip_background_estimation=True,
                    )
                oracle_metrics = _batch_metrics(
                    oracle_prediction,
                    batch.ground_truth,
                    matter_config=pipeline.config.matter,
                )
                for name in (
                    "straight_foreground_valid_mae",
                    "straight_foreground_gradient_mae",
                    "straight_foreground_gradient_energy_ratio",
                    "render_object_psnr",
                ):
                    if name in baseline_metrics:
                        _merge_metrics(
                            totals,
                            {f"estimated_background_probe_{name}": baseline_metrics[name]},
                        )
                    if name in oracle_metrics:
                        _merge_metrics(
                            totals,
                            {f"oracle_background_probe_{name}": oracle_metrics[name]},
                        )
                background_probe_batches += 1
                del oracle_prediction
            group_ids = getattr(batch, "paired_background_group_ids", [])
            if len(group_ids) != len(set(group_ids)):
                _merge_metrics(
                    totals,
                    _paired_recomposition_metrics(
                        prediction,
                        batch.ground_truth,
                        group_ids,
                    ),
                )
            if qualitative_dir is not None and qualitative_saved < qualitative_limit:
                sample_ids = getattr(
                    batch,
                    "sequence_ids",
                    [
                        f"sample_{batches}_{index}"
                        for index in range(batch.frames.shape[0])
                    ],
                )
                qualitative_saved += _save_qualitative(
                    prediction,
                    batch.ground_truth,
                    qualitative_dir,
                    sample_ids,
                    remaining=qualitative_limit - qualitative_saved,
                )
            if (
                wandb_logger is not None
                and wandb_logger.enabled
                and wandb_prefix is not None
                and wandb_image_limit > 0
                and not wandb_images_logged
            ):
                sample_ids = getattr(
                    batch,
                    "sequence_ids",
                    [
                        f"sample_{batches}_{index}"
                        for index in range(batch.ground_truth.frames.shape[0])
                    ],
                )
                montages = _supervised_prediction_montages(
                    prediction,
                    batch.ground_truth,
                    sample_ids,
                    limit=wandb_image_limit,
                    stage=stage,
                    teacher_forcing=stage == "2",
                )
                wandb_logger.log_images(
                    f"{wandb_prefix}/qualitative",
                    [image for image, _ in montages],
                    wandb_step,
                    captions=[caption for _, caption in montages],
                    commit=False,
                )
                wandb_images_logged = True
            batches += 1
            predictor.clear_mam2_cache()
            del losses, prediction, batch, raw_batch
    if batches == 0:
        raise RuntimeError("evaluation dataloader is empty")
    result = {name: value / max(count, 1.0) for name, (value, count) in totals.items()}
    if "background_direct_pixel_mse" in result:
        # A PSNR computed *after* pixel-weighted aggregation cannot be
        # inflated by many tiny, almost-perfect direct regions masking a few
        # badly contaminated samples.
        result["background_direct_pixel_psnr"] = -10.0 * math.log10(
            max(result["background_direct_pixel_mse"], 1e-12)
        )
    if "matter_object_support_iou" in result and "render_object_psnr" in result:
        # A high reconstruction PSNR cannot win by flooding the object mask.
        # A tenfold IoU drop costs 20 dB while retaining the same PSNR units.
        result["render_support_aware_score"] = _render_support_aware_score(
            result["render_object_psnr"],
            result["matter_object_support_iou"],
        )
    if background_probe_batches:
        result["background_probe_batches"] = float(background_probe_batches)
    result["runtime_seconds_per_sequence"] = runtime_seconds / max(runtime_sequences, 1)
    result["runtime_seconds_per_frame"] = runtime_seconds / max(runtime_frames, 1)
    if device.type == "cuda":
        result["peak_memory_mb"] = peak_memory_mb
    non_finite = {
        name: value for name, value in result.items() if not math.isfinite(value)
    }
    if non_finite:
        raise FloatingPointError(f"non-finite evaluation metrics: {non_finite}")
    return result


def _configure_stage(
    stage: str,
    predictor: MAM2VideoPredictor,
    pipeline: RefractiveMAM2,
) -> list[torch.nn.Parameter]:
    if stage == "1a":
        return configure_stage1a(predictor)
    if stage == "1b":
        return configure_stage1b(predictor)
    if stage == "2":
        return configure_stage2(predictor, pipeline)
    if stage == "3":
        return configure_stage3(predictor, pipeline)
    if stage == "4":
        return configure_stage4(predictor, pipeline)
    raise ValueError(f"unsupported training stage: {stage}")


def _training_teacher_forcing(
    stage: str, probability: float, *, evidence_flux_only: bool,
    optical_curriculum: bool = False,
) -> bool:
    """Use oracle B only for Stage 2 or an explicitly enabled Stage-3 curriculum."""

    if stage == "2":
        return True
    if stage == "3" and (optical_curriculum or not evidence_flux_only):
        return random.random() < probability
    return False


def _optical_curriculum_probability(epoch: int, total_epochs: int) -> float:
    """Start Stage 3 with GT B, then fade it out before FLUX-only fine-tuning."""

    progress = epoch / max(total_epochs - 1, 1)
    if progress <= 1.0 / 3.0:
        return 1.0
    if progress >= 2.0 / 3.0:
        return 0.0
    return 2.0 - 3.0 * progress


def _should_validate_epoch(epoch: int, total_epochs: int, interval: int) -> bool:
    """Always validate the first and final epoch, even with sparse checks."""

    if interval < 1:
        raise ValueError("validation interval must be positive")
    return epoch == 0 or (epoch + 1) % interval == 0 or epoch + 1 == total_epochs


def _should_check_parameter_finiteness(global_step: int, interval: int) -> bool:
    """Keep the first-step check while avoiding a full parameter scan every step."""

    if interval < 1:
        raise ValueError("parameter finiteness interval must be positive")
    return global_step == 0 or (global_step + 1) % interval == 0


def _validate_diffusion_training_mode(
    *, stage: str, mode: str, completion_variant: str,
    evidence_flux_only: bool,
) -> None:
    if completion_variant != "diffusion" or mode == "test":
        return
    if evidence_flux_only and stage in ("3", "4"):
        return
    raise SystemExit(
        "pretrained inpainting training is allowed only with --evidence-flux-only "
        "(alias --evidence-pretrained-only) in Stage 3/4; otherwise use --mode test"
    )


def _optimizer_groups(
    stage: str,
    parameters: list[torch.nn.Parameter],
    predictor: MAM2VideoPredictor,
    *,
    learning_rate: float,
    joint_mam2_lr_scale: float,
    foreground_prior_model: torch.nn.Module | None = None,
    fba_decoder_lr_scale: float = 0.1,
) -> list[dict[str, object]] | list[torch.nn.Parameter]:
    """Keep pretrained decoders at a conservative LR during joint training."""

    if stage != "4" and foreground_prior_model is None:
        return parameters
    mam2_ids = (
        {
            id(parameter)
            for parameter in predictor.parameters()
            if parameter.requires_grad
        }
        if stage == "4"
        else set()
    )
    prior_ids = (
        {
            id(parameter)
            for parameter in foreground_prior_model.parameters()
            if parameter.requires_grad
        }
        if foreground_prior_model is not None
        else set()
    )
    physics = [
        parameter for parameter in parameters
        if id(parameter) not in mam2_ids and id(parameter) not in prior_ids
    ]
    mam2 = [parameter for parameter in parameters if id(parameter) in mam2_ids]
    prior = [parameter for parameter in parameters if id(parameter) in prior_ids]
    groups: list[dict[str, object]] = []
    if physics:
        groups.append({"params": physics, "lr": learning_rate, "name": "physics"})
    if mam2:
        groups.append(
            {
                "params": mam2,
                "lr": learning_rate * joint_mam2_lr_scale,
                "name": "mam2",
            }
        )
    if prior:
        groups.append(
            {
                "params": prior,
                "lr": learning_rate * fba_decoder_lr_scale,
                "name": "fba_decoder",
            }
        )
    return groups


def _operator_head_gradient_metrics(pipeline: RefractiveMAM2) -> dict[str, Tensor]:
    """Expose whether F, flow, transmission, and optical adapters update.

    Raw loss magnitudes have different physical units. These inexpensive
    output-channel gradient norms are measured after global clipping, so their
    relative size is still informative for a few-run diagnostic.
    """

    metrics: dict[str, Tensor] = {}
    heads = (
        ("pam", pipeline.matter.head, (("f", 1, 4), ("flow", 4, 6), ("t", 7, 10))),
        (
            "bridge",
            pipeline.matter.latent_bridge.head
            if pipeline.matter.latent_bridge is not None else None,
            (("f", 0, 3), ("flow", 3, 5), ("t", 5, 8)),
        ),
    )
    for prefix, head, channels in heads:
        if head is None or head.weight.grad is None:
            continue
        for name, start, end in channels:
            squared = head.weight.grad[start:end].float().square().sum()
            if head.bias is not None and head.bias.grad is not None:
                squared = squared + head.bias.grad[start:end].float().square().sum()
            metrics[f"train/{prefix}_head_grad_{name}"] = squared.sqrt()
    optical_adapters = (
        (
            "matte_flow_adapter",
            getattr(getattr(pipeline.backbone, "mam2_matter", None),
                    "flow_feature_modulator", None),
        ),
        (
            "background_flow_adapter",
            getattr(getattr(pipeline.background_model, "diffusion_completion", None),
                    "flow_latent_adapter", None),
        ),
        (
            "background_pixel_refiner",
            getattr(getattr(pipeline.background_model, "diffusion_completion", None),
                    "pixel_refiner", None),
        ),
    )
    for name, module in optical_adapters:
        if module is None:
            continue
        gradients = [
            parameter.grad.float().square().sum()
            for parameter in module.parameters()
            if parameter.grad is not None
        ]
        if gradients:
            metrics[f"train/{name}_grad"] = torch.stack(gradients).sum().sqrt()
    return metrics


def _loader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    workers: int,
    paired_backgrounds: bool,
    seed: int,
) -> DataLoader[RCTransBatch]:
    if isinstance(dataset, IterableDataset):
        if workers not in (0, 1):
            raise ValueError(
                "remote shard streaming permits only --workers 0 or 1 so one "
                "process owns the bounded download cache"
            )
        options = dict(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=prism_collate,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=paired_backgrounds,
        )
        if workers == 1:
            # The worker fills two CPU batches while the main process executes
            # CUDA work. A new worker is spawned each epoch so set_epoch state
            # is copied correctly into the IterableDataset instance.
            options["prefetch_factor"] = 2
            options["persistent_workers"] = False
        return DataLoader(
            **options,
        )
    if paired_backgrounds:
        return build_paired_prism_dataloader(
            dataset,
            backgrounds_per_group=batch_size,
            shuffle=shuffle,
            seed=seed,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
        )
    options = dict(
        dataset=dataset,
        batch_size=batch_size,
        sampler=EpochShuffleSampler(dataset, shuffle=shuffle, seed=seed),
        collate_fn=prism_collate,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    if workers > 0:
        options["prefetch_factor"] = 2
        # Training workers must be recreated after dataset.set_epoch() so the
        # epoch-dependent temporal crop/flip state reaches worker processes.
        options["persistent_workers"] = not shuffle
    return DataLoader(**options)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PRISM stage-wise training and evaluation"
    )
    parser.add_argument("--train-data", type=Path)
    parser.add_argument("--val-data", type=Path)
    parser.add_argument("--test-data", type=Path)
    parser.add_argument(
        "--archive-volume",
        help=(
            "stream train/validation/test tar shards from this VESSL volume "
            "instead of materializing the complete split"
        ),
    )
    parser.add_argument("--archive-storage-name", default="vessl-storage")
    parser.add_argument(
        "--archive-cache-root",
        type=Path,
        default=Path("/root/workspace/prism-shard-cache"),
    )
    parser.add_argument(
        "--archive-shuffle-buffer",
        type=int,
        default=16,
        help="number of eagerly loaded sequence samples shuffled in bounded memory",
    )
    parser.add_argument(
        "--archive-max-shards",
        type=int,
        help="limit each split to its first N shards for a bounded smoke run",
    )
    parser.add_argument(
        "--archive-download-retries",
        type=int,
        default=5,
        help="whole-shard retries with refreshed VESSL credentials",
    )
    parser.add_argument(
        "--archive-prefetch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "download and verify the next remote shard in a background thread "
            "while the GPU trains on the current shard"
        ),
    )
    parser.add_argument(
        "--archive-decode-workers",
        type=int,
        default=1,
        help="parallel CPU sequence decoders inside the single archive worker",
    )
    parser.add_argument(
        "--materialized-eval-data",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="read validation/test locally while continuing to stream train shards",
    )
    parser.add_argument(
        "--sam2-config",
        default=SAM2_CONFIG,
        help=f"official SAM2 Hydra config (default: {SAM2_CONFIG})",
    )
    parser.add_argument(
        "--sam2-checkpoint",
        type=Path,
        default=SAM2_CHECKPOINT,
        help=f"official SAM2 checkpoint (default: {SAM2_CHECKPOINT})",
    )
    parser.add_argument(
        "--mam2-matter-backend",
        choices=("builtin", "external_mematte", "official_mam2", "official_sam2matting"),
        default="external_mematte",
        help="released MAM2, standalone MEMatte, or builtin smoke-test ablation",
    )
    parser.add_argument("--official-mam2-root", type=Path)
    parser.add_argument("--official-mam2-config", type=Path)
    parser.add_argument("--official-sam2matting-root", type=Path)
    parser.add_argument("--official-sam2matting-checkpoint", type=Path)
    parser.add_argument(
        "--official-sam2matting-variant", choices=("tiny", "base_plus"), default="tiny"
    )
    parser.add_argument("--mematte-root", type=Path, default=MEMATTE_ROOT)
    parser.add_argument("--mematte-config", type=Path, default=MEMATTE_CONFIG)
    parser.add_argument("--mematte-checkpoint", type=Path, default=MEMATTE_CHECKPOINT)
    parser.add_argument("--mematte-max-tokens", type=int, default=12000)
    parser.add_argument(
        "--mematte-patch-decoder",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--mematte-train-decoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="fine-tune only MEMatte's decoder in Stage 1B/4",
    )
    parser.add_argument(
        "--mematte-train-backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "fine-tune MEMatte adaptive-token backbone in Stage 1B/4; disable "
            "only for the lower-memory decoder-only ablation"
        ),
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--checkpoint-mam2-only",
        action="store_true",
        help="Stage 2 only: transfer final Stage 1B MAM2 weights, reinitializing PAM",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--mode", choices=("train", "test", "both"), default="both")
    parser.add_argument(
        "--stage",
        choices=("1a", "1b", "2", "3", "4"),
        default="4",
        help="1a semantics, 1b alpha matter, 2 PAM, 3 background, 4 joint",
    )
    parser.add_argument(
        "--stage1-manifest",
        type=Path,
        action="append",
        default=[],
        help="Stage 1A/1B JSONL VOS/matting manifest; may be repeated",
    )
    parser.add_argument("--prompt-seed", type=int, default=0)
    parser.add_argument("--prompt-jitter-pixels", type=float, default=0.0)
    parser.add_argument("--prompt-robustness-runs", type=int, default=0)
    parser.add_argument("--semantic-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--loss-profile", choices=(
            "standard", "optical_priority", "joint_sharp", "foreground_recovery",
            "detail_recovery", "flow_recovery",
        ),
        default="standard",
        help="joint_sharp adds flow-coupled and object-edge detail to optical_priority",
    )
    parser.add_argument(
        "--validate-every-epochs", type=int, default=1,
        help="run full validation at epoch 1, every N epochs, and the final epoch",
    )
    parser.add_argument(
        "--progress-interval-steps",
        type=int,
        default=50,
        help="print a compact optimizer-step heartbeat every N steps; 0 disables",
    )
    parser.add_argument(
        "--checkpoint-interval-steps",
        type=int,
        default=50,
        help="atomically update a resumable checkpoint every N optimizer steps; 0 disables",
    )
    parser.add_argument(
        "--keep-epoch-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="retain a separate checkpoint for every epoch in addition to resume/best",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        help="cap optimizer batches per epoch for an end-to-end smoke run",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        help="cap validation/test batches for an end-to-end smoke run",
    )
    parser.add_argument("--clip-length", type=int)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument(
        "--temporal-correspondence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="load pose/depth-verified object motion for Stage 3/4 temporal losses",
    )
    parser.add_argument(
        "--random-horizontal-flip",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--eval-workers",
        type=int,
        help="validation/test loader workers; defaults to --workers",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--joint-mam2-lr-scale",
        type=float,
        default=0.1,
        help="Stage-4 LR multiplier for PDD/MSS/LoRA and the alpha decoder",
    )
    parser.add_argument(
        "--fba-decoder-lr-scale",
        type=float,
        default=0.1,
        help="LR multiplier for an optionally fine-tuned pretrained FBA decoder",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", choices=("cosine", "none"), default="cosine")
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--amp-dtype",
        choices=("float32", "bfloat16", "float16"),
        default="float32",
        help="autocast dtype for model forward/loss; use bfloat16 on A100",
    )
    parser.add_argument(
        "--activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="recompute builtin MAM2 matte residual blocks during backward",
    )
    parser.add_argument(
        "--matte-full-activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="recompute each alpha-matte frame chunk during backward",
    )
    parser.add_argument(
        "--matte-frame-chunk-size",
        type=int,
        default=4,
        help="number of flattened video frames processed by alpha matter at once",
    )
    parser.add_argument(
        "--sam2-temporal-activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="checkpoint SAM2 image-encoder work along the clip time axis",
    )
    parser.add_argument(
        "--sam2-temporal-checkpoint-chunk-size",
        type=int,
        default=4,
        help="number of frames per checkpointed SAM2 image-encoder call",
    )
    parser.add_argument(
        "--sam2-temporal-detach-interval",
        type=int,
        default=0,
        help="detach recurrent SAM2 memory every N frames; 0 keeps full BPTT",
    )
    parser.add_argument("--teacher-forcing-start", type=float, default=1.0)
    parser.add_argument("--teacher-forcing-end", type=float, default=0.0)
    parser.add_argument(
        "--optical-curriculum", action=argparse.BooleanOptionalAction,
        default=False,
        help="Stage 3 only: GT-background warm-up, transition, then FLUX-only",
    )
    parser.add_argument("--paired-backgrounds", action="store_true")
    parser.add_argument(
        "--paired-microbatch-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "checkpoint Stage 3/4 paired samples independently, then concatenate "
            "their outputs for the unchanged batch-level objective"
        ),
    )
    parser.add_argument("--paired-eval", action="store_true")
    parser.add_argument("--allow-unpaired-joint", action="store_true")
    parser.add_argument("--allow-uninitialized-stage", action="store_true")
    parser.add_argument("--selection-metric")
    parser.add_argument("--refinement-steps", type=int, default=3)
    parser.add_argument("--matter-width", type=int, default=96)
    parser.add_argument(
        "--max-refractive-flow-fraction",
        type=float,
        default=0.5,
        help="maximum coarse displacement as a fraction of image width/height",
    )
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=0.5,
        help="maximum absolute object-attached residual radiance",
    )
    parser.add_argument(
        "--alpha-refinement-scale",
        type=float,
        default=0.5,
        help="maximum MAM2 alpha correction, including object interiors",
    )
    parser.add_argument(
        "--antialiased-boundary-support",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="retain a MAM2-supported one-pixel shell beyond the center-ray mask",
    )
    parser.add_argument(
        "--inverse-min-reliable-weight",
        type=float,
        default=0.05,
        help="minimum inverse-splat weight before it counts as background evidence",
    )
    parser.add_argument(
        "--inverse-variance-scale",
        type=float,
        default=12.0,
        help="downweight inverse background samples that disagree across frames",
    )
    parser.add_argument(
        "--matter-encoder-depths",
        type=int,
        nargs=3,
        default=(2, 3, 4),
        metavar=("FULL", "HALF", "QUARTER"),
    )
    parser.add_argument("--matter-max-channels", type=int, default=512)
    parser.add_argument("--matter-temporal-blocks", type=int, default=2)
    parser.add_argument(
        "--flow-refinement",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="refine coarse refractive flow using its photometric residual",
    )
    parser.add_argument("--flow-refinement-width", type=int, default=32)
    parser.add_argument("--flow-refinement-max-fraction", type=float, default=0.0625)
    parser.add_argument(
        "--foreground-refinement",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="refine straight F from the flow-guided photometric residual",
    )
    parser.add_argument("--foreground-refinement-width", type=int, default=32)
    parser.add_argument(
        "--learned-object-support",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="train a PAM object-support correction over the frozen SAM2 mask",
    )
    parser.add_argument(
        "--foreground-refinement-logit-scale", type=float, default=2.0,
    )
    parser.add_argument(
        "--flow-guided-latent-bridge",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="couple MAM2 foreground features and FLUX background features at x+flow",
    )
    parser.add_argument("--latent-bridge-width", type=int, default=32)
    parser.add_argument("--latent-bridge-flow-fraction", type=float, default=0.0625)
    parser.add_argument(
        "--latent-bridge-foreground-logit-scale", type=float, default=1.0,
    )
    parser.add_argument(
        "--latent-bridge-transmission-logit-scale", type=float, default=1.0,
    )
    parser.add_argument(
        "--use-fba-foreground-prior",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use official pretrained FBA pre-fusion decoder features in the flow-guided bridge",
    )
    parser.add_argument(
        "--use-mam2-foreground-prior",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="feed pretrained SAM2/MEMatte features directly to the flow-guided bridge",
    )
    parser.add_argument("--fba-source-root", type=Path)
    parser.add_argument("--fba-checkpoint", type=Path)
    parser.add_argument("--fba-input-side", type=int, default=320)
    parser.add_argument(
        "--fba-train-decoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="fine-tune FBA decoder through PRISM physical losses; encoder stays frozen",
    )
    parser.add_argument(
        "--evidence-flux-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="legacy spelling for pretrained-only background completion",
    )
    parser.add_argument(
        "--evidence-pretrained-only",
        dest="evidence_flux_only",
        action="store_true",
        help="use released inpainting weights plus inverse-flow evidence without the legacy completion net",
    )
    parser.add_argument(
        "--predict-geometry",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "append object-surface normal, metric depth, and geometry-confidence "
            "heads to PAM and load the v17 N_object/D_object supervision"
        ),
    )
    parser.add_argument("--geometry-min-depth", type=float, default=1e-3)
    parser.add_argument("--refractive-kernel-size", type=int, default=1)
    parser.add_argument(
        "--refractive-kernel-radius-fraction",
        type=float,
        default=0.015625,
    )
    parser.add_argument(
        "--refractive-kernel-center-bias",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--ablation",
        choices=(
            "full",
            "no_inverse",
            "no_joint_refinement",
            "detached_paths",
            "scalar_transmission",
            "no_residual",
            "no_completion",
            "direct_only",
        ),
        default="full",
    )
    parser.add_argument(
        "--selection-mode",
        choices=("max", "min"),
        default=None,
    )
    parser.add_argument(
        "--strict-contract", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--runtime-contract-checks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "repeat expensive per-sample numeric contract checks; disable only "
            "after archive SHA and generator preflight validation"
        ),
    )
    parser.add_argument(
        "--parameter-finiteness-interval-steps", type=int, default=1,
        help="scan all PAM parameters on the first and every Nth optimizer step",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--cudnn-benchmark",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="benchmark deterministic cuDNN kernels for fixed-size training clips",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--prompt-mode",
        choices=("point", "box", "mask"),
        default="point",
        help="mask is an oracle protocol and should not be the primary result",
    )
    parser.add_argument("--project-name", default="PRISM")
    parser.add_argument("--qualitative-limit", type=int, default=8)
    parser.add_argument("--compute-lpips", action="store_true")
    parser.add_argument(
        "--compute-connectivity",
        action="store_true",
        help="compute the CPU-heavy matting connectivity metric during final test",
    )
    parser.add_argument(
        "--wandb-mode", choices=("disabled", "offline", "online"), default="disabled"
    )
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-tags", nargs="*", default=[])
    parser.add_argument("--wandb-notes")
    parser.add_argument(
        "--wandb-image-interval",
        type=int,
        default=200,
        help="log train image panels every N optimizer steps; 0 disables them",
    )
    parser.add_argument(
        "--wandb-scalar-interval-steps", type=int, default=1,
        help="log train scalars every N optimizer steps; avoids per-step GPU sync",
    )
    parser.add_argument(
        "--wandb-image-limit",
        type=int,
        default=2,
        help="maximum sample panels per train/validation/test image log",
    )
    parser.add_argument(
        "--completion-variant",
        choices=("base", "diffusion"),
        default="base",
    )
    parser.add_argument(
        "--completion-backbone",
        choices=("ffc", "dilated"),
        default="ffc",
        help="per-iteration PRISM completion backbone; ffc is GLaMa-style",
    )
    parser.add_argument("--completion-width", type=int, default=64)
    parser.add_argument("--completion-down-blocks", type=int, default=3)
    parser.add_argument("--completion-residual-blocks", type=int, default=9)
    parser.add_argument("--completion-global-ratio", type=float, default=0.5)
    parser.add_argument("--completion-max-channels", type=int, default=512)
    parser.add_argument(
        "--background-exclusion-dilation",
        type=int,
        default=BackgroundConfig.exclusion_dilation,
        help="odd exclusion kernel width in pixels; 1 disables spatial expansion",
    )
    parser.add_argument(
        "--flux-direct-observation-threshold",
        type=float,
        default=BackgroundConfig.flux_direct_observation_threshold,
        help="maximum predicted exclusion probability for immutable direct B evidence",
    )
    parser.add_argument("--diffusion-model")
    parser.add_argument(
        "--diffusion-backend", choices=("auto", "objectclear"), default="auto",
        help="ObjectClear uses its released SDXL weights and a trainable flow-conditioned latent adapter",
    )
    parser.add_argument(
        "--objectclear-source-root", type=Path,
        help="checkout of the released ObjectClear source; required unless the package is installed",
    )
    parser.add_argument(
        "--objectclear-dtype", choices=("bfloat16", "float32"), default="bfloat16",
        help="numerically safe VAE dtype for differentiable ObjectClear latent decoding",
    )
    parser.add_argument(
        "--objectclear-fused-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use ObjectClear's final attention-guided-fusion image before the flow latent adapter",
    )
    parser.add_argument(
        "--objectclear-late-denoise-steps", type=int, default=0,
        help="experimental: inject optical evidence before the final K frozen ObjectClear denoising steps; requires --no-objectclear-fused-output and no CPU offload",
    )
    parser.add_argument(
        "--objectclear-pixel-refinement",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="train a bounded 64px background detail residual after the frozen ObjectClear VAE",
    )
    parser.add_argument("--diffusion-adapter")
    parser.add_argument("--diffusion-revision")
    parser.add_argument("--diffusion-prompt", default=BackgroundConfig.diffusion_prompt)
    parser.add_argument(
        "--diffusion-negative-prompt",
        default=BackgroundConfig.diffusion_negative_prompt,
    )
    parser.add_argument("--diffusion-steps", type=int, default=25)
    parser.add_argument("--diffusion-guidance-scale", type=float, default=7.5)
    parser.add_argument("--diffusion-seed", type=int, default=0)
    parser.add_argument("--diffusion-mask-dilation", type=int, default=0)
    parser.add_argument("--diffusion-min-working-side", type=int, default=0)
    parser.add_argument(
        "--diffusion-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument("--diffusion-device")
    parser.add_argument(
        "--diffusion-cpu-offload",
        choices=("auto", "on", "off"),
        default="auto",
        help="auto keeps FLUX on GPU only when at least 56 GiB is free",
    )
    return parser


def _diffusion_cpu_offload(mode: str, device: torch.device) -> bool:
    """Keep the frozen FLUX model resident only with ample GPU headroom."""

    if mode == "on":
        return True
    if mode == "off":
        return False
    if mode != "auto":
        raise ValueError(f"unknown diffusion CPU offload mode: {mode}")
    if device.type != "cuda":
        return False
    free_bytes, _ = torch.cuda.mem_get_info(device)
    return free_bytes < 56 * 1024**3


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.mode in {"train", "both"}:
        if args.val_data is None:
            raise SystemExit("--val-data is required for train/both mode")
        if args.train_data is None and not (
            args.stage in {"1a", "1b"} and args.stage1_manifest
        ):
            raise SystemExit(
                "--train-data is required unless Stage 1A/1B uses --stage1-manifest"
            )
        if (
            args.stage != "1a"
            and args.checkpoint is None
            and args.resume is None
            and not args.allow_uninitialized_stage
            and not (
                args.stage == "2"
                and args.mam2_matter_backend in ("official_mam2", "official_sam2matting")
            )
        ):
            raise SystemExit(
                "Stage 1B/2/3/4 training requires the preceding checkpoint; "
                "Stage 2 can instead start from a released pretrained video-matting backend"
            )
        if (
            args.stage in {"3", "4"}
            and not args.paired_backgrounds
            and not args.allow_unpaired_joint
        ):
            raise SystemExit(
                "Stage 3/4 requires --paired-backgrounds; use --allow-unpaired-joint "
                "only for the no-reuse ablation"
            )
    if args.mode in {"test", "both"} and args.test_data is None:
        raise SystemExit("--test-data is required for test/both mode")
    if args.batch_size < 1 or args.epochs < 1:
        raise SystemExit("--batch-size and --epochs must be positive")
    if args.workers < 0 or (args.eval_workers is not None and args.eval_workers < 0):
        raise SystemExit("--workers and --eval-workers must be non-negative")
    if args.checkpoint_interval_steps < 0:
        raise SystemExit("--checkpoint-interval-steps must be non-negative")
    if args.parameter_finiteness_interval_steps < 1:
        raise SystemExit("--parameter-finiteness-interval-steps must be positive")
    if args.progress_interval_steps < 0:
        raise SystemExit("--progress-interval-steps must be non-negative")
    if args.archive_shuffle_buffer < 1:
        raise SystemExit("--archive-shuffle-buffer must be positive")
    if args.archive_max_shards is not None and args.archive_max_shards < 1:
        raise SystemExit("--archive-max-shards must be positive")
    if args.archive_download_retries < 1:
        raise SystemExit("--archive-download-retries must be positive")
    if args.archive_decode_workers < 1:
        raise SystemExit("--archive-decode-workers must be positive")
    if args.archive_volume and args.workers not in (0, 1):
        raise SystemExit(
            "--archive-volume requires --workers 0 or 1 so only one process "
            "owns the bounded shard cache"
        )
    if args.lr <= 0 or args.joint_mam2_lr_scale <= 0 or args.fba_decoder_lr_scale <= 0:
        raise SystemExit("--lr, --joint-mam2-lr-scale and --fba-decoder-lr-scale must be positive")
    if args.wandb_image_interval < 0 or args.wandb_image_limit < 0:
        raise SystemExit("W&B image interval and limit must be non-negative")
    if args.wandb_scalar_interval_steps < 1:
        raise SystemExit("--wandb-scalar-interval-steps must be positive")
    if args.matte_frame_chunk_size < 1:
        raise SystemExit("--matte-frame-chunk-size must be positive")
    if args.sam2_temporal_checkpoint_chunk_size < 1:
        raise SystemExit("--sam2-temporal-checkpoint-chunk-size must be positive")
    if args.sam2_temporal_detach_interval < 0:
        raise SystemExit("--sam2-temporal-detach-interval must be non-negative")
    if args.prompt_jitter_pixels < 0 or args.prompt_robustness_runs < 0:
        raise SystemExit("prompt jitter and robustness runs must be non-negative")
    if args.refinement_steps < 0:
        raise SystemExit("refinement steps must be non-negative")
    if args.validate_every_epochs < 1:
        raise SystemExit("--validate-every-epochs must be positive")
    if (
        args.matter_width < 1
        or args.matter_max_channels < args.matter_width
        or min(args.matter_encoder_depths) < 1
        or args.matter_temporal_blocks < 0
    ):
        raise SystemExit("matter network dimensions must be positive and consistent")
    if args.refractive_kernel_size < 1 or args.refractive_kernel_size % 2 == 0:
        raise SystemExit("--refractive-kernel-size must be a positive odd integer")
    if args.flow_refinement_width < 1 or args.flow_refinement_max_fraction <= 0:
        raise SystemExit("flow refinement width and maximum fraction must be positive")
    if (
        args.foreground_refinement_width < 1
        or args.foreground_refinement_logit_scale <= 0
    ):
        raise SystemExit("foreground refinement width and logit scale must be positive")
    if (
        args.max_refractive_flow_fraction <= 0
        or args.residual_scale <= 0
        or args.alpha_refinement_scale <= 0
    ):
        raise SystemExit("flow, residual and alpha refinement scales must be positive")
    if args.inverse_min_reliable_weight < 0:
        raise SystemExit("inverse reliable weight must be non-negative")
    if args.inverse_variance_scale < 0:
        raise SystemExit("inverse variance scale must be non-negative")
    if args.diffusion_min_working_side < 0:
        raise SystemExit("diffusion minimum working side must be non-negative")
    if args.evidence_flux_only and args.stage in ("3", "4") and args.completion_variant != "diffusion":
        raise SystemExit("--evidence-flux-only Stage 3/4 requires --completion-variant diffusion")
    if (
        args.background_exclusion_dilation < 1
        or args.background_exclusion_dilation % 2 == 0
    ):
        raise SystemExit(
            "--background-exclusion-dilation must be a positive odd integer"
        )
    if args.refractive_kernel_radius_fraction < 0:
        raise SystemExit("--refractive-kernel-radius-fraction must be non-negative")
    if args.refractive_kernel_center_bias < 0:
        raise SystemExit("--refractive-kernel-center-bias must be non-negative")
    if (
        min(
            args.completion_width,
            args.completion_down_blocks,
            args.completion_residual_blocks,
            args.completion_max_channels,
        )
        < 1
    ):
        raise SystemExit("completion dimensions must be positive")
    if not 0.0 < args.completion_global_ratio < 1.0:
        raise SystemExit("--completion-global-ratio must lie strictly between 0 and 1")
    if args.completion_max_channels < args.completion_width:
        raise SystemExit(
            "--completion-max-channels must be at least --completion-width"
        )
    if args.checkpoint is not None and args.resume is not None:
        raise SystemExit("--checkpoint and --resume are mutually exclusive")
    if args.checkpoint_mam2_only and (
        args.stage != "2" or args.checkpoint is None or args.resume is not None
    ):
        raise SystemExit(
            "--checkpoint-mam2-only requires Stage 2 and a Stage 1B --checkpoint"
        )
    if args.stage not in {"1a", "1b"} and args.stage1_manifest:
        raise SystemExit("--stage1-manifest is valid only for Stage 1A/1B")
    if args.paired_backgrounds and args.batch_size < 2:
        raise SystemExit("--paired-backgrounds requires --batch-size >= 2")
    if args.paired_eval and args.batch_size < 2:
        raise SystemExit("--paired-eval requires --batch-size >= 2")
    if args.completion_variant == "diffusion" and args.diffusion_model is None:
        raise SystemExit("--diffusion-model is required for PRISM-Diffusion")
    if args.diffusion_backend == "objectclear" and args.completion_variant != "diffusion":
        raise SystemExit("ObjectClear requires --completion-variant diffusion")
    _validate_diffusion_training_mode(
        stage=args.stage,
        mode=args.mode,
        completion_variant=args.completion_variant,
        evidence_flux_only=args.evidence_flux_only,
    )
    if args.mam2_matter_backend == "external_mematte":
        if not all((args.mematte_root, args.mematte_config, args.mematte_checkpoint)):
            raise SystemExit(
                "external MEMatte requires --mematte-root, --mematte-config, "
                "and --mematte-checkpoint"
            )
        missing_mematte = [
            path
            for path in (
                args.mematte_root / "modeling" / "meta_arch" / "mematte.py",
                args.mematte_config,
                args.mematte_checkpoint,
            )
            if not path.is_file()
        ]
        if missing_mematte:
            raise SystemExit(
                "official MEMatte assets are missing: "
                + ", ".join(str(path) for path in missing_mematte)
                + f"; {mematte_setup_hint()}"
            )
    if args.mam2_matter_backend == "official_mam2":
        if args.official_mam2_root is None:
            raise SystemExit("official MAM2 requires --official-mam2-root")
        if args.stage in ("1a", "1b") or args.checkpoint_mam2_only:
            raise SystemExit("released pretrained MAM2 is supported for Stage 2/3/4 only")
        if not (args.official_mam2_root / "mam2" / "inference.py").is_file():
            raise SystemExit("--official-mam2-root is not the released MAM2 checkout")
    if args.mam2_matter_backend == "official_sam2matting":
        if args.official_sam2matting_root is None or args.official_sam2matting_checkpoint is None:
            raise SystemExit(
                "official SAM2Matting requires --official-sam2matting-root "
                "and --official-sam2matting-checkpoint"
            )
        if args.stage in ("1a", "1b") or args.checkpoint_mam2_only:
            raise SystemExit("released pretrained SAM2Matting is supported for Stage 2/3/4 only")
        if not (args.official_sam2matting_root / "sam2" / "sam2matting_video_predictor.py").is_file():
            raise SystemExit("--official-sam2matting-root is not the released checkout")
        if not args.official_sam2matting_checkpoint.is_file():
            raise SystemExit("--official-sam2matting-checkpoint does not exist")
    if args.mematte_max_tokens < 1:
        raise SystemExit("--mematte-max-tokens must be positive")
    data_paths = [
        path.resolve()
        for path in (args.train_data, args.val_data, args.test_data)
        if path is not None
    ]
    if len(data_paths) != len(set(data_paths)):
        raise SystemExit("train, validation, and test directories must be distinct")
    for value in (args.teacher_forcing_start, args.teacher_forcing_end):
        if not 0.0 <= value <= 1.0:
            raise SystemExit("teacher-forcing probabilities must be in [0,1]")
    if args.geometry_min_depth <= 0:
        raise SystemExit("--geometry-min-depth must be positive")
    if args.max_train_batches is not None and args.max_train_batches < 1:
        raise SystemExit("--max-train-batches must be positive")
    if args.max_eval_batches is not None and args.max_eval_batches < 1:
        raise SystemExit("--max-eval-batches must be positive")

    _seed_everything(args.seed)
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    torch.backends.cudnn.deterministic = True
    device = _device(args.device)
    if args.amp_dtype != "float32" and device.type != "cuda":
        raise SystemExit("mixed precision training currently requires CUDA")
    args.save_dir.mkdir(parents=True, exist_ok=True)

    evaluation_dataset = dict(
        clip_length=args.clip_length,
        frame_stride=args.frame_stride,
        strict_contract=args.strict_contract,
        verify_numeric_contract=args.runtime_contract_checks,
    )
    eval_workers = args.workers if args.eval_workers is None else args.eval_workers

    def prism_dataset(
        path: Path,
        *,
        split: str,
        training: bool = False,
        paired: bool = False,
    ):
        use_local_data = not args.archive_volume or (
            args.materialized_eval_data and not training
        )
        semantic_only = args.stage in {"1a", "1b"}
        load_geometry = args.predict_geometry and not semantic_only
        load_temporal_correspondence = (
            args.temporal_correspondence and args.stage in {"3", "4"}
        )
        if use_local_data:
            return RCTransPRISMDataset(
                path,
                **evaluation_dataset,
                semantic_only=semantic_only,
                load_geometry=load_geometry,
                load_temporal_correspondence=load_temporal_correspondence,
                random_temporal_crop=training,
                random_horizontal_flip=(
                    args.random_horizontal_flip if training else False
                ),
                augmentation_seed=args.seed,
            )
        manifest = path.parent / "dataset_manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(
                f"shard streaming requires the materialized metadata manifest: {manifest}"
            )
        return VesslShardCyclingDataset(
            dataset_manifest=manifest,
            split=split,
            storage_name=args.archive_storage_name,
            archive_volume=args.archive_volume,
            cache_root=args.archive_cache_root,
            paired_backgrounds=paired,
            pair_size=args.batch_size,
            shuffle_buffer=args.archive_shuffle_buffer,
            max_shards=args.archive_max_shards,
            download_retries=args.archive_download_retries,
            prefetch_next_shard=args.archive_prefetch,
            decode_workers=args.archive_decode_workers,
            semantic_only=semantic_only,
            load_geometry=load_geometry,
            load_temporal_correspondence=load_temporal_correspondence,
            random_temporal_crop=training,
            random_horizontal_flip=(args.random_horizontal_flip if training else False),
            augmentation_seed=args.seed,
            **evaluation_dataset,
        )

    test_loader = None
    if args.test_data is not None:
        test_dataset = prism_dataset(
            args.test_data,
            split="test",
            paired=args.paired_eval,
        )
        test_loader = _loader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            workers=eval_workers,
            paired_backgrounds=args.paired_eval,
            seed=args.seed,
        )
    val_loader = None
    if args.val_data is not None:
        val_dataset = prism_dataset(
            args.val_data, split="validation", paired=args.paired_eval
        )
        val_loader = _loader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            workers=eval_workers,
            paired_backgrounds=args.paired_eval,
            seed=args.seed,
        )
    train_loader = None
    if args.stage in {"1a", "1b"} and args.stage1_manifest:
        train_dataset = ManifestSemanticDataset(
            args.stage1_manifest,
            image_size=args.semantic_size,
            clip_length=args.clip_length,
            seed=args.seed,
            random_horizontal_flip=args.random_horizontal_flip,
            require_alpha=args.stage == "1b",
        )
        semantic_loader_options = dict(
            dataset=train_dataset,
            batch_size=args.batch_size,
            sampler=EpochShuffleSampler(train_dataset, shuffle=True, seed=args.seed),
            collate_fn=semantic_collate,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
        )
        if args.workers > 0:
            semantic_loader_options["prefetch_factor"] = 2
            semantic_loader_options["persistent_workers"] = False
        train_loader = DataLoader(**semantic_loader_options)
    elif args.train_data is not None:
        train_dataset = prism_dataset(
            args.train_data,
            split="train",
            training=True,
            paired=args.paired_backgrounds,
        )
        train_loader = _loader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            workers=args.workers,
            paired_backgrounds=args.paired_backgrounds,
            seed=args.seed,
        )

    if args.mam2_matter_backend == "official_sam2matting":
        from .official_sam2matting import OfficialSAM2MattingClipBackbone

        predictor = OfficialSAM2MattingClipBackbone.from_pretrained(
            args.official_sam2matting_root,
            args.official_sam2matting_checkpoint,
            variant=args.official_sam2matting_variant,
            device=device,
        )
        print("PRISM_OFFICIAL_SAM2MATTING_CHECKPOINT_LOADED", flush=True)
    elif args.mam2_matter_backend == "official_mam2":
        from .official_mam2 import OfficialMAM2ClipBackbone

        predictor = OfficialMAM2ClipBackbone.from_pretrained(
            args.official_mam2_root,
            config_path=args.official_mam2_config,
            device=device,
        )
        print("PRISM_OFFICIAL_MAM2_CHECKPOINTS_LOADED", flush=True)
    else:
        predictor = build_mam2_video_predictor(
            args.sam2_config,
            args.sam2_checkpoint,
            device=device,
            mode="train" if args.mode != "test" else "eval",
            integration_config=SAM2IntegrationConfig(
                matte=MAM2MatteConfig(
                    backend=args.mam2_matter_backend,
                    external_root=(
                        None if args.mematte_root is None else str(args.mematte_root)
                    ),
                    external_config=(
                        None if args.mematte_config is None else str(args.mematte_config)
                    ),
                    external_checkpoint=(
                        None
                        if args.mematte_checkpoint is None
                        else str(args.mematte_checkpoint)
                    ),
                    external_max_tokens=args.mematte_max_tokens,
                    external_patch_decoder=args.mematte_patch_decoder,
                    external_train_decoder=args.mematte_train_decoder,
                    external_train_backbone=args.mematte_train_backbone,
                    activation_checkpointing=args.activation_checkpointing,
                    full_activation_checkpointing=(
                        args.matte_full_activation_checkpointing
                    ),
                    frame_chunk_size=args.matte_frame_chunk_size,
                ),
                temporal_activation_checkpointing=(
                    args.sam2_temporal_activation_checkpointing
                ),
                temporal_checkpoint_chunk_size=(args.sam2_temporal_checkpoint_chunk_size),
                temporal_detach_interval=args.sam2_temporal_detach_interval,
            ),
        )
    background_config = BackgroundConfig(
        exclusion_dilation=args.background_exclusion_dilation,
        flux_direct_observation_threshold=args.flux_direct_observation_threshold,
        inverse_min_reliable_weight=args.inverse_min_reliable_weight,
        inverse_variance_scale=args.inverse_variance_scale,
        evidence_flux_only=args.evidence_flux_only,
        completion_backbone=args.completion_backbone,
        completion_width=args.completion_width,
        completion_down_blocks=args.completion_down_blocks,
        completion_residual_blocks=args.completion_residual_blocks,
        completion_global_ratio=args.completion_global_ratio,
        completion_max_channels=args.completion_max_channels,
        completion_variant=args.completion_variant,
        diffusion_model=args.diffusion_model,
        diffusion_adapter=args.diffusion_adapter,
        diffusion_revision=args.diffusion_revision,
        diffusion_prompt=args.diffusion_prompt,
        diffusion_negative_prompt=args.diffusion_negative_prompt,
        diffusion_inference_steps=args.diffusion_steps,
        diffusion_guidance_scale=args.diffusion_guidance_scale,
        diffusion_seed=args.diffusion_seed,
        diffusion_mask_dilation=args.diffusion_mask_dilation,
        diffusion_min_working_side=args.diffusion_min_working_side,
        diffusion_dtype=args.diffusion_dtype,
    )
    diffusion_completion = None
    if args.completion_variant == "diffusion":
        diffusion_device = args.diffusion_device or str(device)
        diffusion_offload = _diffusion_cpu_offload(
            args.diffusion_cpu_offload, torch.device(diffusion_device)
        )
        if args.diffusion_backend == "objectclear":
            from .objectclear_completion import (
                FlowGuidedObjectClearCompleter, ObjectClearSettings,
            )

            if args.diffusion_adapter is not None:
                raise SystemExit("ObjectClear does not accept --diffusion-adapter")
            if args.refinement_steps < 1 or not background_config.use_inverse_evidence:
                raise SystemExit("ObjectClear joint mode requires inverse evidence and >=1 refinement step")
            defaults = ObjectClearSettings()
            diffusion_completion = FlowGuidedObjectClearCompleter.from_pretrained(
                args.diffusion_model,
                source_root=args.objectclear_source_root,
                revision=args.diffusion_revision,
                settings=ObjectClearSettings(
                    prompt=(defaults.prompt if args.diffusion_prompt == BackgroundConfig.diffusion_prompt else args.diffusion_prompt),
                    inference_steps=(defaults.inference_steps if args.diffusion_steps == 25 else args.diffusion_steps),
                    guidance_scale=(defaults.guidance_scale if args.diffusion_guidance_scale == 7.5 else args.diffusion_guidance_scale),
                    seed=(defaults.seed if args.diffusion_seed == 0 else args.diffusion_seed),
                    mask_dilation=args.diffusion_mask_dilation,
                    min_working_side=max(defaults.min_working_side, args.diffusion_min_working_side),
                    use_fused_output=args.objectclear_fused_output,
                    late_denoise_steps=args.objectclear_late_denoise_steps,
                    pixel_refinement=args.objectclear_pixel_refinement,
                ),
                device=diffusion_device,
                dtype=args.objectclear_dtype,
                cpu_offload=diffusion_offload,
            )
            print("PRISM_OBJECTCLEAR_FLOW_LATENT_READY", flush=True)
        else:
            print(
                "PRISM_FLUX_MEMORY_MODE "
                + json.dumps({"cpu_offload": diffusion_offload, "device": diffusion_device}),
                flush=True,
            )
            diffusion_completion = FrozenDiffusionBackgroundCompleter.from_pretrained(
                args.diffusion_model,
                adapter=args.diffusion_adapter,
                revision=args.diffusion_revision,
                settings=DiffusionCompletionSettings(
                    prompt=args.diffusion_prompt,
                    negative_prompt=args.diffusion_negative_prompt,
                    inference_steps=args.diffusion_steps,
                    guidance_scale=args.diffusion_guidance_scale,
                    seed=args.diffusion_seed,
                    mask_dilation=args.diffusion_mask_dilation,
                    min_working_side=args.diffusion_min_working_side,
                ),
                device=diffusion_device,
                dtype=args.diffusion_dtype,
                cpu_offload=diffusion_offload,
            )
    config = PipelineConfig(
        background=background_config,
        matter=MatterConfig(
            width=args.matter_width,
            max_refractive_flow_fraction=args.max_refractive_flow_fraction,
            residual_scale=args.residual_scale,
            alpha_refinement_scale=args.alpha_refinement_scale,
            antialiased_boundary_support=args.antialiased_boundary_support,
            encoder_depths=tuple(args.matter_encoder_depths),
            max_channels=args.matter_max_channels,
            temporal_blocks=args.matter_temporal_blocks,
            refine_refractive_flow=args.flow_refinement,
            flow_refinement_width=args.flow_refinement_width,
            flow_refinement_max_fraction=args.flow_refinement_max_fraction,
            refine_straight_foreground=args.foreground_refinement,
            learned_object_support=args.learned_object_support,
            foreground_refinement_width=args.foreground_refinement_width,
            foreground_refinement_logit_scale=args.foreground_refinement_logit_scale,
            flow_guided_latent_bridge=args.flow_guided_latent_bridge,
            use_fba_foreground_prior=args.use_fba_foreground_prior,
            use_mam2_foreground_prior=args.use_mam2_foreground_prior,
            latent_bridge_width=args.latent_bridge_width,
            latent_bridge_flow_fraction=args.latent_bridge_flow_fraction,
            latent_bridge_foreground_logit_scale=(
                args.latent_bridge_foreground_logit_scale
            ),
            latent_bridge_transmission_logit_scale=(
                args.latent_bridge_transmission_logit_scale
            ),
            refractive_kernel_size=args.refractive_kernel_size,
            refractive_kernel_radius_fraction=(args.refractive_kernel_radius_fraction),
            refractive_kernel_center_bias=args.refractive_kernel_center_bias,
            predict_geometry=args.predict_geometry,
            geometry_min_depth=args.geometry_min_depth,
        ),
        joint_refinement_steps=args.refinement_steps,
    )
    if args.ablation in ("no_inverse", "direct_only"):
        config.background.use_inverse_evidence = False
    if args.ablation == "no_joint_refinement":
        config.joint_refinement_steps = 0
    if args.ablation == "detached_paths":
        config.detach_semantics_for_background = True
        config.detach_background_for_matter = True
    if args.ablation == "scalar_transmission":
        config.matter.use_rgb_transmission = False
    if args.ablation == "no_residual":
        config.matter.use_residual = False
    if args.ablation in ("no_completion", "direct_only"):
        config.background.use_temporal_completion = False
    background_model = MaskedTemporalBackground(
        config.background,
        diffusion_completion=diffusion_completion,
    )
    foreground_prior_model = None
    if args.use_fba_foreground_prior:
        if not args.flow_guided_latent_bridge:
            raise ValueError("FBA foreground prior requires --flow-guided-latent-bridge")
        if args.fba_source_root is None or args.fba_checkpoint is None:
            raise ValueError("FBA prior requires --fba-source-root and --fba-checkpoint")
        foreground_prior_model = FBAPretrainedForegroundPrior(
            args.fba_source_root,
            args.fba_checkpoint,
            input_side=args.fba_input_side,
            train_decoder=args.fba_train_decoder,
        )
    pipeline = RefractiveMAM2(
        predictor,
        config,
        background_model=background_model,
        foreground_prior_model=foreground_prior_model,
    ).to(device)
    initial_checkpoint = args.resume or args.checkpoint
    if initial_checkpoint is not None:
        if args.checkpoint_mam2_only:
            load_stage1_mam2_checkpoint(initial_checkpoint, predictor)
            print(
                f"PRISM_STAGE1_MAM2_TRANSFERRED checkpoint={initial_checkpoint} "
                "pam=fresh background=fresh",
                flush=True,
            )
        else:
            load_refractive_checkpoint(initial_checkpoint, predictor, pipeline)

    logger = WandbLogger(
        args.wandb_project or args.project_name,
        args.wandb_run_name or f"PRISM-stage{args.stage}-{args.mode}",
        _wandb_config(args),
        mode=args.wandb_mode,
        entity=args.wandb_entity,
        group=args.wandb_group,
        tags=args.wandb_tags,
        notes=args.wandb_notes,
    )
    global_step = 0
    start_epoch = 0
    try:
        if args.mode in {"train", "both"}:
            assert train_loader is not None
            assert val_loader is not None
            parameters = _configure_stage(args.stage, predictor, pipeline)
            if not parameters:
                raise RuntimeError(f"stage {args.stage} has no trainable parameters")
            profile_loss_weights = _loss_weights_for_profile(args.loss_profile)
            optimizer = torch.optim.AdamW(
                _optimizer_groups(
                    args.stage,
                    parameters,
                    predictor,
                    learning_rate=args.lr,
                    joint_mam2_lr_scale=args.joint_mam2_lr_scale,
                    foreground_prior_model=foreground_prior_model,
                    fba_decoder_lr_scale=args.fba_decoder_lr_scale,
                ),
                lr=args.lr,
                weight_decay=args.weight_decay,
                fused=device.type == "cuda",
            )
            batches_per_epoch = len(train_loader)
            if args.max_train_batches is not None:
                batches_per_epoch = min(batches_per_epoch, args.max_train_batches)
            total_steps = max(batches_per_epoch * args.epochs, 1)
            scheduler = (
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=total_steps,
                )
                if args.scheduler == "cosine"
                else None
            )
            default_metric, default_mode = _default_checkpoint_selection(
                args.stage, evidence_flux_only=args.evidence_flux_only,
                learned_object_support=args.learned_object_support,
            )
            selection_metric = args.selection_metric or default_metric
            selection_mode = args.selection_mode or default_mode
            best_value = float("-inf") if selection_mode == "max" else float("inf")
            start_batch_in_epoch = 0
            if args.resume is not None:
                (
                    start_epoch,
                    start_batch_in_epoch,
                    global_step,
                    best_value,
                ) = _restore_training_state(
                    load_refractive_training_state(args.resume),
                    optimizer,
                    scheduler,
                )
            local_best_checkpoint = args.save_dir / f"prism_stage{args.stage}_best.pt"
            # A resumed run may intentionally write to another directory. Until
            # the new run improves the validation score, the historical resume
            # checkpoint is the only valid best model we can test.
            best_checkpoint = (
                local_best_checkpoint
                if local_best_checkpoint.is_file()
                else args.resume if args.resume is not None else local_best_checkpoint
            )
            resume_checkpoint = args.save_dir / f"prism_stage{args.stage}_resume.pt"
            for epoch in range(start_epoch, args.epochs):
                dataset = getattr(train_loader, "dataset", None)
                if hasattr(dataset, "set_epoch"):
                    dataset.set_epoch(epoch)
                sampler = getattr(train_loader, "batch_sampler", None)
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)
                sample_sampler = getattr(train_loader, "sampler", None)
                if hasattr(sample_sampler, "set_epoch"):
                    sample_sampler.set_epoch(epoch)
                _configure_stage(args.stage, predictor, pipeline)
                for batch_index, raw_batch in enumerate(train_loader):
                    if (
                        args.max_train_batches is not None
                        and batch_index >= args.max_train_batches
                    ):
                        break
                    if epoch == start_epoch and batch_index < start_batch_in_epoch:
                        continue
                    batch = raw_batch.to(device, non_blocking=True)
                    progress = global_step / max(total_steps - 1, 1)
                    probability = (
                        args.teacher_forcing_start * (1.0 - progress)
                        + args.teacher_forcing_end * progress
                    )
                    if args.optical_curriculum and args.stage == "3":
                        probability = _optical_curriculum_probability(epoch, args.epochs)
                    teacher_forcing = _training_teacher_forcing(
                        args.stage,
                        probability,
                        evidence_flux_only=args.evidence_flux_only,
                        optical_curriculum=args.optical_curriculum,
                    )
                    oracle_background_scale = (
                        _stage4_oracle_background_scale(progress)
                        if args.stage == "4" and args.evidence_flux_only else 1.0
                    )
                    optimizer.zero_grad(set_to_none=True)
                    with _autocast_context(device, args.amp_dtype):
                        if args.stage in {"1a", "1b"}:
                            semantics = _semantic_forward(
                                predictor,
                                batch.ground_truth,
                                prompt_mode=args.prompt_mode,
                                prompt_seed=args.prompt_seed + epoch,
                                prompt_jitter_pixels=args.prompt_jitter_pixels,
                                frames_are_linear=not hasattr(batch, "dataset_kinds"),
                                compute_alpha=args.stage != "1a",
                            )
                            losses = _semantic_loss(
                                semantics,
                                batch.ground_truth,
                                getattr(batch, "dataset_kinds", None),
                                stage=args.stage,
                            )
                        else:
                            use_paired_microbatch = (
                                args.stage in {"3", "4"}
                                and args.paired_backgrounds
                                and args.paired_microbatch_checkpointing
                                and batch.ground_truth.frames.shape[0] > 1
                            )
                            predict_function = (
                                _checkpointed_paired_predict
                                if use_paired_microbatch
                                else _predict
                            )
                            prediction = predict_function(
                                predictor,
                                pipeline,
                                batch.ground_truth,
                                teacher_forcing=teacher_forcing,
                                prompt_mode=args.prompt_mode,
                                prompt_seed=args.prompt_seed + epoch,
                                prompt_jitter_pixels=args.prompt_jitter_pixels,
                                skip_background_estimation=args.stage == "2",
                            )
                            losses = _loss_terms(
                                args.stage, prediction, batch,
                                loss_profile=args.loss_profile,
                                oracle_background_scale=oracle_background_scale,
                            )
                    if not bool(torch.isfinite(losses["total"])):
                        raise FloatingPointError(
                            f"non-finite training loss at step {global_step}: "
                            f"{float(losses['total'].detach().cpu())}"
                        )
                    losses["total"].backward()
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        parameters,
                        args.gradient_clip,
                        error_if_nonfinite=True,
                    )
                    optimizer.step()
                    if (
                        args.runtime_contract_checks
                        and args.stage not in {"1a", "1b"}
                        and _should_check_parameter_finiteness(
                            global_step, args.parameter_finiteness_interval_steps
                        )
                    ):
                        matter_parameters = tuple(pipeline.matter.named_parameters())
                        finite_parameters = torch.stack(
                            [torch.isfinite(parameter).all() for _, parameter in matter_parameters]
                        )
                        if not bool(finite_parameters.all()):
                            invalid_names = [
                                name
                                for (name, _), valid in zip(
                                    matter_parameters, finite_parameters.tolist()
                                )
                                if not valid
                            ]
                            raise FloatingPointError(
                                "PAM parameters became NaN or Inf after optimizer step "
                                f"{global_step}: {invalid_names[:8]}"
                            )
                    if scheduler is not None:
                        scheduler.step()
                    log_train_images = (
                        logger.enabled
                        and args.wandb_image_interval > 0
                        and args.wandb_image_limit > 0
                        and global_step % args.wandb_image_interval == 0
                    )
                    log_train_scalars = logger.enabled and (
                        global_step == 0
                        or (global_step + 1) % args.wandb_scalar_interval_steps == 0
                        or log_train_images
                    )
                    if log_train_scalars:
                        logger.log(
                            {
                                **{f"train/{key}": value for key, value in losses.items()},
                                **(
                                    {
                                        f"train/weighted_loss/{key}": (
                                            getattr(profile_loss_weights, key) * value
                                            * (
                                                oracle_background_scale
                                                if key == "render_oracle_background"
                                                else 1.0
                                            )
                                        )
                                        for key, value in losses.items()
                                        if key != "total"
                                        and hasattr(profile_loss_weights, key)
                                    }
                                    if args.stage not in {"1a", "1b"} else {}
                                ),
                                **{
                                    f"train/{key}": value
                                    for key, value in _alpha_gt_statistics(
                                        batch.ground_truth
                                    ).items()
                                },
                                "train/gradient_norm": gradient_norm,
                                **(
                                    _operator_head_gradient_metrics(pipeline)
                                    if args.stage not in {"1a", "1b"} else {}
                                ),
                                "train/teacher_forcing": float(teacher_forcing),
                                "train/oracle_background_loss_scale": oracle_background_scale,
                                "train/pam_oracle_background": float(args.stage == "2"),
                                "train/epoch": epoch + 1,
                                "train/learning_rate": optimizer.param_groups[0]["lr"],
                                **(
                                    {
                                        "train/mam2_learning_rate": group["lr"]
                                        for group in optimizer.param_groups
                                        if group.get("name") == "mam2"
                                    }
                                    if args.stage == "4"
                                    else {}
                                ),
                                **{
                                    "train/fba_decoder_learning_rate": group["lr"]
                                    for group in optimizer.param_groups
                                    if group.get("name") == "fba_decoder"
                                },
                                **(
                                    {
                                        "train/flux_image_cache_hit_rate": (
                                            diffusion_completion.image_cache_hits
                                            / max(
                                                diffusion_completion.image_cache_hits
                                                + diffusion_completion.image_cache_misses,
                                                1,
                                            )
                                        ),
                                        "train/flux_image_cache_entries": (
                                            diffusion_completion.image_cache_size
                                        ),
                                    }
                                    if diffusion_completion is not None else {}
                                ),
                            },
                            global_step,
                            commit=not log_train_images,
                        )
                    if args.progress_interval_steps > 0 and (
                        global_step == 0
                        or (global_step + 1) % args.progress_interval_steps == 0
                    ):
                        print(
                            "PRISM_TRAIN_PROGRESS "
                            + json.dumps(
                                {
                                    "stage": args.stage,
                                    "epoch": epoch + 1,
                                    "batch": batch_index + 1,
                                    "step": global_step + 1,
                                    "loss": float(losses["total"].detach()),
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                    if log_train_images:
                        sample_ids = getattr(
                            batch,
                            "sequence_ids",
                            [
                                f"train_{global_step}_{index}"
                                for index in range(batch.ground_truth.frames.shape[0])
                            ],
                        )
                        if args.stage in {"1a", "1b"}:
                            montages = _semantic_montages(
                                semantics,
                                batch.ground_truth,
                                sample_ids,
                                limit=args.wandb_image_limit,
                                stage=args.stage,
                            )
                        else:
                            montages = _supervised_prediction_montages(
                                prediction,
                                batch.ground_truth,
                                sample_ids,
                                limit=args.wandb_image_limit,
                                stage=args.stage,
                                teacher_forcing=teacher_forcing,
                            )
                        alpha_values = _alpha_gt_values(batch.ground_truth)
                        if alpha_values is not None and alpha_values.numel() > 0:
                            logger.log_histograms(
                                {"train/alpha_gt_distribution": alpha_values},
                                global_step,
                                commit=False,
                            )
                        logger.log_images(
                            "train/qualitative",
                            [image for image, _ in montages],
                            global_step,
                            captions=[caption for _, caption in montages],
                        )
                    global_step += 1
                    if (
                        args.checkpoint_interval_steps > 0
                        and global_step % args.checkpoint_interval_steps == 0
                    ):
                        save_refractive_checkpoint(
                            resume_checkpoint,
                            predictor,
                            pipeline,
                            metadata={
                                "epoch": epoch,
                                "batch_in_epoch": batch_index + 1,
                                "global_step": global_step,
                                "kind": "periodic_resume",
                            },
                            training_state=_training_state(
                                optimizer,
                                scheduler,
                                epoch=epoch,
                                batch_in_epoch=batch_index + 1,
                                global_step=global_step,
                                best_value=best_value,
                            ),
                        )
                    predictor.clear_mam2_cache()
                    del losses, gradient_norm, batch, raw_batch
                    if args.stage in {"1a", "1b"}:
                        del semantics
                    else:
                        del prediction

                start_batch_in_epoch = 0

                should_validate = _should_validate_epoch(
                    epoch, args.epochs, args.validate_every_epochs
                )
                validation: dict[str, float] = {}
                improved = False
                if should_validate:
                    validation = evaluate(
                        predictor,
                        pipeline,
                        val_loader,
                        device,
                        stage=args.stage,
                        prompt_mode=args.prompt_mode,
                        prompt_seed=args.prompt_seed,
                        wandb_logger=logger,
                        wandb_prefix="validation",
                        wandb_step=global_step,
                        wandb_image_limit=args.wandb_image_limit,
                        amp_dtype=args.amp_dtype,
                        max_batches=args.max_eval_batches,
                        loss_profile=args.loss_profile,
                    )
                    logger.log(
                        {f"validation/{k}": v for k, v in validation.items()},
                        global_step,
                    )
                    if selection_metric not in validation:
                        raise RuntimeError(
                            f"selection metric {selection_metric!r} is absent; "
                            f"available={sorted(validation)}"
                        )
                    current_value = validation[selection_metric]
                    improved = (
                        current_value > best_value
                        if selection_mode == "max"
                        else current_value < best_value
                    )
                    if improved:
                        best_value = current_value
                checkpoint = (
                    args.save_dir / f"prism_stage{args.stage}_epoch{epoch + 1:03d}.pt"
                )
                if args.keep_epoch_checkpoints:
                    save_refractive_checkpoint(
                        checkpoint,
                        predictor,
                        pipeline,
                        metadata={
                            "epoch": epoch + 1,
                            "global_step": global_step,
                            "validation": validation,
                            "arguments": {
                                key: str(value) if isinstance(value, Path) else value
                                for key, value in vars(args).items()
                            },
                        },
                        training_state=_training_state(
                            optimizer,
                            scheduler,
                            epoch=epoch + 1,
                            global_step=global_step,
                            best_value=best_value,
                        ),
                    )
                save_refractive_checkpoint(
                    resume_checkpoint,
                    predictor,
                    pipeline,
                    metadata={
                        "epoch": epoch + 1,
                        "batch_in_epoch": 0,
                        "global_step": global_step,
                        "kind": "epoch_resume",
                    },
                    training_state=_training_state(
                        optimizer,
                        scheduler,
                        epoch=epoch + 1,
                        batch_in_epoch=0,
                        global_step=global_step,
                        best_value=best_value,
                    ),
                )
                if improved:
                    best_checkpoint = local_best_checkpoint
                    save_refractive_checkpoint(
                        best_checkpoint,
                        predictor,
                        pipeline,
                        metadata={
                            "epoch": epoch + 1,
                            "global_step": global_step,
                            "validation": validation,
                            "selection_metric": selection_metric,
                            "selection_mode": selection_mode,
                            "selection_value": best_value,
                            "arguments": {
                                key: str(value) if isinstance(value, Path) else value
                                for key, value in vars(args).items()
                            },
                        },
                        # Optimizer/RNG state lives in the atomic resume file.
                        # Keeping best checkpoints model-only avoids duplicating
                        # several gigabytes at every stage boundary.
                        training_state=None,
                    )
                print(
                    json.dumps(
                        {"epoch": epoch + 1, "validated": should_validate, **validation},
                        sort_keys=True,
                    )
                )
            args.checkpoint = best_checkpoint
            load_refractive_checkpoint(best_checkpoint, predictor, pipeline)

        if args.mode in {"test", "both"}:
            assert test_loader is not None
            if args.mode == "test" and args.checkpoint is None:
                raise SystemExit("--checkpoint is required in test mode")
            metrics = evaluate(
                predictor,
                pipeline,
                test_loader,
                device,
                stage=args.stage,
                prompt_mode=args.prompt_mode,
                prompt_seed=args.prompt_seed,
                qualitative_dir=args.save_dir / "qualitative",
                qualitative_limit=args.qualitative_limit,
                compute_lpips=args.compute_lpips,
                compute_connectivity=args.compute_connectivity,
                wandb_logger=logger,
                wandb_prefix="test",
                wandb_step=global_step,
                wandb_image_limit=args.wandb_image_limit,
                amp_dtype=args.amp_dtype,
                max_batches=args.max_eval_batches,
                loss_profile=args.loss_profile,
            )
            if args.prompt_robustness_runs > 0 and args.prompt_jitter_pixels > 0:
                robustness: list[dict[str, float]] = []
                for run_index in range(args.prompt_robustness_runs):
                    robustness.append(
                        evaluate(
                            predictor,
                            pipeline,
                            test_loader,
                            device,
                            stage=args.stage,
                            prompt_mode=args.prompt_mode,
                            prompt_seed=args.prompt_seed + run_index + 1,
                            prompt_jitter_pixels=args.prompt_jitter_pixels,
                            compute_connectivity=args.compute_connectivity,
                            amp_dtype=args.amp_dtype,
                            max_batches=args.max_eval_batches,
                            loss_profile=args.loss_profile,
                        )
                    )
                for name in sorted(set.intersection(*(set(run) for run in robustness))):
                    values = torch.tensor([run[name] for run in robustness])
                    metrics[f"prompt_robust/{name}_mean"] = float(values.mean())
                    metrics[f"prompt_robust/{name}_std"] = float(
                        values.std(unbiased=False)
                    )
            metrics_path = args.save_dir / f"prism_stage{args.stage}_metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "checkpoint": str(args.checkpoint),
                        "dataset": str(args.test_data),
                        "pipeline_config": asdict(pipeline.config),
                        "metrics": metrics,
                        "reproducibility": _reproducibility_record(args),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            logger.log(
                {f"test/{key}": value for key, value in metrics.items()}, global_step
            )
            print(json.dumps(metrics, indent=2, sort_keys=True))
            print(f"metrics written to {metrics_path}")
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
