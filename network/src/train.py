"""Reproducible stage-wise training and evaluation entry point for PRISM."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from .background import MaskedTemporalBackground
from .completion import (
    DiffusionCompletionSettings,
    FrozenDiffusionBackgroundCompleter,
)
from .config import BackgroundConfig, MatterConfig, PipelineConfig
from .dataset import (
    RCTransBatch,
    RCTransPRISMDataset,
    build_paired_prism_dataloader,
    prism_collate,
)
from .logger import WandbLogger
from .losses import RefractiveGroundTruth, reusable_operator_consistency_in_batch
from .pipeline import RefractiveMAM2, RefractiveMAM2Output
from .renderer import recompose
from .runner import (
    load_refractive_checkpoint,
    load_refractive_training_state,
    save_refractive_checkpoint,
)
from .sam2_integration import MAM2VideoPredictor, build_mam2_video_predictor
from .semantic_dataset import ManifestSemanticDataset, semantic_collate
from .training import (
    SemanticTargets,
    configure_joint,
    configure_stage1,
    configure_stage2,
    joint_stage_loss,
    normalize_sam2_training_frames,
    physics_stage_loss,
    selective_semantic_loss,
)
from .types import MAM2BackboneOutput
from .vendor import SAM2_CHECKPOINT, SAM2_CONFIG


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _training_state(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    *,
    epoch: int,
    global_step: int,
    best_value: float,
) -> dict[str, object]:
    return {
        "epoch": epoch,
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
) -> tuple[int, int, float]:
    if not state:
        raise RuntimeError("resume checkpoint contains no training state")
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    random.setstate(state["python_random_state"])
    torch.random.set_rng_state(state["torch_rng_state"])
    cuda_state = state.get("cuda_rng_state", [])
    if torch.cuda.is_available() and cuda_state:
        torch.cuda.set_rng_state_all(cuda_state)
    return (
        int(state["epoch"]),
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
    paths = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
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
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "seed": args.seed,
        "prompt_mode": args.prompt_mode,
        "checkpoint_sha256": _sha256_path(args.checkpoint),
        "sam2_checkpoint_sha256": _sha256_path(args.sam2_checkpoint),
        "dataset_artifact_manifest_sha256": _sha256_path(manifest),
        "diffusion_model": args.diffusion_model,
        "diffusion_revision": args.diffusion_revision,
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
    array = (
        value.permute(1, 2, 0).mul(255).round().byte().cpu().numpy()
    )
    return Image.fromarray(np.asarray(array), mode="RGB")


def _preview_mask(value: Tensor):
    value = value.detach().float().clamp(0, 1)
    if value.ndim == 3:
        value = value[0]
    return _preview_rgb(value.unsqueeze(0).expand(3, -1, -1))


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


def _preview_flow(value: Tensor):
    """Encode dx, dy, and magnitude for a dependency-free W&B preview."""

    if value.ndim != 3 or value.shape[0] != 2:
        raise ValueError("flow preview requires [2,H,W]")
    flow = value.detach().float()
    magnitude = flow.square().sum(dim=0).sqrt()
    scale = torch.quantile(magnitude.flatten(), 0.95).clamp_min(1e-6)
    dx = (0.5 + 0.5 * flow[0] / scale).clamp(0, 1)
    dy = (0.5 + 0.5 * flow[1] / scale).clamp(0, 1)
    return _preview_rgb(torch.stack((dx, dy, (magnitude / scale).clamp(0, 1))))


def _labeled_grid(panels, *, columns: int = 4):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:  # pragma: no cover
        raise ImportError("qualitative output requires Pillow") from exc
    if not panels:
        raise ValueError("at least one panel is required")
    width, height = panels[0][1].size
    title_height = 20
    rows = math.ceil(len(panels) / columns)
    montage = Image.new("RGB", (columns * width, rows * (height + title_height)), "black")
    draw = ImageDraw.Draw(montage)
    font = ImageFont.load_default()
    for index, (label, panel) in enumerate(panels):
        if panel.size != (width, height):
            panel = panel.resize((width, height))
        x = (index % columns) * width
        y = (index // columns) * (height + title_height)
        montage.paste(panel, (x, y + title_height))
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
            prediction.reconstructed_frames[index, 0] - target.frames[index, 0]
        ).abs().mul(4.0).clamp(0, 1)
        panels = [
            ("input I", _preview_rgb(target.frames[index, 0])),
            ("reconstruction", _preview_rgb(prediction.reconstructed_frames[index, 0])),
            ("|I-I_hat| x4", _preview_rgb(error)),
            ("background pred", _preview_rgb(prediction.background.background[index])),
            ("evidence bg", _preview_rgb(prediction.background.evidence_background[index])),
            ("alpha pred", _preview_mask(prediction.matter.alpha[index, 0])),
            ("direct support", _preview_mask(prediction.background.direct_support[index])),
            ("inverse support", _preview_mask(prediction.background.inverse_support[index])),
            ("true hole", _preview_mask(prediction.background.true_hole[index])),
            ("flow pred", _preview_flow(prediction.matter.refractive_flow[index, 0])),
        ]
        if background_gt is not None:
            panels.insert(5, ("background GT", _preview_rgb(background_gt)))
        if target.alpha is not None:
            panels.insert(7, ("alpha GT", _preview_mask(target.alpha[index, 0])))
        caption = sample_ids[index] if index < len(sample_ids) else f"sample_{index}"
        output.append((_labeled_grid(panels), caption))
    return output


def _semantic_montages(
    semantics: MAM2BackboneOutput,
    target: RefractiveGroundTruth,
    sample_ids: list[str],
    *,
    limit: int,
):
    count = min(limit, semantics.mask_logits.shape[0])
    size = target.frames.shape[-2:]
    masks = F.interpolate(
        semantics.mask_logits[:, 0], size=size, mode="bilinear", align_corners=False
    ).sigmoid()
    trimaps = F.interpolate(
        semantics.trimap_logits[:, 0], size=size, mode="bilinear", align_corners=False
    ).argmax(dim=1)
    output = []
    for index in range(count):
        panels = [
            ("input I", _preview_rgb(target.frames[index, 0])),
            ("mask pred", _preview_mask(masks[index])),
            ("trimap pred", _preview_trimap(trimaps[index])),
        ]
        if target.object_mask is not None:
            panels.append(("mask GT", _preview_mask(target.object_mask[index, 0])))
        if target.trimap is not None:
            panels.append(("trimap GT", _preview_trimap(target.trimap[index, 0])))
        caption = sample_ids[index] if index < len(sample_ids) else f"sample_{index}"
        output.append((_labeled_grid(panels), caption))
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
        torch.save(
            {
                "background": prediction.background.background[index].detach().cpu(),
                "evidence_background": prediction.background.evidence_background[index].detach().cpu(),
                "direct_support": prediction.background.direct_support[index].detach().cpu(),
                "inverse_support": prediction.background.inverse_support[index].detach().cpu(),
                "true_hole": prediction.background.true_hole[index].detach().cpu(),
                "alpha": prediction.matter.alpha[index].detach().cpu(),
                "premultiplied_foreground": prediction.matter.premultiplied_foreground[index].detach().cpu(),
                "transmittance": prediction.matter.transmittance[index].detach().cpu(),
                "refractive_flow": prediction.matter.refractive_flow[index].detach().cpu(),
                "residual": prediction.matter.residual[index].detach().cpu(),
                "reconstruction": prediction.reconstructed_frames[index].detach().cpu(),
            },
            sample_dir / "prediction.pt",
        )
    return count


def _semantic_forward(
    predictor: MAM2VideoPredictor,
    target: RefractiveGroundTruth,
    *,
    prompt_mode: str,
    prompt_seed: int = 0,
    prompt_jitter_pixels: float = 0.0,
) -> MAM2BackboneOutput:
    """Run differentiable SAM2/PDD/MSS with a reproducible prompt protocol."""

    if target.object_mask is None:
        raise ValueError("RCTrans training requires object_mask for the first-frame prompt")
    size = (predictor.image_size, predictor.image_size)
    batch, frames, _, height, width = target.frames.shape
    resized = F.interpolate(
        target.frames.reshape(batch * frames, 3, height, width),
        size=size,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).reshape(batch, frames, 3, *size)
    normalized = normalize_sam2_training_frames(resized.clamp(0.0, 1.0))
    first_mask = target.object_mask[:, 0]
    if prompt_mode == "mask":
        return predictor.forward_mam2_clip(
            normalized,
            first_frame_mask_inputs=first_mask,
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
            labels.append(
                torch.tensor((2, 3), dtype=torch.int32, device=mask.device)
            )
    coords = torch.stack(coordinates).to(device=mask.device, dtype=torch.float32)
    scale = coords.new_tensor(
        (size / max(width, 1), size / max(height, 1))
    ).view(1, 1, 2)
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
    )


def _semantic_loss(
    semantics: MAM2BackboneOutput,
    target: RefractiveGroundTruth,
    dataset_kinds: list[str] | None = None,
) -> dict[str, Tensor]:
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
                non_memory_features=semantics.non_memory_features.index_select(0, indices),
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
            )
            selected_terms = selective_semantic_loss(
                selected.mask_logits,
                selected.trimap_logits,
                selected_target,
                dataset_kind=kind,
            )
            for name, value in selected_terms.items():
                if name != "total":
                    terms[name] = terms.get(name, value.new_zeros(())) + value
        terms["total"] = sum(terms.values(), semantics.mask_logits.new_zeros(()))
        return terms
    return selective_semantic_loss(
        semantics.mask_logits,
        semantics.trimap_logits,
        SemanticTargets(object_mask=target.object_mask, trimap=target.trimap),
        dataset_kind="synthetic_physics",
    )


def _semantic_metrics(
    semantics: MAM2BackboneOutput,
    target: RefractiveGroundTruth,
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
    return metrics


def _loss_terms(
    stage: int,
    prediction: RefractiveMAM2Output,
    batch: RCTransBatch,
) -> dict[str, Tensor]:
    target = batch.ground_truth
    if stage == 1:
        return _semantic_loss(prediction.backbone, target)
    if stage == 2:
        return physics_stage_loss(prediction, target)
    paired_ids = batch.paired_background_group_ids
    has_pair = len(paired_ids) != len(set(paired_ids))
    support = None
    if has_pair and target.object_mask is not None:
        support = target.object_mask
        if target.refractive_validity is not None:
            support = support * target.refractive_validity
    return joint_stage_loss(
        prediction,
        target,
        paired_background_group_ids=paired_ids if has_pair else None,
        operator_support=support,
    )


def _batch_metrics(
    prediction: RefractiveMAM2Output,
    target: RefractiveGroundTruth,
    *,
    compute_lpips: bool = False,
) -> dict[str, tuple[float, int]]:
    """Return sums and denominators so metrics are batch-size independent."""

    metrics: dict[str, tuple[float, int]] = {}

    def add(name: str, value: Tensor, count: int) -> None:
        metrics[name] = (float(value.detach().sum().cpu()), count)

    batch, frames = target.frames.shape[:2]
    operator_support = target.object_mask

    def operator_mae(predicted: Tensor, truth: Tensor) -> Tensor:
        error = (predicted - truth).abs().mean(dim=2)
        if operator_support is None:
            return error.mean(dim=(2, 3))
        weight = operator_support.squeeze(2).to(error.dtype)
        return (error * weight).sum(dim=(2, 3)) / weight.sum(
            dim=(2, 3)
        ).clamp_min(1.0)

    render_mse = (prediction.reconstructed_frames - target.frames).square().mean(
        dim=(2, 3, 4)
    )
    add("render_mse", render_mse, batch * frames)
    add("render_psnr", -10.0 * torch.log10(render_mse.clamp_min(1e-12)), batch * frames)

    if target.alpha is not None:
        alpha_error = prediction.matter.alpha - target.alpha
        add("alpha_mse", alpha_error.square().mean(dim=(2, 3, 4)), batch * frames)
        add("alpha_sad", alpha_error.abs().sum(dim=(2, 3, 4)) / 1000.0, batch * frames)
        pred_dx = prediction.matter.alpha[..., :, 1:] - prediction.matter.alpha[..., :, :-1]
        true_dx = target.alpha[..., :, 1:] - target.alpha[..., :, :-1]
        pred_dy = prediction.matter.alpha[..., 1:, :] - prediction.matter.alpha[..., :-1, :]
        true_dy = target.alpha[..., 1:, :] - target.alpha[..., :-1, :]
        gradient_mse = 0.5 * (
            (pred_dx - true_dx).square().mean(dim=(2, 3, 4))
            + (pred_dy - true_dy).square().mean(dim=(2, 3, 4))
        )
        add("alpha_gradient_mse", gradient_mse, batch * frames)
        boundary_f1 = _alpha_boundary_f1(prediction.matter.alpha, target.alpha)
        add("alpha_boundary_f1", boundary_f1, batch * frames)
        add(
            "alpha_connectivity_error",
            _alpha_connectivity_error(prediction.matter.alpha, target.alpha),
            batch * frames,
        )
    if target.counterfactual_background is not None:
        background = target.counterfactual_background
        if background.ndim == 5:
            background = background[:, 0]
        bg_mse = (prediction.background.background - background).square().mean(
            dim=(1, 2, 3)
        )
        add("background_mse", bg_mse, batch)
        add("background_psnr", -10.0 * torch.log10(bg_mse.clamp_min(1e-12)), batch)
        add("background_ssim", _ssim(prediction.background.background, background), batch)
        if compute_lpips:
            add(
                "background_lpips",
                _lpips(prediction.background.background, background),
                batch,
            )
        direct = prediction.background.direct_support
        inverse = prediction.background.inverse_support & ~direct
        true_hole = prediction.background.true_hole
        for name, mask in (
            ("direct", direct),
            ("inverse_only", inverse),
            ("true_hole", true_hole),
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
        add(
            "background_true_hole_fraction",
            true_hole.float().mean(dim=(1, 2, 3)),
            batch,
        )
        supported = ~true_hole
        preservation = (
            prediction.background.background
            - prediction.background.evidence_background
        ).abs() * supported
        add(
            "evidence_preservation_l1",
            preservation.sum(dim=(1, 2, 3))
            / (supported.sum(dim=(1, 2, 3)).clamp_min(1) * 3),
            batch,
        )
    if target.refractive_flow is not None:
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
            add(name, operator_mae(predicted, truth), batch * frames)
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
    var_pred = F.avg_pool2d(prediction.square(), kernel, 1, padding) - mean_pred.square()
    var_true = F.avg_pool2d(target.square(), kernel, 1, padding) - mean_true.square()
    covariance = F.avg_pool2d(prediction * target, kernel, 1, padding) - mean_pred * mean_true
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mean_pred * mean_true + c1) * (2 * covariance + c2)) / (
        (mean_pred.square() + mean_true.square() + c1)
        * (var_pred + var_true + c2)
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
        raise ImportError("connectivity evaluation requires refractive-mam2[data]") from exc
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
    stage: int,
    prompt_mode: str,
    prompt_seed: int = 0,
    prompt_jitter_pixels: float = 0.0,
    qualitative_dir: Path | None = None,
    qualitative_limit: int = 0,
    compute_lpips: bool = False,
    wandb_logger: WandbLogger | None = None,
    wandb_prefix: str | None = None,
    wandb_step: int = 0,
    wandb_image_limit: int = 0,
) -> dict[str, float]:
    predictor.eval()
    pipeline.eval()
    totals: dict[str, list[float]] = {}
    batches = 0
    qualitative_saved = 0
    wandb_images_logged = False
    runtime_seconds = 0.0
    runtime_sequences = 0
    runtime_frames = 0
    peak_memory_mb = 0.0
    with torch.inference_mode():
        for raw_batch in dataloader:
            batch = raw_batch.to(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            start_time = time.perf_counter()
            if stage == 1:
                semantics = _semantic_forward(
                    predictor,
                    batch.ground_truth,
                    prompt_mode=prompt_mode,
                    prompt_seed=prompt_seed,
                    prompt_jitter_pixels=prompt_jitter_pixels,
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
                losses = _semantic_loss(semantics, batch.ground_truth)
                for name, value in losses.items():
                    _merge_metrics(
                        totals, {f"loss/{name}": (float(value.cpu()), 1)}
                    )
                _merge_metrics(
                    totals, _semantic_metrics(semantics, batch.ground_truth)
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
                continue
            prediction = _predict(
                predictor,
                pipeline,
                batch.ground_truth,
                teacher_forcing=False,
                prompt_mode=prompt_mode,
                prompt_seed=prompt_seed,
                prompt_jitter_pixels=prompt_jitter_pixels,
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
            losses = _loss_terms(stage, prediction, batch)
            for name, value in losses.items():
                _merge_metrics(totals, {f"loss/{name}": (float(value.cpu()), 1)})
            _merge_metrics(
                totals,
                _batch_metrics(
                    prediction,
                    batch.ground_truth,
                    compute_lpips=compute_lpips,
                ),
            )
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
                    [f"sample_{batches}_{index}" for index in range(batch.frames.shape[0])],
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
                montages = _prediction_montages(
                    prediction,
                    batch.ground_truth,
                    sample_ids,
                    limit=wandb_image_limit,
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
    if batches == 0:
        raise RuntimeError("evaluation dataloader is empty")
    result = {
        name: value / max(count, 1.0)
        for name, (value, count) in totals.items()
    }
    result["runtime_seconds_per_sequence"] = runtime_seconds / max(runtime_sequences, 1)
    result["runtime_seconds_per_frame"] = runtime_seconds / max(runtime_frames, 1)
    if device.type == "cuda":
        result["peak_memory_mb"] = peak_memory_mb
    non_finite = {name: value for name, value in result.items() if not math.isfinite(value)}
    if non_finite:
        raise FloatingPointError(f"non-finite evaluation metrics: {non_finite}")
    return result


def _configure_stage(
    stage: int,
    predictor: MAM2VideoPredictor,
    pipeline: RefractiveMAM2,
    *,
    train_background_completion: bool = True,
) -> list[torch.nn.Parameter]:
    if stage == 1:
        return configure_stage1(predictor)
    if stage == 2:
        return configure_stage2(
            predictor,
            pipeline,
            train_background_completion=train_background_completion,
        )
    return configure_joint(predictor, pipeline)


def _loader(
    dataset: RCTransPRISMDataset,
    *,
    batch_size: int,
    shuffle: bool,
    workers: int,
    paired_backgrounds: bool,
    seed: int,
) -> DataLoader[RCTransBatch]:
    if paired_backgrounds:
        return build_paired_prism_dataloader(
            dataset,
            backgrounds_per_group=batch_size,
            shuffle=shuffle,
            seed=seed,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=prism_collate,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PRISM stage-wise training and evaluation")
    parser.add_argument("--train-data", type=Path)
    parser.add_argument("--val-data", type=Path)
    parser.add_argument("--test-data", type=Path)
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
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--mode", choices=("train", "test", "both"), default="both")
    parser.add_argument("--stage", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument(
        "--stage1-manifest",
        type=Path,
        action="append",
        default=[],
        help="JSONL VOS/matting manifest; may be repeated",
    )
    parser.add_argument("--prompt-seed", type=int, default=0)
    parser.add_argument("--prompt-jitter-pixels", type=float, default=0.0)
    parser.add_argument("--prompt-robustness-runs", type=int, default=0)
    parser.add_argument("--semantic-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--clip-length", type=int)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument(
        "--random-horizontal-flip",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", choices=("cosine", "none"), default="cosine")
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--teacher-forcing-start", type=float, default=1.0)
    parser.add_argument("--teacher-forcing-end", type=float, default=0.0)
    parser.add_argument("--stage2-matter-warmup-epochs", type=int, default=2)
    parser.add_argument("--paired-backgrounds", action="store_true")
    parser.add_argument("--paired-eval", action="store_true")
    parser.add_argument("--allow-unpaired-joint", action="store_true")
    parser.add_argument("--allow-uninitialized-stage", action="store_true")
    parser.add_argument("--selection-metric")
    parser.add_argument("--refinement-steps", type=int, default=2)
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
        default="max",
    )
    parser.add_argument("--strict-contract", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
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
    parser.add_argument("--wandb-mode", choices=("disabled", "offline", "online"), default="disabled")
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
    parser.add_argument("--diffusion-model")
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
    parser.add_argument("--diffusion-mask-dilation", type=int, default=8)
    parser.add_argument(
        "--diffusion-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument("--diffusion-device")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.mode in {"train", "both"}:
        if args.val_data is None:
            raise SystemExit("--val-data is required for train/both mode")
        if args.train_data is None and not (args.stage == 1 and args.stage1_manifest):
            raise SystemExit(
                "--train-data is required unless Stage 1 uses --stage1-manifest"
            )
        if (
            args.stage > 1
            and args.checkpoint is None
            and args.resume is None
            and not args.allow_uninitialized_stage
        ):
            raise SystemExit(
                "stage 2/3 training requires the preceding checkpoint; "
                "use --allow-uninitialized-stage only for a deliberate ablation"
            )
        if args.stage == 3 and not args.paired_backgrounds and not args.allow_unpaired_joint:
            raise SystemExit(
                "stage 3 requires --paired-backgrounds; use --allow-unpaired-joint "
                "only for the no-reuse ablation"
            )
    if args.mode in {"test", "both"} and args.test_data is None:
        raise SystemExit("--test-data is required for test/both mode")
    if args.batch_size < 1 or args.epochs < 1:
        raise SystemExit("--batch-size and --epochs must be positive")
    if args.wandb_image_interval < 0 or args.wandb_image_limit < 0:
        raise SystemExit("W&B image interval and limit must be non-negative")
    if args.prompt_jitter_pixels < 0 or args.prompt_robustness_runs < 0:
        raise SystemExit("prompt jitter and robustness runs must be non-negative")
    if args.refinement_steps < 0 or args.stage2_matter_warmup_epochs < 0:
        raise SystemExit("refinement steps and Stage-2 warmup must be non-negative")
    if args.checkpoint is not None and args.resume is not None:
        raise SystemExit("--checkpoint and --resume are mutually exclusive")
    if args.stage != 1 and args.stage1_manifest:
        raise SystemExit("--stage1-manifest is valid only for Stage 1")
    if args.paired_backgrounds and args.batch_size < 2:
        raise SystemExit("--paired-backgrounds requires --batch-size >= 2")
    if args.paired_eval and args.batch_size < 2:
        raise SystemExit("--paired-eval requires --batch-size >= 2")
    if args.completion_variant == "diffusion" and args.diffusion_model is None:
        raise SystemExit("--diffusion-model is required for PRISM-Diffusion")
    if args.completion_variant == "diffusion" and args.mode != "test":
        raise SystemExit(
            "PRISM-Diffusion is a frozen evaluation extension; train PRISM-Base "
            "first and run diffusion with --mode test"
        )
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

    _seed_everything(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = _device(args.device)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    evaluation_dataset = dict(
        clip_length=args.clip_length,
        frame_stride=args.frame_stride,
        strict_contract=args.strict_contract,
    )
    test_loader = None
    if args.test_data is not None:
        test_dataset = RCTransPRISMDataset(args.test_data, **evaluation_dataset)
        test_loader = _loader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            workers=args.workers,
            paired_backgrounds=args.paired_eval,
            seed=args.seed,
        )
    val_loader = None
    if args.val_data is not None:
        val_dataset = RCTransPRISMDataset(args.val_data, **evaluation_dataset)
        val_loader = _loader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            workers=args.workers,
            paired_backgrounds=False,
            seed=args.seed,
        )
    train_loader = None
    if args.stage == 1 and args.stage1_manifest:
        train_dataset = ManifestSemanticDataset(
            args.stage1_manifest,
            image_size=args.semantic_size,
            clip_length=args.clip_length,
            seed=args.seed,
            random_horizontal_flip=args.random_horizontal_flip,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=semantic_collate,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
        )
    elif args.train_data is not None:
        train_dataset = RCTransPRISMDataset(
            args.train_data,
            **evaluation_dataset,
            random_temporal_crop=True,
            random_horizontal_flip=args.random_horizontal_flip,
            augmentation_seed=args.seed,
        )
        train_loader = _loader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            workers=args.workers,
            paired_backgrounds=args.paired_backgrounds,
            seed=args.seed,
        )

    predictor = build_mam2_video_predictor(
        args.sam2_config,
        args.sam2_checkpoint,
        device=device,
        mode="train" if args.mode != "test" else "eval",
    )
    background_config = BackgroundConfig(
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
        diffusion_dtype=args.diffusion_dtype,
    )
    diffusion_completion = None
    if args.completion_variant == "diffusion":
        diffusion_device = args.diffusion_device or str(device)
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
            ),
            device=diffusion_device,
            dtype=args.diffusion_dtype,
        )
    config = PipelineConfig(
        background=background_config,
        matter=MatterConfig(),
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
    pipeline = RefractiveMAM2(
        predictor,
        config,
        background_model=background_model,
    ).to(device)
    initial_checkpoint = args.resume or args.checkpoint
    if initial_checkpoint is not None:
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
            optimizer = torch.optim.AdamW(
                parameters,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
            total_steps = max(len(train_loader) * args.epochs, 1)
            scheduler = (
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=total_steps,
                )
                if args.scheduler == "cosine"
                else None
            )
            selection_metric = args.selection_metric or (
                "mask_iou" if args.stage == 1 else "background_psnr"
            )
            best_value = float("-inf") if args.selection_mode == "max" else float("inf")
            if args.resume is not None:
                start_epoch, global_step, best_value = _restore_training_state(
                    load_refractive_training_state(args.resume),
                    optimizer,
                    scheduler,
                )
            local_best_checkpoint = (
                args.save_dir / f"prism_stage{args.stage}_best.pt"
            )
            # A resumed run may intentionally write to another directory. Until
            # the new run improves the validation score, the historical resume
            # checkpoint is the only valid best model we can test.
            best_checkpoint = (
                args.resume if args.resume is not None else local_best_checkpoint
            )
            for epoch in range(start_epoch, args.epochs):
                dataset = getattr(train_loader, "dataset", None)
                if hasattr(dataset, "set_epoch"):
                    dataset.set_epoch(epoch)
                sampler = getattr(train_loader, "batch_sampler", None)
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)
                matter_warmup = (
                    args.stage == 2 and epoch < args.stage2_matter_warmup_epochs
                )
                _configure_stage(
                    args.stage,
                    predictor,
                    pipeline,
                    train_background_completion=not matter_warmup,
                )
                for raw_batch in train_loader:
                    batch = raw_batch.to(device, non_blocking=True)
                    progress = global_step / max(total_steps - 1, 1)
                    probability = (
                        args.teacher_forcing_start * (1.0 - progress)
                        + args.teacher_forcing_end * progress
                    )
                    teacher_forcing = args.stage == 2 and (
                        matter_warmup or random.random() < probability
                    )
                    optimizer.zero_grad(set_to_none=True)
                    if args.stage == 1:
                        semantics = _semantic_forward(
                            predictor,
                            batch.ground_truth,
                            prompt_mode=args.prompt_mode,
                            prompt_seed=args.prompt_seed + epoch,
                            prompt_jitter_pixels=args.prompt_jitter_pixels,
                        )
                        losses = _semantic_loss(
                            semantics,
                            batch.ground_truth,
                            getattr(batch, "dataset_kinds", None),
                        )
                    else:
                        prediction = _predict(
                            predictor,
                            pipeline,
                            batch.ground_truth,
                            teacher_forcing=teacher_forcing,
                            prompt_mode=args.prompt_mode,
                            prompt_seed=args.prompt_seed + epoch,
                            prompt_jitter_pixels=args.prompt_jitter_pixels,
                        )
                        losses = _loss_terms(args.stage, prediction, batch)
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
                    if scheduler is not None:
                        scheduler.step()
                    log_train_images = (
                        logger.enabled
                        and args.wandb_image_interval > 0
                        and args.wandb_image_limit > 0
                        and global_step % args.wandb_image_interval == 0
                    )
                    logger.log(
                        {
                            **{f"train/{key}": value for key, value in losses.items()},
                            "train/gradient_norm": gradient_norm,
                            "train/teacher_forcing": float(teacher_forcing),
                            "train/matter_warmup": float(matter_warmup),
                            "train/epoch": epoch + 1,
                            "train/learning_rate": optimizer.param_groups[0]["lr"],
                        },
                        global_step,
                        commit=not log_train_images,
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
                        if args.stage == 1:
                            montages = _semantic_montages(
                                semantics,
                                batch.ground_truth,
                                sample_ids,
                                limit=args.wandb_image_limit,
                            )
                        else:
                            montages = _prediction_montages(
                                prediction,
                                batch.ground_truth,
                                sample_ids,
                                limit=args.wandb_image_limit,
                            )
                        logger.log_images(
                            "train/qualitative",
                            [image for image, _ in montages],
                            global_step,
                            captions=[caption for _, caption in montages],
                        )
                    global_step += 1

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
                )
                logger.log({f"validation/{k}": v for k, v in validation.items()}, global_step)
                if selection_metric not in validation:
                    raise RuntimeError(
                        f"selection metric {selection_metric!r} is absent; "
                        f"available={sorted(validation)}"
                    )
                current_value = validation[selection_metric]
                improved = (
                    current_value > best_value
                    if args.selection_mode == "max"
                    else current_value < best_value
                )
                if improved:
                    best_value = current_value
                checkpoint = args.save_dir / f"prism_stage{args.stage}_epoch{epoch + 1:03d}.pt"
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
                            "selection_mode": args.selection_mode,
                            "selection_value": best_value,
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
                print(json.dumps({"epoch": epoch + 1, **validation}, sort_keys=True))
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
                wandb_logger=logger,
                wandb_prefix="test",
                wandb_step=global_step,
                wandb_image_limit=args.wandb_image_limit,
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
                        )
                    )
                for name in sorted(set.intersection(*(set(run) for run in robustness))):
                    values = torch.tensor([run[name] for run in robustness])
                    metrics[f"prompt_robust/{name}_mean"] = float(values.mean())
                    metrics[f"prompt_robust/{name}_std"] = float(values.std(unbiased=False))
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
            logger.log({f"test/{key}": value for key, value in metrics.items()}, global_step)
            print(json.dumps(metrics, indent=2, sort_keys=True))
            print(f"metrics written to {metrics_path}")
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
