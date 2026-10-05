from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .losses import (
    RefractiveGroundTruth,
    RefractiveLoss,
    _mask_focal_dice,
    reusable_operator_consistency,
    reusable_operator_consistency_in_batch,
)
from .pipeline import RefractiveMAM2, RefractiveMAM2Output
from .sam2_integration import (
    MAM2VideoPredictor,
    mark_only_mam2_matter_trainable,
    mark_only_mam2_semantics_trainable,
    mark_only_mam2_trainable,
)
DatasetKind = Literal["vos", "video_matting", "image_matting", "synthetic_physics"]

SAM2_IMAGE_MEAN = (0.485, 0.456, 0.406)
SAM2_IMAGE_STD = (0.229, 0.224, 0.225)


@dataclass
class SemanticTargets:
    object_mask: Tensor | None = None
    trimap: Tensor | None = None
    alpha: Tensor | None = None
    alpha_validity: Tensor | None = None


def normalize_sam2_training_frames(frames: Tensor) -> Tensor:
    """Normalize resized RGB ``[B,T,3,H,W]`` frames from [0,1] for SAM2."""

    if frames.ndim != 5 or frames.shape[2] != 3:
        raise ValueError("frames must have shape [B,T,3,H,W]")
    mean = torch.as_tensor(SAM2_IMAGE_MEAN, device=frames.device, dtype=frames.dtype)
    std = torch.as_tensor(SAM2_IMAGE_STD, device=frames.device, dtype=frames.dtype)
    return (frames - mean.view(1, 1, 3, 1, 1)) / std.view(1, 1, 3, 1, 1)


def _resize_video_logits(logits: Tensor, size: tuple[int, int]) -> Tensor:
    if logits.ndim != 5:
        raise ValueError("logits must have shape [B,T,C,h,w]")
    b, t, c = logits.shape[:3]
    resized = F.interpolate(
        logits.reshape(b * t, c, *logits.shape[-2:]),
        size=size,
        mode="bilinear",
        align_corners=False,
    )
    return resized.reshape(b, t, c, *size)


def _normalized_focal_loss(
    logits: Tensor,
    target: Tensor,
    *,
    gamma: float = 2.0,
) -> Tensor:
    """Normalized focal loss used for the MAM2 trimap branch."""

    log_probability = F.log_softmax(logits, dim=1)
    probability = log_probability.exp()
    target = target.long()
    if target.ndim == logits.ndim and target.shape[1] == 1:
        target = target.squeeze(1)
    if target.ndim != logits.ndim - 1:
        raise ValueError("trimap target must have shape [N,H,W] or [N,1,H,W]")
    log_pt = log_probability.gather(1, target.unsqueeze(1)).squeeze(1)
    pt = probability.gather(1, target.unsqueeze(1)).squeeze(1)
    focal_weight = (1.0 - pt).pow(gamma)
    per_sample = -(focal_weight * log_pt).flatten(1).sum(dim=1)
    normalizer = focal_weight.flatten(1).sum(dim=1).clamp_min(1e-6)
    return (per_sample / normalizer).mean()


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    mask = mask.to(value.dtype)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def _gaussian_kernel(value: Tensor) -> Tensor:
    kernel = value.new_tensor(
        (
            (1, 4, 6, 4, 1),
            (4, 16, 24, 16, 4),
            (6, 24, 36, 24, 6),
            (4, 16, 24, 16, 4),
            (1, 4, 6, 4, 1),
        )
    )
    return (kernel / 256.0).reshape(1, 1, 5, 5)


def _gaussian_convolution(value: Tensor, kernel: Tensor) -> Tensor:
    batch, channels, height, width = value.shape
    flat = value.reshape(batch * channels, 1, height, width)
    flat = F.pad(flat, (2, 2, 2, 2), mode="reflect")
    return F.conv2d(flat, kernel).reshape(batch, channels, height, width)


def _laplacian_pyramid_loss(prediction: Tensor, target: Tensor) -> Tensor:
    kernel = _gaussian_kernel(prediction)
    prediction = prediction.flatten(0, 1)
    target = target.flatten(0, 1)
    total = prediction.new_zeros(())
    levels = 0
    for level in range(5):
        height = prediction.shape[-2] - prediction.shape[-2] % 2
        width = prediction.shape[-1] - prediction.shape[-1] % 2
        if min(height, width) < 4:
            break
        prediction = prediction[..., :height, :width]
        target = target[..., :height, :width]
        prediction_down = _gaussian_convolution(prediction, kernel)[..., ::2, ::2]
        target_down = _gaussian_convolution(target, kernel)[..., ::2, ::2]

        def upsample(value: Tensor, size: tuple[int, int]) -> Tensor:
            up = value.new_zeros((*value.shape[:-2], size[0], size[1]))
            up[..., ::2, ::2] = value * 4.0
            return _gaussian_convolution(up, kernel)

        prediction_laplacian = prediction - upsample(
            prediction_down, prediction.shape[-2:]
        )
        target_laplacian = target - upsample(target_down, target.shape[-2:])
        total = total + (2**level) * F.l1_loss(
            prediction_laplacian, target_laplacian
        )
        levels += 1
        prediction, target = prediction_down, target_down
    return total / max(levels, 1)


def _alpha_matte_loss_terms(
    prediction: Tensor,
    target: Tensor,
    trimap: Tensor,
    validity: Tensor,
) -> dict[str, Tensor]:
    """MAM2/MEMatte alpha objective from the published appendix."""

    validity = validity.to(prediction.dtype)
    unknown = (trimap == 1).to(prediction.dtype) * validity
    known = (trimap != 1).to(prediction.dtype) * validity
    absolute = (prediction - target).abs()
    squared = (prediction - target).square()
    terms = {
        "mam2_alpha_unknown_l1": _masked_mean(absolute, unknown),
        "mam2_alpha_known_l1": _masked_mean(absolute, known),
        "mam2_alpha_l2": _masked_mean(squared, validity),
        "mam2_alpha_laplacian": _laplacian_pyramid_loss(
            prediction * validity,
            target * validity,
        ),
    }
    flat_prediction = prediction.flatten(0, 1)
    flat_target = target.flatten(0, 1)
    flat_unknown = unknown.flatten(0, 1)
    sobel_x = prediction.new_tensor(
        ((((-1, 0, 1), (-2, 0, 2), (-1, 0, 1)),),)
    )
    sobel_y = prediction.new_tensor(
        ((((-1, -2, -1), (0, 0, 0), (1, 2, 1)),),)
    )
    pred_dx = F.conv2d(flat_prediction, sobel_x, padding=1)
    true_dx = F.conv2d(flat_target, sobel_x, padding=1)
    pred_dy = F.conv2d(flat_prediction, sobel_y, padding=1)
    true_dy = F.conv2d(flat_target, sobel_y, padding=1)
    terms["mam2_alpha_gradient"] = (
        _masked_mean((pred_dx - true_dx).abs(), flat_unknown)
        + _masked_mean((pred_dy - true_dy).abs(), flat_unknown)
        + 0.01 * _masked_mean(pred_dx.abs(), flat_unknown)
        + 0.01 * _masked_mean(pred_dy.abs(), flat_unknown)
    )
    return terms


def selective_semantic_loss(
    mask_logits: Tensor,
    trimap_logits: Tensor,
    alpha_matte: Tensor,
    target: SemanticTargets,
    dataset_kind: DatasetKind,
) -> dict[str, Tensor]:
    """MAM2 stage-1 selective supervision.

    VOS data supervises the stable mask path. Image/video matting data
    supervises trimaps. Synthetic physics clips can supervise both because both
    labels are exact by construction.
    """

    terms: dict[str, Tensor] = {}
    if dataset_kind in ("vos", "synthetic_physics"):
        if target.object_mask is None:
            raise ValueError(f"{dataset_kind} requires object_mask")
        mask = _resize_video_logits(mask_logits, target.object_mask.shape[-2:])
        terms["mask"] = _mask_focal_dice(mask, target.object_mask)
    if dataset_kind in ("video_matting", "image_matting", "synthetic_physics"):
        if target.trimap is None:
            raise ValueError(f"{dataset_kind} requires trimap")
        trimap = _resize_video_logits(trimap_logits, target.trimap.shape[-2:])
        terms["trimap"] = _normalized_focal_loss(
            trimap.flatten(0, 1),
            target.trimap.flatten(0, 1).long(),
        )
        if target.alpha is not None:
            alpha = _resize_video_logits(alpha_matte, target.alpha.shape[-2:])
            validity = (
                torch.ones_like(target.alpha)
                if target.alpha_validity is None
                else target.alpha_validity.to(alpha.dtype)
            )
            terms.update(
                _alpha_matte_loss_terms(
                    alpha,
                    target.alpha,
                    target.trimap,
                    validity,
                )
            )
    if not terms:
        raise ValueError(f"no loss is defined for dataset_kind={dataset_kind!r}")
    terms["total"] = sum(terms.values(), mask_logits.new_zeros(()))
    return terms


def configure_stage1a(predictor: MAM2VideoPredictor) -> list[nn.Parameter]:
    """Train mask/trimap semantics; freeze both alpha matter and SAM2 base."""

    predictor.train()
    return mark_only_mam2_semantics_trainable(predictor)


def configure_stage1b(predictor: MAM2VideoPredictor) -> list[nn.Parameter]:
    """Train alpha matter from RGB+trimap; freeze mask/trimap semantics."""

    predictor.train()
    return mark_only_mam2_matter_trainable(predictor)


def configure_stage2(
    predictor: MAM2VideoPredictor,
    physics_pipeline: RefractiveMAM2,
) -> list[nn.Parameter]:
    """Warm up PRISM-PAM with oracle background; freeze background recovery."""

    physics_pipeline.train().requires_grad_(False)
    # ``physics_pipeline.train()`` recurses into its registered backbone, so
    # place the frozen semantic predictor back in eval mode afterwards.
    predictor.eval().requires_grad_(False)
    physics_pipeline.matter.requires_grad_(True)
    return [parameter for parameter in physics_pipeline.parameters() if parameter.requires_grad]


def configure_stage3(
    predictor: MAM2VideoPredictor,
    physics_pipeline: RefractiveMAM2,
) -> list[nn.Parameter]:
    """Train PRISM-PAM and background recovery; keep MAM2 frozen."""

    physics_pipeline.train().requires_grad_(False)
    predictor.eval().requires_grad_(False)
    physics_pipeline.background_model.requires_grad_(True)
    physics_pipeline.matter.requires_grad_(True)
    return [parameter for parameter in physics_pipeline.parameters() if parameter.requires_grad]


def configure_stage4(
    predictor: MAM2VideoPredictor,
    physics_pipeline: RefractiveMAM2,
) -> list[nn.Parameter]:
    """Jointly tune adapters, alpha decoder, PAM and background recovery.

    Original SAM2 stays frozen. The paper configuration tunes MEMatte's
    adaptive-token backbone and decoder; a decoder-only memory ablation is
    configurable. Semantic outputs, inverse refractive splatting and the shared
    background remain in one autograd graph.
    """

    physics_pipeline.train().requires_grad_(False)
    # Freeze the full registered pipeline first, then re-enable the semantic
    # extension. Reversing this order silently disables PDD/MSS and LoRA.
    semantic_parameters = mark_only_mam2_trainable(predictor)
    predictor.train()
    physics_pipeline.background_model.requires_grad_(True)
    physics_pipeline.matter.requires_grad_(True)
    physics_parameters = [
        parameter
        for parameter in physics_pipeline.parameters()
        if parameter.requires_grad
    ]
    unique: dict[int, nn.Parameter] = {}
    for parameter in (*semantic_parameters, *physics_parameters):
        unique[id(parameter)] = parameter
    return list(unique.values())


# Compatibility aliases for earlier experiments. New runs should use the
# explicit 1A/1B/2/3/4 functions above.
configure_stage1 = configure_stage1a
configure_joint = configure_stage4


def physics_stage_loss(
    prediction: RefractiveMAM2Output,
    target: RefractiveGroundTruth,
    loss: RefractiveLoss | None = None,
) -> dict[str, Tensor]:
    return (loss or RefractiveLoss())(prediction, target)


def joint_stage_loss(
    prediction: RefractiveMAM2Output,
    target: RefractiveGroundTruth,
    *,
    paired_background_prediction: RefractiveMAM2Output | None = None,
    paired_background_group_ids: Sequence[str] | None = None,
    operator_support: Tensor | None = None,
    loss: RefractiveLoss | None = None,
) -> dict[str, Tensor]:
    """Full objective with optional cross-background operator invariance.

    ``paired_background_group_ids`` is the efficient path for a paired RCTrans
    batch.  ``paired_background_prediction`` remains for two separately run
    predictions and is mutually exclusive with it.
    """

    criterion = loss or RefractiveLoss()
    terms = criterion(prediction, target)
    if (
        paired_background_prediction is not None
        and paired_background_group_ids is not None
    ):
        raise ValueError("Choose one paired-operator supervision API")
    if paired_background_prediction is not None:
        reuse = reusable_operator_consistency(
            prediction.matter,
            paired_background_prediction.matter,
            operator_support,
        )
        terms["operator_reuse"] = reuse
        terms["total"] = terms["total"] + criterion.weights.operator_reuse * reuse
    elif paired_background_group_ids is not None:
        reuse = reusable_operator_consistency_in_batch(
            prediction.matter,
            paired_background_group_ids,
            operator_support,
        )
        terms["operator_reuse"] = reuse
        terms["total"] = terms["total"] + criterion.weights.operator_reuse * reuse
    return terms
