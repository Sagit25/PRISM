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
from .sam2_integration import MAM2VideoPredictor, mark_only_mam2_trainable
from .logger import WandbLogger

DatasetKind = Literal["vos", "video_matting", "image_matting", "synthetic_physics"]

SAM2_IMAGE_MEAN = (0.485, 0.456, 0.406)
SAM2_IMAGE_STD = (0.229, 0.224, 0.225)


@dataclass
class SemanticTargets:
    object_mask: Tensor | None = None
    trimap: Tensor | None = None


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


def selective_semantic_loss(
    mask_logits: Tensor,
    trimap_logits: Tensor,
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
        terms["trimap"] = F.cross_entropy(
            trimap.flatten(0, 1),
            target.trimap.flatten(0, 1).long(),
            weight=trimap.new_tensor((1.0, 2.0, 1.0)),
        )
    if not terms:
        raise ValueError(f"no loss is defined for dataset_kind={dataset_kind!r}")
    terms["total"] = sum(terms.values(), mask_logits.new_zeros(()))
    return terms


def configure_stage1(predictor: MAM2VideoPredictor) -> list[nn.Parameter]:
    """Train PDD/MSS and encoder LoRA while freezing original SAM2 weights."""

    predictor.train()
    return mark_only_mam2_trainable(predictor)


def configure_stage2(
    predictor: MAM2VideoPredictor,
    physics_pipeline: RefractiveMAM2,
    *,
    train_background_completion: bool = True,
) -> list[nn.Parameter]:
    """Freeze semantic tracking and train the physical decomposition heads."""

    predictor.eval().requires_grad_(False)
    physics_pipeline.train().requires_grad_(False)
    physics_pipeline.matter.requires_grad_(True)
    if train_background_completion:
        physics_pipeline.background_model.requires_grad_(True)
    return [parameter for parameter in physics_pipeline.parameters() if parameter.requires_grad]


def configure_joint(
    predictor: MAM2VideoPredictor,
    physics_pipeline: RefractiveMAM2,
) -> list[nn.Parameter]:
    """Final stage: jointly train PDD/MSS/LoRA, background and operator.

    Original SAM2 weights stay frozen.  Unlike stage 2, semantic outputs,
    inverse refractive splatting and the shared background remain in one
    autograd graph.
    """

    semantic_parameters = mark_only_mam2_trainable(predictor)
    predictor.train()
    physics_pipeline.train().requires_grad_(False)
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
