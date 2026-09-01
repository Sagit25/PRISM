from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import LossWeights
from .matter import PhysicsMatterOutput
from .pipeline import RefractiveMAM2Output
from .renderer import warp_background


@dataclass
class RefractiveGroundTruth:
    frames: Tensor
    object_mask: Tensor | None = None
    trimap: Tensor | None = None
    alpha: Tensor | None = None
    straight_foreground: Tensor | None = None
    premultiplied_foreground: Tensor | None = None
    color_transmission: Tensor | None = None
    transmittance: Tensor | None = None
    source_coordinates: Tensor | None = None
    refractive_flow: Tensor | None = None
    residual: Tensor | None = None
    confidence: Tensor | None = None
    refractive_validity: Tensor | None = None
    counterfactual_background: Tensor | None = None
    temporal_flow_to_next: Tensor | None = None

    def select(self, index: int | slice | Tensor) -> "RefractiveGroundTruth":
        """Select batch elements while preserving optional RCTrans labels."""

        values: dict[str, Tensor | None] = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            values[name] = None if value is None else value[index]
            if value is not None and isinstance(index, int):
                values[name] = values[name].unsqueeze(0)
        return RefractiveGroundTruth(**values)


def _charbonnier(value: Tensor, eps: float = 1e-3) -> Tensor:
    return torch.sqrt(value.square() + eps * eps).mean()


def _masked_charbonnier(
    value: Tensor,
    mask: Tensor | float,
    eps: float = 1e-3,
) -> Tensor:
    """Robust mean over valid elements without diluting sparse supervision."""

    if not isinstance(mask, Tensor):
        return _charbonnier(value, eps)
    weight = mask.to(device=value.device, dtype=value.dtype)
    while weight.ndim < value.ndim:
        weight = weight.unsqueeze(-3)
    weight = weight.expand_as(value)
    robust = torch.sqrt(value.square() + eps * eps) - eps
    return (robust * weight).sum() / weight.sum().clamp_min(1.0)


def source_coordinates_from_flow(flow: Tensor) -> Tensor:
    """Return absolute RCTrans source coordinates ``Phi=x+u`` in pixels."""

    if flow.ndim != 5 or flow.shape[2] != 2:
        raise ValueError("flow must have shape [B,T,2,H,W]")
    h, w = flow.shape[-2:]
    y, x = torch.meshgrid(
        torch.arange(h, device=flow.device, dtype=flow.dtype),
        torch.arange(w, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    coordinates = torch.stack((x, y), dim=0).view(1, 1, 2, h, w)
    return coordinates + flow


def _gradient_loss(prediction: Tensor, target: Tensor) -> Tensor:
    pred_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    pred_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    return _charbonnier(pred_dx - target_dx) + _charbonnier(pred_dy - target_dy)


def _mask_focal_dice(logits: Tensor, target: Tensor) -> Tensor:
    target = target.to(logits.dtype)
    probability = logits.sigmoid()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = probability * target + (1.0 - probability) * (1.0 - target)
    focal = ((1.0 - pt).square() * bce).mean()
    dims = tuple(range(2, logits.ndim))
    intersection = (probability * target).sum(dim=dims)
    denominator = probability.sum(dim=dims) + target.sum(dim=dims)
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return focal + dice


def _edge_aware_flow_smoothness(flow: Tensor, frames: Tensor) -> Tensor:
    flow_dx = flow[..., :, 1:] - flow[..., :, :-1]
    flow_dy = flow[..., 1:, :] - flow[..., :-1, :]
    image_dx = (frames[..., :, 1:] - frames[..., :, :-1]).abs().mean(dim=2, keepdim=True)
    image_dy = (frames[..., 1:, :] - frames[..., :-1, :]).abs().mean(dim=2, keepdim=True)
    return (flow_dx.abs() * torch.exp(-10.0 * image_dx)).mean() + (
        flow_dy.abs() * torch.exp(-10.0 * image_dy)
    ).mean()


def _flow_out_of_bounds(flow: Tensor) -> Tensor:
    h, w = flow.shape[-2:]
    y, x = torch.meshgrid(
        torch.arange(h, device=flow.device, dtype=flow.dtype),
        torch.arange(w, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    source_x = x + flow[:, :, 0]
    source_y = y + flow[:, :, 1]
    violation = (
        (-source_x).relu()
        + (source_x - (w - 1)).relu()
        + (-source_y).relu()
        + (source_y - (h - 1)).relu()
    )
    return violation.mean() / max(h, w, 1)


def _multiscale_render_loss(prediction: Tensor, target: Tensor) -> Tensor:
    b, t, c, h, w = prediction.shape
    total = prediction.new_zeros(())
    levels = 0
    for scale in (0.5, 0.25):
        size = (max(1, round(h * scale)), max(1, round(w * scale)))
        pred_scaled = F.interpolate(
            prediction.reshape(b * t, c, h, w), size=size, mode="area"
        )
        target_scaled = F.interpolate(
            target.reshape(b * t, c, h, w), size=size, mode="area"
        )
        total = total + _charbonnier(pred_scaled - target_scaled)
        levels += 1
    return total / levels


def _resize_video_logits(logits: Tensor, size: tuple[int, int]) -> Tensor:
    b, t, c = logits.shape[:3]
    resized = F.interpolate(
        logits.reshape(b * t, c, *logits.shape[-2:]),
        size=size,
        mode="bilinear",
        align_corners=False,
    )
    return resized.reshape(b, t, c, *size)


def _global_background(value: Tensor, frames: Tensor) -> Tensor:
    if value.shape == (frames.shape[0], 3, *frames.shape[-2:]):
        return value
    if value.shape == frames.shape:
        return value[:, 0]
    raise ValueError("background GT must be [B,3,H,W] or legacy [B,T,3,H,W]")


def reusable_operator_consistency(
    first: PhysicsMatterOutput,
    second: PhysicsMatterOutput,
    support: Tensor | None = None,
) -> Tensor:
    """Keep the same object/pose operator invariant across two backgrounds.

    Synthetic training should render a shared object trajectory on two
    backgrounds and pass the paired predictions here.  The backgrounds may
    differ; the reusable operator must not.
    """

    weight: Tensor | float = 1.0 if support is None else support
    return (
        _masked_charbonnier(first.alpha - second.alpha, weight)
        + _masked_charbonnier(
            first.premultiplied_foreground - second.premultiplied_foreground,
            weight,
        )
        + _masked_charbonnier(first.transmittance - second.transmittance, weight)
        + _masked_charbonnier(
            first.color_transmission - second.color_transmission, weight
        )
        + _masked_charbonnier(first.refractive_flow - second.refractive_flow, weight)
        + _masked_charbonnier(first.residual - second.residual, weight)
        + _masked_charbonnier(first.confidence - second.confidence, weight)
    )


def reusable_operator_consistency_in_batch(
    matter: PhysicsMatterOutput,
    paired_background_group_ids: Sequence[str],
    support: Tensor | None = None,
) -> Tensor:
    """Compare operator predictions belonging to the same RCTrans pair group."""

    batch_size = matter.alpha.shape[0]
    if len(paired_background_group_ids) != batch_size:
        raise ValueError("pair-group count must match prediction batch size")
    groups: dict[str, list[int]] = {}
    for index, group_id in enumerate(paired_background_group_ids):
        groups.setdefault(group_id, []).append(index)

    def selected(index: int) -> PhysicsMatterOutput:
        return PhysicsMatterOutput(
            **{
                name: getattr(matter, name)[index : index + 1]
                for name in matter.__dataclass_fields__
            }
        )

    terms: list[Tensor] = []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        anchor = indices[0]
        for paired in indices[1:]:
            pair_support = None
            if support is not None:
                pair_support = torch.minimum(
                    support[anchor : anchor + 1], support[paired : paired + 1]
                )
            terms.append(
                reusable_operator_consistency(
                    selected(anchor), selected(paired), pair_support
                )
            )
    if not terms:
        raise ValueError("No repeated paired_background_group_id in this batch")
    return torch.stack(terms).mean()


def _temporal_loss(video: Tensor, flow_to_next: Tensor) -> Tensor:
    """Align frame t+1 to t using a target(t)-to-source(t+1) flow."""

    if video.shape[1] < 2:
        return video.new_zeros(())
    if flow_to_next.shape != (
        video.shape[0],
        video.shape[1] - 1,
        2,
        video.shape[-2],
        video.shape[-1],
    ):
        raise ValueError("temporal_flow_to_next has an invalid shape")
    aligned_next = warp_background(video[:, 1:], flow_to_next)
    return _charbonnier(video[:, :-1] - aligned_next)


class RefractiveLoss(nn.Module):
    def __init__(self, weights: LossWeights | None = None) -> None:
        super().__init__()
        self.weights = weights or LossWeights()

    def forward(
        self,
        prediction: RefractiveMAM2Output,
        target: RefractiveGroundTruth,
    ) -> dict[str, Tensor]:
        h, w = target.frames.shape[-2:]
        terms: dict[str, Tensor] = {}
        mask_logits = _resize_video_logits(prediction.backbone.mask_logits, (h, w))
        trimap_logits = _resize_video_logits(prediction.backbone.trimap_logits, (h, w))

        if target.object_mask is not None:
            terms["mask"] = _mask_focal_dice(mask_logits, target.object_mask)
        if target.trimap is not None:
            class_weight = trimap_logits.new_tensor((1.0, 2.0, 1.0))
            terms["trimap"] = F.cross_entropy(
                trimap_logits.flatten(0, 1),
                target.trimap.flatten(0, 1).long(),
                weight=class_weight,
            )
        if target.alpha is not None:
            terms["alpha"] = _charbonnier(prediction.matter.alpha - target.alpha)
            terms["alpha_gradient"] = _gradient_loss(
                prediction.matter.alpha, target.alpha
            )

        target_g = target.premultiplied_foreground
        if target_g is None and target.straight_foreground is not None and target.alpha is not None:
            target_g = target.alpha * target.straight_foreground
        if target_g is not None:
            support = target.object_mask if target.object_mask is not None else 1.0
            terms["premultiplied_foreground"] = _masked_charbonnier(
                prediction.matter.premultiplied_foreground - target_g, support
            )

        validity: Tensor | float = (
            target.refractive_validity
            if target.refractive_validity is not None
            else 1.0
        )
        refractive_support: Tensor | float = validity
        if target.object_mask is not None:
            refractive_support = target.object_mask * validity

        if target.color_transmission is not None:
            terms["color_transmission"] = _masked_charbonnier(
                prediction.matter.color_transmission - target.color_transmission,
                refractive_support,
            )

        if target.transmittance is not None:
            terms["transmittance"] = _masked_charbonnier(
                prediction.matter.transmittance - target.transmittance,
                refractive_support,
            )

        if target.residual is not None:
            support = target.object_mask if target.object_mask is not None else 1.0
            terms["residual"] = _masked_charbonnier(
                prediction.matter.residual - target.residual, support
            )
        terms["residual_sparsity"] = prediction.matter.residual.abs().mean()

        if target.refractive_flow is not None:
            endpoint = torch.linalg.vector_norm(
                prediction.matter.refractive_flow - target.refractive_flow,
                dim=2,
                keepdim=True,
            )
            terms["refractive_flow"] = _masked_charbonnier(
                endpoint, refractive_support
            )
        if target.source_coordinates is not None:
            predicted_phi = source_coordinates_from_flow(
                prediction.matter.refractive_flow
            )
            terms["source_coordinates"] = _masked_charbonnier(
                predicted_phi - target.source_coordinates,
                refractive_support,
            )
        if target.counterfactual_background is not None:
            background_gt = _global_background(
                target.counterfactual_background, target.frames
            )
            terms["background"] = _charbonnier(
                prediction.background.background - background_gt
            )
            terms["observed_background"] = _masked_charbonnier(
                prediction.background.observed_background - background_gt,
                prediction.background.direct_coverage,
            )
            terms["inverse_background"] = _masked_charbonnier(
                prediction.background.inverse_background - background_gt,
                prediction.background.inverse_coverage,
            )
            terms["background_true_hole"] = _masked_charbonnier(
                prediction.background.background - background_gt,
                prediction.background.true_hole,
            )

        terms["render"] = _charbonnier(
            prediction.reconstructed_frames - target.frames
        )
        terms["render_multiscale"] = _multiscale_render_loss(
            prediction.reconstructed_frames, target.frames
        )
        terms["flow_smoothness"] = _edge_aware_flow_smoothness(
            prediction.matter.refractive_flow, target.frames
        )
        terms["flow_out_of_bounds"] = _flow_out_of_bounds(
            prediction.matter.refractive_flow
        )
        if target.confidence is not None:
            confidence_target = target.confidence.to(
                prediction.matter.confidence.dtype
            ).clamp(0.0, 1.0)
        else:
            render_error = (
                prediction.reconstructed_frames - target.frames
            ).abs().mean(dim=2, keepdim=True)
            confidence_target = torch.exp(-10.0 * render_error).detach()
        confidence_loss = F.binary_cross_entropy(
            prediction.matter.confidence.clamp(1e-5, 1.0 - 1e-5),
            confidence_target,
            reduction="none",
        )
        if isinstance(refractive_support, Tensor):
            confidence_support = refractive_support.to(confidence_loss.dtype)
        else:
            confidence_support = (
                prediction.matter.alpha.detach() > 1e-3
            ).to(confidence_loss.dtype)
        terms["confidence"] = (confidence_loss * confidence_support).sum() / (
            confidence_support.sum().clamp_min(1.0)
        )
        if target.temporal_flow_to_next is not None:
            terms["temporal"] = (
                _temporal_loss(prediction.matter.alpha, target.temporal_flow_to_next)
                + _temporal_loss(
                    prediction.matter.premultiplied_foreground,
                    target.temporal_flow_to_next,
                )
                + _temporal_loss(
                    prediction.matter.refractive_flow,
                    target.temporal_flow_to_next,
                )
            )

        total = target.frames.new_zeros(())
        for name, value in terms.items():
            total = total + getattr(self.weights, name) * value
        terms["total"] = total
        return terms
