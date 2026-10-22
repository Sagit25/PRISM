from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import LossWeights
from .matter import PhysicsMatterOutput
from .pipeline import RefractiveMAM2Output
from .renderer import warp_background, warp_background_kernel


@dataclass
class RefractiveGroundTruth:
    frames: Tensor
    object_mask: Tensor | None = None
    trimap: Tensor | None = None
    alpha: Tensor | None = None
    alpha_validity: Tensor | None = None
    straight_foreground: Tensor | None = None
    premultiplied_foreground: Tensor | None = None
    color_transmission: Tensor | None = None
    transmittance: Tensor | None = None
    source_coordinates: Tensor | None = None
    refractive_flow: Tensor | None = None
    residual: Tensor | None = None
    confidence: Tensor | None = None
    refractive_validity: Tensor | None = None
    surface_normal: Tensor | None = None
    depth: Tensor | None = None
    normal_validity: Tensor | None = None
    depth_validity: Tensor | None = None
    counterfactual_background: Tensor | None = None
    temporal_flow_to_next: Tensor | None = None
    temporal_flow_validity: Tensor | None = None

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


def _straight_foreground_support(
    target: RefractiveGroundTruth, min_alpha: float = 0.05
) -> Tensor | None:
    """F is identifiable only on reliable, sufficiently opaque GT pixels."""

    if target.alpha is None:
        return None
    support = (target.alpha >= min_alpha).to(target.alpha.dtype)
    if target.alpha_validity is not None:
        support = support * target.alpha_validity
    return support


def _object_coverage_support(target: RefractiveGroundTruth) -> Tensor | None:
    """Include antialiased foreground missed by a center-ray binary mask."""

    support = target.object_mask
    if target.alpha is not None:
        alpha_coverage = (target.alpha > 1e-3).to(target.alpha.dtype)
        support = (
            alpha_coverage if support is None
            else torch.maximum(support, alpha_coverage)
        )
    return support


def _far_background_support(object_mask: Tensor) -> Tensor:
    """Exclude the one-pixel antialiased silhouette from leakage supervision."""

    if object_mask.ndim != 5 or object_mask.shape[2] != 1:
        raise ValueError("object_mask must have shape [B,T,1,H,W]")
    expanded = F.max_pool2d(
        object_mask.flatten(0, 1).float(), 3, stride=1, padding=1
    )
    return (1.0 - expanded).clamp(0.0, 1.0).reshape_as(object_mask)


def _identifiable_color_transmission(target: RefractiveGroundTruth) -> Tensor | float:
    """T is defined only where the object transmits non-negligible light.

    Unlike relative color T, absolute RGB tau remains supervised at zero, for
    example at totally internally reflected or fully opaque object pixels.
    """

    coverage = _object_coverage_support(target)
    support: Tensor | float = 1.0 if coverage is None else coverage
    if target.transmittance is not None:
        strength = target.transmittance.amax(dim=2, keepdim=True)
    elif target.alpha is not None:
        strength = 1.0 - target.alpha
    else:
        return support
    return support * (strength > 1e-3).to(strength.dtype)


def _past_visible_refracted_background(
    target: RefractiveGroundTruth,
    flow: Tensor | None = None,
) -> tuple[Tensor, Tensor] | None:
    """Sample only *earlier* directly visible pixels at a source Phi.

    This uses the fixed-camera temporal background assumption. A bilinear
    source is trusted only when every contributing source pixel was seen
    without the object in an earlier frame; no inpainted pixels enter here.
    """

    if target.object_mask is None or target.refractive_flow is None:
        return None
    if target.frames.shape[1] < 2:
        return None
    # A center-ray binary mask can miss antialiased/transmitting silhouette
    # pixels. Their RGB is not a clean background observation either.
    visible = target.object_mask < 0.5
    if target.alpha is not None:
        visible = visible & (target.alpha < 1e-3)
    visible = visible.to(target.frames.dtype)
    past_count = visible.cumsum(dim=1) - visible
    past_sum = (target.frames * visible).cumsum(dim=1) - target.frames * visible
    past_background = past_sum / past_count.clamp_min(1.0)
    source_flow = target.refractive_flow if flow is None else flow
    sampled_background = warp_background(past_background, source_flow)
    sampled_support = warp_background(
        (past_count > 0).to(target.frames.dtype), source_flow
    )
    support = (sampled_support > 1.0 - 1e-4).to(target.frames.dtype)
    if target.refractive_validity is not None:
        support = support * target.refractive_validity
    support = support * _identifiable_color_transmission(target)
    return sampled_background, support


def _probability_binary_cross_entropy(
    probability: Tensor,
    target: Tensor,
    *,
    name: str = "confidence",
) -> Tensor:
    """Evaluate probability-space BCE in FP32 outside mixed precision.

    The matter head intentionally exposes a bounded confidence probability,
    rather than an unbounded logit, because the same value is consumed by the
    physical compositor. CUDA autocast rejects probability-space BCE, so keep
    this small loss calculation in FP32 while preserving gradients to the
    confidence head.
    """

    with torch.autocast(device_type=probability.device.type, enabled=False):
        probability = probability.float()
        target = target.float()
        if not bool(torch.isfinite(probability).all()):
            raise FloatingPointError(f"{name} prediction contains NaN or Inf")
        if not bool(torch.isfinite(target).all()):
            raise FloatingPointError(f"{name} target contains NaN or Inf")
        if bool(((probability < 0.0) | (probability > 1.0)).any()):
            raise ValueError(f"{name} prediction is outside [0, 1]")
        if bool(((target < 0.0) | (target > 1.0)).any()):
            raise ValueError(f"{name} target is outside [0, 1]")
        return F.binary_cross_entropy(
            probability.clamp(1e-5, 1.0 - 1e-5),
            target,
            reduction="none",
        )


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


def _masked_gradient_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
) -> Tensor:
    """Match local geometry gradients only when both adjacent pixels are valid."""

    pred_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    pred_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    valid_dx = mask[..., :, 1:] * mask[..., :, :-1]
    valid_dy = mask[..., 1:, :] * mask[..., :-1, :]
    return _masked_charbonnier(pred_dx - target_dx, valid_dx) + _masked_charbonnier(
        pred_dy - target_dy, valid_dy
    )


def _masked_frequency_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
) -> Tensor:
    """Match true-hole structure in the Fourier domain.

    The spatial true-hole objective remains the primary reconstruction term.
    This auxiliary GLaMa-style term exposes periodic/ripple errors and global
    structure mismatches that a purely pixelwise loss can under-penalize.
    """

    weight = mask.to(device=prediction.device, dtype=prediction.dtype)
    prediction = prediction * weight
    target = target * weight
    prediction_spectrum = torch.fft.rfft2(prediction.float(), norm="ortho")
    target_spectrum = torch.fft.rfft2(target.float(), norm="ortho")
    difference = torch.view_as_real(prediction_spectrum - target_spectrum)
    return _charbonnier(difference).to(prediction.dtype)


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


def _edge_aware_flow_smoothness(
    flow: Tensor,
    frames: Tensor,
    target_flow: Tensor | None = None,
    support: Tensor | float = 1.0,
) -> Tensor:
    """Smooth only unsupervised flow; with GT, smooth the flow error.

    Penalizing the raw flow gradient at supervised pixels conflicts with a
    sharp but valid refractive correspondence. Adjacent validity prevents a
    background/object boundary or invalid ray from affecting this term.
    """

    if target_flow is not None:
        flow = flow - target_flow
    flow_dx = flow[..., :, 1:] - flow[..., :, :-1]
    flow_dy = flow[..., 1:, :] - flow[..., :-1, :]
    image_dx = (frames[..., :, 1:] - frames[..., :, :-1]).abs().mean(dim=2, keepdim=True)
    image_dy = (frames[..., 1:, :] - frames[..., :-1, :]).abs().mean(dim=2, keepdim=True)
    if isinstance(support, Tensor):
        valid_dx = support[..., :, 1:] * support[..., :, :-1]
        valid_dy = support[..., 1:, :] * support[..., :-1, :]
    else:
        valid_dx = valid_dy = support
    return _masked_charbonnier(
        flow_dx * torch.exp(-10.0 * image_dx), valid_dx
    ) + _masked_charbonnier(flow_dy * torch.exp(-10.0 * image_dy), valid_dy)


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
    backgrounds and pass the paired predictions here.  The physical operator
    should agree across backgrounds, but R is the signed remainder of the
    single-flow image formation model and may change with the background.
    """

    weight: Tensor | float = 1.0 if support is None else support
    total = (
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
        + _masked_charbonnier(first.confidence - second.confidence, weight)
    )
    if (
        first.refractive_kernel_weights is not None
        and second.refractive_kernel_weights is not None
    ):
        total = total + _masked_charbonnier(
            first.refractive_kernel_weights - second.refractive_kernel_weights,
            weight,
        )
    if (
        first.refractive_kernel_flows is not None
        and second.refractive_kernel_flows is not None
    ):
        total = total + _masked_charbonnier(
            first.refractive_kernel_flows - second.refractive_kernel_flows,
            weight,
        )
    for name in ("surface_normal", "depth", "geometry_confidence"):
        first_value = getattr(first, name)
        second_value = getattr(second, name)
        if first_value is not None and second_value is not None:
            total = total + _masked_charbonnier(
                first_value - second_value,
                weight,
            )
    return total


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
        def select_field(name: str) -> Tensor | None:
            value = getattr(matter, name)
            return None if value is None else value[index : index + 1]

        return PhysicsMatterOutput(
            **{
                name: select_field(name)
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


def _temporal_loss(video: Tensor, flow_to_next: Tensor, support: Tensor) -> Tensor:
    """Align only visible, depth-consistent primary object surfaces."""

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
    return _masked_charbonnier(video[:, :-1] - aligned_next, support)


def _leave_one_out_background(target: RefractiveGroundTruth) -> tuple[Tensor, Tensor] | None:
    """Build B for frame t exclusively from object-free *other* frames.

    This is a measured background, not the trainable/inpainted prediction.
    The alpha check rejects antialiased transmission outside the binary mask.
    """

    if target.frames.shape[1] < 2 or target.object_mask is None:
        return None
    visible = target.object_mask < 0.5
    if target.alpha is not None:
        visible = visible & (target.alpha < 1e-3)
    visible = visible.to(target.frames.dtype)
    others_count = visible.sum(dim=1, keepdim=True) - visible
    others_sum = (target.frames * visible).sum(dim=1, keepdim=True) - (
        target.frames * visible
    )
    background = others_sum / others_count.clamp_min(1.0)
    return background, (others_count > 0).to(target.frames.dtype)


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

        object_coverage = _object_coverage_support(target)
        if object_coverage is not None:
            terms["mask"] = _mask_focal_dice(mask_logits, object_coverage)
        if target.trimap is not None:
            class_weight = trimap_logits.new_tensor((1.0, 2.0, 1.0))
            terms["trimap"] = F.cross_entropy(
                trimap_logits.flatten(0, 1),
                target.trimap.flatten(0, 1).long(),
                weight=class_weight,
            )
        if target.alpha is not None:
            mam2_alpha = _resize_video_logits(
                prediction.backbone.alpha_matte, (h, w)
            )
            terms["mam2_alpha"] = _charbonnier(mam2_alpha - target.alpha)
            terms["mam2_alpha_gradient"] = _gradient_loss(
                mam2_alpha, target.alpha
            )
            terms["alpha"] = _charbonnier(prediction.matter.alpha - target.alpha)
            terms["alpha_gradient"] = _gradient_loss(
                prediction.matter.alpha, target.alpha
            )

        target_g = target.premultiplied_foreground
        if target_g is None and target.straight_foreground is not None and target.alpha is not None:
            target_g = target.alpha * target.straight_foreground
        if target_g is not None:
            support = object_coverage if object_coverage is not None else 1.0
            terms["premultiplied_foreground"] = _masked_charbonnier(
                prediction.matter.premultiplied_foreground - target_g, support
            )
            if isinstance(support, Tensor):
                edge_support = F.max_pool2d(
                    support.flatten(0, 1).float(), 3, stride=1, padding=1,
                ).reshape_as(support)
                terms["premultiplied_foreground_gradient"] = _masked_gradient_loss(
                    prediction.matter.premultiplied_foreground, target_g,
                    edge_support,
                )
        if target.straight_foreground is not None:
            foreground_support = _straight_foreground_support(target)
            if foreground_support is not None:
                terms["straight_foreground"] = _masked_charbonnier(
                    prediction.matter.straight_foreground
                    - target.straight_foreground,
                    foreground_support,
                )
                terms["straight_foreground_gradient"] = _masked_gradient_loss(
                    prediction.matter.straight_foreground,
                    target.straight_foreground,
                    foreground_support,
                )
        if target.object_mask is not None:
            # The binary centre-ray mask may omit antialiased edge coverage;
            # supervise leakage only on certain background pixels.
            far_background = _far_background_support(target.object_mask)
            terms["foreground_background_leakage"] = _masked_charbonnier(
                prediction.matter.premultiplied_foreground, far_background
            )
            terms["flow_background_leakage"] = _masked_charbonnier(
                prediction.matter.refractive_flow / float(max(h, w, 1)),
                far_background,
            )
            terms["residual_background_leakage"] = _masked_charbonnier(
                prediction.matter.residual, far_background
            )

        validity: Tensor | float = (
            target.refractive_validity
            if target.refractive_validity is not None
            else 1.0
        )
        refractive_support: Tensor | float = validity
        if object_coverage is not None:
            refractive_support = object_coverage * validity

        if target.color_transmission is not None:
            color_support = _identifiable_color_transmission(target)
            terms["color_transmission"] = _masked_charbonnier(
                prediction.matter.color_transmission - target.color_transmission,
                color_support,
            )
            if self.weights.color_transmission_chroma > 0:
                # Penalize a neutral RGB transmission that has the correct
                # overall brightness but misses the object's color filtering.
                predicted_chroma = (
                    prediction.matter.color_transmission
                    - prediction.matter.color_transmission.mean(dim=2, keepdim=True)
                )
                target_chroma = (
                    target.color_transmission
                    - target.color_transmission.mean(dim=2, keepdim=True)
                )
                terms["color_transmission_chroma"] = _masked_charbonnier(
                    predicted_chroma - target_chroma, color_support,
                )

        if target.transmittance is not None:
            terms["transmittance"] = _masked_charbonnier(
                prediction.matter.transmittance - target.transmittance,
                object_coverage if object_coverage is not None else 1.0,
            )

        # An earlier frame may directly reveal Phi(x), even when the current
        # object occludes it. Compare transmitted radiance without dividing by
        # a potentially dark source pixel.
        past_evidence = _past_visible_refracted_background(target)
        if (
            past_evidence is not None
            and target_g is not None
            and target.residual is not None
        ):
            past_background, past_support = past_evidence
            observed_transmitted = target.frames - target_g - target.residual
            terms["temporal_visible_transmission"] = _masked_charbonnier(
                prediction.matter.transmittance * past_background
                - observed_transmitted,
                past_support,
            )
            # The GT-flow term above isolates T. A second photometric term
            # samples the same directly observed history at the *predicted*
            # source, so both T and flow receive a gradient. Require complete
            # bilinear visibility at both sources; never train on a hole fill.
            predicted_evidence = _past_visible_refracted_background(
                target, prediction.matter.refractive_flow
            )
            if predicted_evidence is not None:
                predicted_background, predicted_support = predicted_evidence
                joint_support = torch.minimum(
                    past_support, predicted_support.detach()
                )
                terms["temporal_visible_joint_recomposition"] = _masked_charbonnier(
                    prediction.matter.transmittance * predicted_background
                    - observed_transmitted,
                    joint_support,
                )

        if target.residual is not None:
            support = target.object_mask if target.object_mask is not None else 1.0
            terms["residual"] = _masked_charbonnier(
                prediction.matter.residual - target.residual, support
            )
        terms["residual_sparsity"] = prediction.matter.residual.abs().mean()

        geometry_support = target.object_mask
        if target.surface_normal is not None:
            if prediction.matter.surface_normal is None:
                raise ValueError(
                    "surface-normal GT was loaded but PAM geometry prediction is disabled"
                )
            normal_validity = (
                target.normal_validity
                if target.normal_validity is not None
                else torch.ones_like(target.surface_normal[:, :, :1])
            )
            if geometry_support is not None:
                normal_validity = normal_validity * geometry_support
            predicted_normal = F.normalize(
                prediction.matter.surface_normal, dim=2, eps=1e-6
            )
            target_normal = F.normalize(target.surface_normal, dim=2, eps=1e-6)
            cosine_error = 1.0 - (
                predicted_normal * target_normal
            ).sum(dim=2, keepdim=True).clamp(-1.0, 1.0)
            terms["surface_normal"] = _masked_charbonnier(
                cosine_error,
                normal_validity,
            )

        if target.depth is not None:
            if prediction.matter.depth is None:
                raise ValueError(
                    "depth GT was loaded but PAM geometry prediction is disabled"
                )
            depth_validity = (
                target.depth_validity
                if target.depth_validity is not None
                else torch.ones_like(target.depth)
            )
            if geometry_support is not None:
                depth_validity = depth_validity * geometry_support
            predicted_log_depth = prediction.matter.depth.clamp_min(1e-6).log()
            target_log_depth = target.depth.clamp_min(1e-6).log()
            terms["depth"] = _masked_charbonnier(
                predicted_log_depth - target_log_depth,
                depth_validity,
            )
            terms["depth_gradient"] = _masked_gradient_loss(
                predicted_log_depth,
                target_log_depth,
                depth_validity,
            )

        if prediction.matter.geometry_confidence is not None and (
            target.surface_normal is not None or target.depth is not None
        ):
            validity_candidates = [
                value
                for value in (target.normal_validity, target.depth_validity)
                if value is not None
            ]
            if validity_candidates:
                geometry_target = validity_candidates[0].to(
                    prediction.matter.geometry_confidence.dtype
                )
                for candidate in validity_candidates[1:]:
                    geometry_target = geometry_target * candidate.to(
                        geometry_target.dtype
                    )
            else:
                geometry_target = torch.ones_like(
                    prediction.matter.geometry_confidence
                )
            confidence_loss = _probability_binary_cross_entropy(
                prediction.matter.geometry_confidence,
                geometry_target,
                name="geometry_confidence",
            )
            confidence_support = (
                torch.ones_like(geometry_target)
                if geometry_support is None
                else geometry_support.to(geometry_target.dtype)
            )
            terms["geometry_confidence"] = (
                confidence_loss * confidence_support
            ).sum() / confidence_support.sum().clamp_min(1.0)

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
            inverse_only = (
                prediction.background.inverse_support
                & ~prediction.background.direct_support
            )
            terms["background_inverse_refined"] = _masked_charbonnier(
                prediction.background.background - background_gt,
                inverse_only,
            )
            terms["background_true_hole"] = _masked_charbonnier(
                prediction.background.background - background_gt,
                prediction.background.true_hole,
            )
            terms["background_frequency"] = _masked_frequency_loss(
                prediction.background.background,
                background_gt,
                prediction.background.true_hole,
            )

        # A frame cannot supply its own counterfactual background. This
        # independent observation path gives tau/flow a cross-frame signal
        # without letting the learned B cancel its own render error.
        holdout = _leave_one_out_background(target)
        if holdout is not None:
            holdout_background, holdout_coverage = holdout
            kernel_weights = prediction.matter.refractive_kernel_weights
            kernel_flows = prediction.matter.refractive_kernel_flows
            if kernel_weights is not None:
                if kernel_flows is None:
                    raise ValueError("refractive kernel weights require kernel flows")
                sampled_background = warp_background_kernel(
                    holdout_background, kernel_flows, kernel_weights
                )
                sampled_coverage = warp_background_kernel(
                    holdout_coverage.expand(-1, -1, 3, -1, -1),
                    kernel_flows,
                    kernel_weights,
                )[:, :, :1]
            else:
                sampled_background = warp_background(
                    holdout_background, prediction.matter.refractive_flow
                )
                sampled_coverage = warp_background(
                    holdout_coverage, prediction.matter.refractive_flow
                )
            support = (sampled_coverage > 1.0 - 1e-4).to(target.frames.dtype)
            if target.object_mask is not None:
                support = support * target.object_mask
            if target.refractive_validity is not None:
                support = support * target.refractive_validity
            support = support * _identifiable_color_transmission(target)
            foreground = (
                target_g if target_g is not None
                else prediction.matter.premultiplied_foreground.detach()
            )
            residual = (
                target.residual if target.residual is not None
                else prediction.matter.residual.detach()
            )
            holdout_render = (
                foreground
                + prediction.matter.transmittance * sampled_background
                + residual
            )
            terms["temporal_holdout_render"] = _masked_charbonnier(
                holdout_render - target.frames, support.detach()
            )

        terms["render"] = _charbonnier(
            prediction.reconstructed_frames - target.frames
        )
        if object_coverage is not None:
            terms["render_object"] = _masked_charbonnier(
                prediction.reconstructed_frames - target.frames,
                object_coverage,
            )
        terms["render_multiscale"] = _multiscale_render_loss(
            prediction.reconstructed_frames, target.frames
        )
        terms["flow_smoothness"] = _edge_aware_flow_smoothness(
            prediction.matter.refractive_flow,
            target.frames,
            target.refractive_flow,
            refractive_support,
        )
        terms["flow_out_of_bounds"] = _flow_out_of_bounds(
            prediction.matter.refractive_flow
        )
        kernel_weights = prediction.matter.refractive_kernel_weights
        kernel_flows = prediction.matter.refractive_kernel_flows
        if kernel_weights is not None and kernel_flows is not None:
            expected = prediction.matter.refractive_flow[:, :, None]
            squared_radius = (kernel_flows - expected).square().sum(dim=3)
            terms["refractive_kernel_spread"] = (
                kernel_weights * squared_radius
            ).sum(dim=2).mean() / float(max(h, w, 1) ** 2)

            weight_dx = kernel_weights[..., :, 1:] - kernel_weights[..., :, :-1]
            weight_dy = kernel_weights[..., 1:, :] - kernel_weights[..., :-1, :]
            flow_dx = kernel_flows[..., :, 1:] - kernel_flows[..., :, :-1]
            flow_dy = kernel_flows[..., 1:, :] - kernel_flows[..., :-1, :]
            terms["refractive_kernel_smoothness"] = (
                weight_dx.abs().mean()
                + weight_dy.abs().mean()
                + flow_dx.abs().mean() / float(max(h, w, 1))
                + flow_dy.abs().mean() / float(max(h, w, 1))
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
        confidence_loss = _probability_binary_cross_entropy(
            prediction.matter.confidence,
            confidence_target,
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
            if target.temporal_flow_validity is None:
                raise ValueError("temporal flow requires occlusion-aware validity")
            temporal_support = target.temporal_flow_validity
            # A same-surface point does not have invariant F, tau or refractive
            # displacement under a view-angle change. Only object occupancy
            # and locally stable GT alpha are regularized across frames.
            terms["temporal"] = _temporal_loss(
                mask_logits.sigmoid(), target.temporal_flow_to_next,
                temporal_support,
            )
            if target.alpha is not None:
                aligned_alpha_gt = warp_background(
                    target.alpha[:, 1:], target.temporal_flow_to_next
                )
                stable_alpha = (
                    (target.alpha[:, :-1] - aligned_alpha_gt).abs() < 0.05
                ).to(temporal_support.dtype)
                terms["temporal"] = terms["temporal"] + _temporal_loss(
                    prediction.matter.alpha, target.temporal_flow_to_next,
                    temporal_support * stable_alpha,
                )

        total = target.frames.new_zeros(())
        for name, value in terms.items():
            total = total + getattr(self.weights, name) * value
        terms["total"] = total
        return terms
