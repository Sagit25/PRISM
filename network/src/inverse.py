from __future__ import annotations

import torch
from torch import Tensor

from .background import RefractiveBackgroundEvidence
from .config import BackgroundConfig
from .matter import PhysicsMatterOutput


def _bilinear_splat(
    values: Tensor,
    flow: Tensor,
    weights: Tensor,
    *,
    eps: float,
) -> RefractiveBackgroundEvidence:
    """Forward-splat target pixels to ``x + flow(x)`` on a shared canvas.

    Bilinear weights remain differentiable with respect to values, confidence,
    transmission and the fractional component of the refractive flow. Integer
    cell selection follows the standard piecewise-differentiable splatting
    convention.
    """

    if values.ndim != 5 or values.shape[2] != 3:
        raise ValueError("values must have shape [B,T,3,H,W]")
    if flow.shape != (values.shape[0], values.shape[1], 2, *values.shape[-2:]):
        raise ValueError("flow must have shape [B,T,2,H,W]")
    if weights.shape != (values.shape[0], values.shape[1], 1, *values.shape[-2:]):
        raise ValueError("weights must have shape [B,T,1,H,W]")

    b, t, _, h, w = values.shape
    dtype, device = values.dtype, values.device
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    target_x = xx.view(1, 1, h, w) + flow[:, :, 0]
    target_y = yy.view(1, 1, h, w) + flow[:, :, 1]
    x0 = target_x.floor()
    y0 = target_y.floor()
    x1 = x0 + 1.0
    y1 = y0 + 1.0

    source_values = values.permute(0, 1, 3, 4, 2).reshape(-1, 3)
    source_weight = weights.reshape(-1)
    batch_index = (
        torch.arange(b, device=device)
        .view(b, 1, 1, 1)
        .expand(b, t, h, w)
        .reshape(-1)
    )
    color_sum = values.new_zeros((b * h * w, 3))
    weight_sum = values.new_zeros((b * h * w, 1))

    neighbors = (
        (x0, y0, (x1 - target_x) * (y1 - target_y)),
        (x1, y0, (target_x - x0) * (y1 - target_y)),
        (x0, y1, (x1 - target_x) * (target_y - y0)),
        (x1, y1, (target_x - x0) * (target_y - y0)),
    )
    for x_neighbor, y_neighbor, bilinear_weight in neighbors:
        valid = (
            (x_neighbor >= 0)
            & (x_neighbor < w)
            & (y_neighbor >= 0)
            & (y_neighbor < h)
        )
        x_index = x_neighbor.clamp(0, w - 1).long().reshape(-1)
        y_index = y_neighbor.clamp(0, h - 1).long().reshape(-1)
        destination = batch_index * (h * w) + y_index * w + x_index
        contribution_weight = (
            source_weight
            * bilinear_weight.reshape(-1)
            * valid.reshape(-1).to(dtype)
        )
        expanded_index = destination[:, None].expand(-1, 3)
        color_sum = color_sum.scatter_add(
            0, expanded_index, source_values * contribution_weight[:, None]
        )
        weight_sum = weight_sum.scatter_add(
            0, destination[:, None], contribution_weight[:, None]
        )

    color_sum = color_sum.reshape(b, h, w, 3).permute(0, 3, 1, 2)
    weight_sum = weight_sum.reshape(b, h, w, 1).permute(0, 3, 1, 2)
    background = color_sum / weight_sum.clamp_min(eps)
    return RefractiveBackgroundEvidence(background=background, weight=weight_sum)


def inverse_refractive_splat(
    frames: Tensor,
    matter: PhysicsMatterOutput,
    object_support: Tensor,
    config: BackgroundConfig | None = None,
) -> RefractiveBackgroundEvidence:
    """Recover shared-background samples from transparent-object interiors.

    For each frame/pixel, the operator gives

    ``B_cf(Phi(x)) = (I(x) - G(x) - R(x)) / tau(x)``.

    The recovered radiance is splatted to ``Phi(x)=x+u(x)`` and robustly
    weighted by object support, predicted confidence and transmission. Direct
    object-free observations are fused separately and retain priority.
    """

    cfg = config or BackgroundConfig()
    if frames.shape != matter.premultiplied_foreground.shape:
        raise ValueError("matter foreground must match frames")
    if matter.transmittance.shape != frames.shape:
        raise ValueError("RGB transmittance must match frames")
    if matter.residual.shape != frames.shape:
        raise ValueError("matter residual must match frames")
    if object_support.shape != (
        frames.shape[0], frames.shape[1], 1, *frames.shape[-2:]
    ):
        raise ValueError("object_support must have shape [B,T,1,H,W]")

    tau = matter.transmittance
    mean_tau = tau.mean(dim=2, keepdim=True)
    valid_tau = (mean_tau >= cfg.inverse_min_transmittance).to(frames.dtype)
    recovered = (
        frames - matter.premultiplied_foreground - matter.residual
    ) / tau.clamp_min(cfg.inverse_min_transmittance)
    recovered = recovered.clamp(0.0, cfg.inverse_max_radiance)
    confidence = matter.confidence.clamp(0, 1)
    weight = (
        object_support.clamp(0, 1)
        * confidence
        * mean_tau
        * valid_tau
        * cfg.inverse_weight_scale
    )
    return _bilinear_splat(
        recovered,
        matter.refractive_flow,
        weight,
        eps=cfg.eps,
    )
