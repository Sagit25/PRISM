from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _flatten_video(tensor: Tensor) -> tuple[Tensor, tuple[int, int] | None]:
    if tensor.ndim == 4:
        return tensor, None
    if tensor.ndim != 5:
        raise ValueError("expected [B,C,H,W] or [B,T,C,H,W]")
    b, t, c, h, w = tensor.shape
    return tensor.reshape(b * t, c, h, w), (b, t)


def _restore_video(tensor: Tensor, video_shape: tuple[int, int] | None) -> Tensor:
    if video_shape is None:
        return tensor
    b, t = video_shape
    return tensor.reshape(b, t, *tensor.shape[1:])


def warp_background(
    background: Tensor,
    flow_pixels: Tensor,
    *,
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> Tensor:
    """Sample background at x + flow(x), with flow in pixel units.

    The convention is target-to-source: flow[...,0,:,:] is dx and
    flow[...,1,:,:] is dy at each output/target pixel.  Out-of-bounds samples
    default to zero, matching RCTrans/OpenCV ``BORDER_CONSTANT`` rendering.
    """

    flat_bg, video_shape = _flatten_video(background)
    flat_flow, flow_video_shape = _flatten_video(flow_pixels)
    if video_shape != flow_video_shape:
        raise ValueError("background and flow must have matching batch/time dimensions")
    if flat_flow.shape[1] != 2:
        raise ValueError("flow must have two channels (dx, dy)")
    if flat_bg.shape[0] != flat_flow.shape[0] or flat_bg.shape[-2:] != flat_flow.shape[-2:]:
        raise ValueError("background and flow spatial dimensions must match")

    n, _, h, w = flat_bg.shape
    dtype = flat_bg.dtype
    device = flat_bg.device
    y, x = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    x = x.unsqueeze(0).expand(n, -1, -1) + flat_flow[:, 0]
    y = y.unsqueeze(0).expand(n, -1, -1) + flat_flow[:, 1]

    if align_corners:
        x = 2.0 * x / max(w - 1, 1) - 1.0
        y = 2.0 * y / max(h - 1, 1) - 1.0
    else:
        x = 2.0 * (x + 0.5) / w - 1.0
        y = 2.0 * (y + 0.5) / h - 1.0
    grid = torch.stack((x, y), dim=-1)
    warped = F.grid_sample(
        flat_bg,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    return _restore_video(warped, video_shape)


def recompose(
    alpha: Tensor,
    premultiplied_foreground: Tensor,
    counterfactual_background: Tensor,
    refractive_flow: Tensor,
    *,
    transmittance: Tensor | None = None,
    residual: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Differentiable colored refractive compositing.

    ``I_hat = G + tau * sample(B_cf, x + u) + R``.

    ``tau`` is RGB so colored transparent materials can attenuate a new
    background per channel.  Passing no ``transmittance`` preserves the
    colorless legacy model with ``tau = 1-alpha``.  ``R`` is an object-attached
    bounded residual and defaults to zero.
    """

    if alpha.shape[-3] != 1:
        raise ValueError("alpha must have one channel")
    if premultiplied_foreground.shape[-3] != 3:
        raise ValueError("premultiplied_foreground must have three channels")
    if transmittance is None:
        transmittance = 1.0 - alpha
    if transmittance.shape[-3] not in (1, 3):
        raise ValueError("transmittance must have one or three channels")
    if residual is None:
        residual = torch.zeros_like(premultiplied_foreground)
    if residual.shape != premultiplied_foreground.shape:
        raise ValueError("residual must match premultiplied_foreground")
    refracted_background = warp_background(counterfactual_background, refractive_flow)
    composite = (
        premultiplied_foreground
        + transmittance * refracted_background
        + residual
    )
    return composite, refracted_background
