from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from refractive_mam2.geometry_posthead import PRISMGeometryPostHead
from refractive_mam2.train import _geometry_only_terms
from refractive_mam2.config import LossWeights


def _prediction(batch: int = 1, time: int = 2, side: int = 16):
    video = lambda channels: torch.rand(batch, time, channels, side, side)
    foreground = video(3).requires_grad_()
    flow = video(2).requires_grad_()
    matter = SimpleNamespace(
        straight_foreground=foreground,
        premultiplied_foreground=video(3),
        alpha=video(1),
        color_transmission=video(3),
        transmittance=video(3),
        refractive_flow=flow,
        residual=video(3),
        confidence=video(1),
        object_support_logits=None,
    )
    output = SimpleNamespace(
        matter=matter,
        backbone=SimpleNamespace(
            mask_logits=video(1), trimap_logits=video(3), alpha_matte=video(1),
        ),
        background=SimpleNamespace(
            background=torch.rand(batch, 3, side, side),
            uncertainty=torch.rand(batch, 1, side, side),
        ),
        refracted_background=video(3),
        reconstructed_frames=video(3),
    )
    return video(3), output


def test_post_head_uses_final_foreground_and_flow_with_finite_gradients():
    frames, prediction = _prediction()
    head = PRISMGeometryPostHead(width=16)
    with torch.no_grad():
        head.head.weight.normal_(std=1e-3)
    geometry = head(frames, prediction)
    assert geometry.depth.shape == (1, 2, 1, 16, 16)
    assert geometry.surface_normal.shape == (1, 2, 3, 16, 16)
    assert bool(torch.isfinite(geometry.depth).all())
    assert bool((geometry.depth > 0).all())
    assert torch.allclose(
        torch.linalg.vector_norm(geometry.surface_normal, dim=2),
        torch.ones_like(geometry.depth[:, :, 0]), atol=1e-5,
    )
    (geometry.depth.mean() + geometry.surface_normal.square().mean()).backward()
    assert prediction.matter.straight_foreground.grad is not None
    assert prediction.matter.refractive_flow.grad is not None
    assert prediction.matter.straight_foreground.grad.abs().sum() > 0
    assert prediction.matter.refractive_flow.grad.abs().sum() > 0


def test_geometry_only_objective_excludes_frozen_prism_losses():
    normal = torch.tensor(0.2, requires_grad=True)
    losses = {
        "surface_normal": normal,
        "depth": torch.tensor(0.3),
        "depth_gradient": torch.tensor(0.4),
        "geometry_confidence": torch.tensor(0.5),
        "render_object": torch.tensor(99.0),
        "total": torch.tensor(100.0),
    }
    selected = _geometry_only_terms(losses, LossWeights())
    assert "render_object" not in selected
    selected["total"].backward()
    assert normal.grad is not None and normal.grad > 0
