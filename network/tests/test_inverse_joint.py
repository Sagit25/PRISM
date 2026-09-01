import pytest

torch = pytest.importorskip("torch")

from refractive_mam2 import (
    BackgroundConfig,
    MaskedTemporalBackground,
    PhysicsMatterOutput,
    RefractiveBackgroundEvidence,
    inverse_refractive_splat,
    reusable_operator_consistency,
    reusable_operator_consistency_in_batch,
)


def _matter(frames, g, tau):
    b, t, _, h, w = frames.shape
    return PhysicsMatterOutput(
        alpha=torch.zeros(b, t, 1, h, w, device=frames.device),
        premultiplied_foreground=g,
        straight_foreground=torch.zeros_like(frames),
        color_transmission=tau,
        transmittance=tau,
        refractive_flow=torch.zeros(b, t, 2, h, w, device=frames.device),
        residual=torch.zeros_like(frames),
        confidence=torch.ones(b, t, 1, h, w, device=frames.device),
    )


def test_inverse_splat_recovers_background_from_object_interior() -> None:
    background = torch.rand(1, 3, 5, 7)
    frames = background[:, None].expand(-1, 3, -1, -1, -1).clone()
    tau = torch.ones_like(frames)
    evidence = inverse_refractive_splat(
        frames,
        _matter(frames, torch.zeros_like(frames), tau),
        torch.ones(1, 3, 1, 5, 7),
        BackgroundConfig(exclusion_dilation=1),
    )
    assert torch.allclose(evidence.background, background, atol=1e-6)
    assert torch.all(evidence.weight > 0)


def test_inverse_reconstruction_is_differentiable_to_operator() -> None:
    frames = torch.rand(1, 2, 3, 4, 6)
    g = torch.zeros_like(frames, requires_grad=True)
    tau = torch.full_like(frames, 0.8, requires_grad=True)
    evidence = inverse_refractive_splat(
        frames,
        _matter(frames, g, tau),
        torch.ones(1, 2, 1, 4, 6),
        BackgroundConfig(exclusion_dilation=1),
    )
    evidence.background.mean().backward()
    assert g.grad is not None and torch.isfinite(g.grad).all()
    assert tau.grad is not None and torch.isfinite(tau.grad).all()


def test_inverse_splat_uses_refractive_destination_coordinates() -> None:
    background = torch.arange(5, dtype=torch.float32).view(1, 1, 1, 5).expand(1, 3, 1, 5)
    # A +1 horizontal refractive flow means I(x) observes B(x+1).
    frames = torch.zeros(1, 1, 3, 1, 5)
    frames[..., :4] = background[:, None, ..., 1:]
    tau = torch.ones_like(frames)
    matter = _matter(frames, torch.zeros_like(frames), tau)
    matter.refractive_flow[:, :, 0] = 1
    support = torch.zeros(1, 1, 1, 1, 5)
    support[..., :4] = 1
    evidence = inverse_refractive_splat(
        frames,
        matter,
        support,
        BackgroundConfig(exclusion_dilation=1, inverse_max_radiance=5.0),
    )
    assert torch.allclose(evidence.background[..., 1:], background[..., 1:])
    assert torch.all(evidence.weight[..., 1:] > 0)
    assert torch.all(evidence.weight[..., 0] == 0)


def test_inverse_evidence_fills_only_unobserved_direct_hole() -> None:
    background = torch.rand(1, 3, 5, 5)
    frames = background[:, None].expand(-1, 2, -1, -1, -1).clone()
    exclusion = torch.zeros(1, 2, 1, 5, 5)
    exclusion[:, :, :, 2, 2] = 1
    model = MaskedTemporalBackground(
        BackgroundConfig(exclusion_dilation=1, use_temporal_completion=False)
    )
    direct = model.observe(frames, exclusion)
    inverse_weight = torch.zeros(1, 1, 5, 5)
    inverse_weight[:, :, 2, 2] = 1
    output = model.fuse(
        direct,
        RefractiveBackgroundEvidence(background=background, weight=inverse_weight),
    )
    assert torch.allclose(output.background, background, atol=1e-6)
    assert output.inverse_coverage[0, 0, 2, 2] > 0


def test_reusable_operator_consistency_is_zero_for_same_operator() -> None:
    frames = torch.rand(1, 2, 3, 4, 5)
    tau = torch.ones_like(frames)
    operator = _matter(frames, torch.zeros_like(frames), tau)
    loss = reusable_operator_consistency(operator, operator)
    # Charbonnier has a small non-zero robust-loss floor per term.
    assert loss < 0.01


def test_reusable_operator_consistency_uses_pair_group_ids() -> None:
    frames = torch.rand(2, 1, 3, 4, 5)
    operator = _matter(frames, torch.zeros_like(frames), torch.ones_like(frames))
    loss = reusable_operator_consistency_in_batch(
        operator, ["shared_operator", "shared_operator"]
    )
    assert loss < 0.01
