import pytest

torch = pytest.importorskip("torch")

from refractive_mam2.renderer import recompose, warp_background


def test_zero_flow_is_identity() -> None:
    background = torch.rand(2, 3, 5, 7)
    flow = torch.zeros(2, 2, 5, 7)
    assert torch.allclose(warp_background(background, flow), background, atol=1e-6)


def test_recomposition_limits() -> None:
    background = torch.rand(1, 2, 3, 6, 8)
    flow = torch.zeros(1, 2, 2, 6, 8)
    zero_alpha = torch.zeros(1, 2, 1, 6, 8)
    zero_g = torch.zeros_like(background)
    transparent, _ = recompose(zero_alpha, zero_g, background, flow)
    assert torch.allclose(transparent, background, atol=1e-6)

    one_alpha = torch.ones_like(zero_alpha)
    foreground = torch.rand_like(background)
    opaque, _ = recompose(one_alpha, foreground, background, flow)
    assert torch.allclose(opaque, foreground, atol=1e-6)


def test_colored_transmission_and_residual() -> None:
    background = torch.ones(1, 1, 3, 3, 4)
    alpha = torch.zeros(1, 1, 1, 3, 4)
    foreground = torch.zeros_like(background)
    flow = torch.zeros(1, 1, 2, 3, 4)
    tau = background.new_tensor([0.25, 0.5, 0.75]).view(1, 1, 3, 1, 1)
    residual = torch.full_like(background, 0.1)
    composite, _ = recompose(
        alpha,
        foreground,
        background,
        flow,
        transmittance=tau,
        residual=residual,
    )
    assert torch.allclose(composite, tau + residual, atol=1e-6)


def test_out_of_bounds_sampling_matches_rctrans_zero_padding() -> None:
    background = torch.ones(1, 3, 3, 4)
    flow = torch.zeros(1, 2, 3, 4)
    flow[:, 0] = 100.0
    warped = warp_background(background, flow)
    assert torch.count_nonzero(warped) == 0
