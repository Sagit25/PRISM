import pytest

torch = pytest.importorskip("torch")

from refractive_mam2 import MatterConfig, PhysicsAwareMatter


def test_matter_predicts_g_directly_and_derives_straight_foreground() -> None:
    config = MatterConfig(
        feature_channels=8,
        width=8,
        hard_support_at_inference=False,
        straight_foreground_min_alpha=0.01,
    )
    model = PhysicsAwareMatter(config).eval()
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
        model.head.bias[0] = -20.0
        model.head.bias[1:4] = 5.0
        model.head.bias[7:10] = 20.0

    frames = torch.rand(1, 1, 3, 8, 8)
    trimap = torch.zeros(1, 1, 3, 4, 4)
    trimap[:, :, 1] = 10.0
    output = model(
        frames,
        torch.rand_like(frames),
        trimap,
        torch.rand(1, 1, 8, 4, 4),
        torch.zeros(1, 1, 1, 8, 8),
    )
    assert output.alpha.max() < 1e-6
    assert output.premultiplied_foreground.mean() > 0.9
    assert torch.count_nonzero(output.straight_foreground) == 0
    assert output.transmittance.shape == frames.shape
    assert output.transmittance.mean() > 0.99
    assert torch.count_nonzero(output.residual) == 0


def test_flow_range_scales_with_resolution() -> None:
    config = MatterConfig(
        feature_channels=8,
        width=8,
        hard_support_at_inference=False,
        flow_parameterization="resolution_fraction",
        max_refractive_flow_fraction=0.25,
    )
    model = PhysicsAwareMatter(config).eval()
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
        model.head.bias[4:6] = 20.0
    frames = torch.rand(1, 1, 3, 32, 64)
    trimap = torch.zeros(1, 1, 3, 8, 16)
    trimap[:, :, 1] = 20.0
    output = model(
        frames,
        torch.rand_like(frames),
        trimap,
        torch.rand(1, 1, 8, 8, 16),
        torch.zeros(1, 1, 1, 32, 64),
    )
    assert torch.allclose(
        output.refractive_flow[0, 0, :, 0, 0],
        torch.tensor((63.0 * 0.25, 31.0 * 0.25)),
        atol=1e-4,
    )


def test_scalar_transmission_and_no_residual_ablation() -> None:
    config = MatterConfig(
        feature_channels=8,
        width=8,
        hard_support_at_inference=False,
        use_rgb_transmission=False,
        use_residual=False,
    )
    model = PhysicsAwareMatter(config).eval()
    frames = torch.rand(1, 1, 3, 8, 8)
    trimap = torch.zeros(1, 1, 3, 4, 4)
    trimap[:, :, 1] = 10
    output = model(
        frames,
        torch.rand_like(frames),
        trimap,
        torch.rand(1, 1, 8, 4, 4),
        torch.zeros(1, 1, 1, 8, 8),
    )
    assert torch.allclose(
        output.color_transmission[:, :, 0],
        output.color_transmission[:, :, 1],
    )
    assert torch.allclose(
        output.color_transmission[:, :, 1],
        output.color_transmission[:, :, 2],
    )
    assert torch.count_nonzero(output.residual) == 0
