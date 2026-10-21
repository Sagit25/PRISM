import pytest

torch = pytest.importorskip("torch")
F = pytest.importorskip("torch.nn.functional")

from refractive_mam2 import MatterConfig, PhysicsAwareMatter, recompose


def test_matter_reports_nonfinite_background_before_convolution() -> None:
    model = PhysicsAwareMatter(MatterConfig(feature_channels=8, width=8))
    frames = torch.rand(1, 1, 3, 8, 8)
    background = torch.rand_like(frames)
    background[0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="counterfactual_background"):
        model(
            frames,
            background,
            torch.zeros(1, 1, 3, 4, 4),
            torch.rand(1, 1, 8, 4, 4),
            torch.zeros(1, 1, 1, 8, 8),
        )


def test_matter_reports_nonfinite_head_output_before_confidence_loss() -> None:
    model = PhysicsAwareMatter(MatterConfig(feature_channels=8, width=8))
    with torch.no_grad():
        model.head.bias[6] = float("nan")
    frames = torch.rand(1, 1, 3, 8, 8)
    with pytest.raises(FloatingPointError, match="PAM raw head output"):
        model(
            frames,
            torch.rand_like(frames),
            torch.zeros(1, 1, 3, 4, 4),
            torch.rand(1, 1, 8, 4, 4),
            torch.zeros(1, 1, 1, 8, 8),
        )


def test_matter_numeric_trace_names_first_invalid_stage(monkeypatch) -> None:
    monkeypatch.setenv("PRISM_NUMERIC_TRACE", "1")
    model = PhysicsAwareMatter(MatterConfig(feature_channels=8, width=8))
    with torch.no_grad():
        model.image_stem[0].bias[0] = float("nan")
    frames = torch.rand(1, 1, 3, 8, 8)
    with pytest.raises(FloatingPointError, match="PAM image_stem"):
        model(
            frames,
            torch.rand_like(frames),
            torch.zeros(1, 1, 3, 4, 4),
            torch.rand(1, 1, 8, 4, 4),
            torch.zeros(1, 1, 1, 8, 8),
        )


def test_matter_predicts_straight_foreground_and_premultiplies_with_alpha() -> None:
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
    assert output.straight_foreground.mean() > 1.0
    assert output.premultiplied_foreground.max() < 1e-6
    torch.testing.assert_close(
        output.premultiplied_foreground,
        output.alpha * output.straight_foreground,
    )
    assert output.transmittance.shape == frames.shape
    assert output.transmittance.mean() > 0.99
    assert torch.count_nonzero(output.residual) == 0


def test_straight_foreground_is_zero_outside_predicted_object() -> None:
    model = PhysicsAwareMatter(MatterConfig(feature_channels=8, width=8)).eval()
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
        model.head.bias[1:4] = 5.0
    frames = torch.rand(1, 1, 3, 8, 8)
    mask_logits = torch.full((1, 1, 1, 8, 8), -20.0)
    mask_logits[..., 2:6, 2:6] = 20.0
    output = model(
        frames,
        torch.rand_like(frames),
        torch.zeros(1, 1, 3, 8, 8),
        torch.rand(1, 1, 8, 8, 8),
        torch.zeros(1, 1, 1, 8, 8),
        object_mask_logits=mask_logits,
    )
    assert torch.count_nonzero(output.straight_foreground[..., :2, :]) == 0
    assert output.straight_foreground[..., 3, 3].mean() > 0.9
    torch.testing.assert_close(
        output.premultiplied_foreground,
        output.alpha * output.straight_foreground,
    )


def test_antialiased_boundary_is_retained_without_far_background_leakage() -> None:
    model = PhysicsAwareMatter(
        MatterConfig(
            feature_channels=8, width=8, antialiased_boundary_support=True,
        )
    ).eval()
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
        model.head.bias[4:6] = 4.0
    mask_logits = torch.full((1, 1, 1, 8, 8), -20.0)
    mask_logits[..., 2:6, 2:6] = 20.0
    matte = torch.zeros(1, 1, 1, 8, 8)
    matte[..., 1, 3] = 0.2  # one antialiased pixel beyond the center-ray mask
    matte[..., 0, 0] = 1.0  # a spurious far-away matte pixel must not leak
    output = model(
        torch.rand(1, 1, 3, 8, 8),
        torch.rand(1, 1, 3, 8, 8),
        torch.zeros(1, 1, 3, 8, 8),
        torch.rand(1, 1, 8, 8, 8),
        torch.zeros(1, 1, 1, 8, 8),
        mam2_alpha=matte,
        object_mask_logits=mask_logits,
    )
    assert output.alpha[0, 0, 0, 1, 3] > 0
    assert output.premultiplied_foreground[0, 0, :, 1, 3].sum() > 0
    assert output.refractive_flow[0, 0, :, 1, 3].abs().sum() > 0
    assert output.alpha[0, 0, 0, 0, 0] == 0
    assert output.refractive_flow[0, 0, :, 0, 0].abs().sum() == 0


def test_alpha_refinement_can_correct_a_filled_object_interior() -> None:
    model = PhysicsAwareMatter(MatterConfig(feature_channels=8, width=8)).eval()
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
        model.head.bias[0] = -2.0
    trimap_logits = torch.zeros(1, 1, 3, 8, 8)
    trimap_logits[:, :, 2] = 20.0  # definite foreground, not unknown
    output = model(
        torch.rand(1, 1, 3, 8, 8),
        torch.rand(1, 1, 3, 8, 8),
        trimap_logits,
        torch.rand(1, 1, 8, 8, 8),
        torch.zeros(1, 1, 1, 8, 8),
        mam2_alpha=torch.full((1, 1, 1, 8, 8), 0.6),
        object_mask_logits=torch.full((1, 1, 1, 8, 8), 20.0),
    )
    assert output.alpha[0, 0, 0, 4, 4] < 0.2


def test_flow_is_hard_zero_outside_object_even_with_soft_support_ablation() -> None:
    model = PhysicsAwareMatter(
        MatterConfig(feature_channels=8, width=8, hard_object_support=False)
    ).eval()
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
        model.head.bias[4:6] = 4.0
    frames = torch.rand(1, 1, 3, 8, 8)
    trimap = torch.zeros(1, 1, 3, 8, 8)
    trimap[:, :, 0] = 4.0
    trimap[:, :, 1, 2:6, 2:6] = 8.0
    output = model(
        frames,
        torch.rand_like(frames),
        trimap,
        torch.rand(1, 1, 8, 8, 8),
        torch.zeros(1, 1, 1, 8, 8),
    )
    assert torch.count_nonzero(output.refractive_flow[..., :2, :]) == 0
    assert torch.count_nonzero(output.straight_foreground[..., :2, :]) == 0
    assert output.refractive_flow[..., 3, 3].abs().sum() > 0


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


def test_flow_refiner_starts_at_coarse_flow_and_uses_render_error() -> None:
    config = MatterConfig(
        feature_channels=8,
        width=8,
        encoder_depths=(1, 1, 1),
        temporal_blocks=0,
        hard_support_at_inference=False,
        refine_refractive_flow=True,
    )
    model = PhysicsAwareMatter(config).eval()
    assert model.flow_refiner is not None
    assert model.ARCHITECTURE.endswith("-flow-residual-v1")
    frames = torch.rand(1, 2, 3, 8, 8)
    background = torch.rand_like(frames)
    trimap = torch.zeros(1, 2, 3, 4, 4)
    trimap[:, :, 1] = 10.0
    features = torch.rand(1, 2, 8, 4, 4)
    uncertainty = torch.zeros(1, 2, 1, 8, 8)
    refiner_inputs: list[torch.Tensor] = []
    hook = model.flow_refiner.register_forward_pre_hook(
        lambda _module, args: refiner_inputs.append(args[0].detach())
    )
    try:
        output = model(frames, background, trimap, features, uncertainty)
    finally:
        hook.remove()
    assert output.coarse_refractive_flow is not None
    assert torch.equal(output.refractive_flow, output.coarse_refractive_flow)
    assert len(refiner_inputs) == 1
    coarse_reconstruction, _ = recompose(
        output.alpha,
        output.premultiplied_foreground,
        background,
        output.coarse_refractive_flow,
        transmittance=output.transmittance,
        residual=output.residual,
    )
    assert torch.allclose(
        refiner_inputs[0][:, 6:9],
        (frames - coarse_reconstruction).flatten(0, 1),
        atol=1e-6,
    )


def test_flow_refiner_corrects_only_supported_pixels_and_receives_gradients() -> None:
    config = MatterConfig(
        feature_channels=8,
        width=8,
        encoder_depths=(1, 1, 1),
        temporal_blocks=0,
        refine_refractive_flow=True,
    )
    model = PhysicsAwareMatter(config).eval()
    assert model.flow_refiner is not None
    with torch.no_grad():
        model.flow_refiner.head.bias[0] = 0.5
    frames = torch.rand(1, 1, 3, 8, 8)
    trimap = torch.zeros(1, 1, 3, 4, 4)
    trimap[:, :, 0] = 20.0
    trimap[:, :, 1, 1:3, 1:3] = 30.0
    output = model(
        frames,
        torch.rand_like(frames),
        trimap,
        torch.rand(1, 1, 8, 4, 4),
        torch.zeros(1, 1, 1, 8, 8),
    )
    assert output.coarse_refractive_flow is not None
    delta = output.refractive_flow - output.coarse_refractive_flow
    assert delta[:, :, 0].max() > 0
    assert delta[:, :, 0].max() <= 7 * config.flow_refinement_max_fraction
    assert torch.count_nonzero(delta[:, :, :, 0, 0]) == 0
    delta.square().mean().backward()
    assert model.flow_refiner.head.weight.grad is not None
    assert torch.isfinite(model.flow_refiner.head.weight.grad).all()


def test_predicted_object_mask_strictly_limits_physics_outputs() -> None:
    config = MatterConfig(
        feature_channels=8,
        width=8,
        encoder_depths=(1, 1, 1),
        temporal_blocks=0,
        refractive_kernel_size=3,
        refine_refractive_flow=True,
        predict_geometry=True,
    )
    model = PhysicsAwareMatter(config).eval()
    frames = torch.rand(1, 1, 3, 8, 8)
    background = torch.rand_like(frames)
    # The trimap says foreground everywhere; only the predicted object mask
    # and its one-pixel MAM2-supported antialias shell may carry physics.
    trimap = torch.zeros(1, 1, 3, 8, 8)
    trimap[:, :, 1] = 20.0
    mask_logits = torch.full((1, 1, 1, 8, 8), -10.0)
    mask_logits[:, :, :, 2:6, 2:6] = 10.0
    output = model(
        frames,
        background,
        trimap,
        torch.rand(1, 1, 8, 4, 4),
        torch.zeros(1, 1, 1, 8, 8),
        mam2_alpha=torch.full((1, 1, 1, 8, 8), 0.7),
        object_mask_logits=mask_logits,
    )
    allowed = F.max_pool2d(
        (mask_logits >= 0).flatten(0, 1).float(), 3, stride=1, padding=1,
    ).reshape(1, 1, 1, 8, 8)
    outside = allowed == 0
    for value in (
        output.alpha,
        output.premultiplied_foreground,
        output.straight_foreground,
        output.refractive_flow,
        output.coarse_refractive_flow,
        output.residual,
        output.confidence,
        output.surface_normal,
        output.depth,
        output.geometry_confidence,
    ):
        assert value is not None
        assert torch.count_nonzero(value * outside) == 0
    assert torch.equal(
        output.color_transmission * outside,
        outside.expand_as(output.color_transmission),
    )
    assert torch.equal(
        output.transmittance * outside,
        outside.expand_as(output.transmittance),
    )
    assert output.refractive_kernel_flows is not None
    assert torch.count_nonzero(output.refractive_kernel_flows * outside[:, :, None]) == 0
    recomposed, _ = recompose(
        output.alpha,
        output.premultiplied_foreground,
        background,
        output.refractive_flow,
        refractive_kernel_flows=output.refractive_kernel_flows,
        refractive_kernel_weights=output.refractive_kernel_weights,
        transmittance=output.transmittance,
        residual=output.residual,
    )
    assert torch.allclose(recomposed * outside, background * outside, atol=1e-6)


def test_hard_object_support_keeps_mask_gradient_during_training() -> None:
    model = PhysicsAwareMatter(
        MatterConfig(feature_channels=8, width=8, temporal_blocks=0)
    ).train()
    frames = torch.rand(1, 1, 3, 8, 8)
    trimap = torch.zeros(1, 1, 3, 8, 8)
    trimap[:, :, 1] = 10.0
    mask_logits = torch.full((1, 1, 1, 8, 8), -1.0, requires_grad=True)
    output = model(
        frames,
        torch.rand_like(frames),
        trimap,
        torch.rand(1, 1, 8, 4, 4),
        torch.zeros(1, 1, 1, 8, 8),
        object_mask_logits=mask_logits,
    )
    assert torch.count_nonzero(output.premultiplied_foreground) == 0
    output.premultiplied_foreground.sum().backward()
    assert mask_logits.grad is not None
    assert torch.isfinite(mask_logits.grad).all()
    assert torch.count_nonzero(mask_logits.grad) > 0


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


def test_rgb_transmission_matches_renderer_normalization_and_learns_color() -> None:
    config = MatterConfig(feature_channels=8, width=8, hard_support_at_inference=True)
    model = PhysicsAwareMatter(config).eval()
    assert "canonical-rgb-transmission-v1" in model.ARCHITECTURE
    assert torch.equal(model.head.bias[7:10], torch.zeros(3))
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
        model.head.bias[7:10] = torch.tensor((-0.6, -0.3, 0.0))

    frames = torch.rand(1, 1, 3, 8, 8)
    trimap = torch.zeros(1, 1, 3, 4, 4)
    trimap[:, :, 1] = 20.0
    output = model(
        frames,
        torch.rand_like(frames),
        trimap,
        torch.rand(1, 1, 8, 4, 4),
        torch.zeros(1, 1, 1, 8, 8),
    )
    expected = torch.exp(torch.tensor((-0.6, -0.3, 0.0)))
    assert torch.allclose(output.color_transmission[0, 0, :, 0, 0], expected)
    assert torch.allclose(
        output.color_transmission.amax(dim=2, keepdim=True),
        torch.ones_like(output.alpha),
    )
    assert torch.allclose(
        output.transmittance.amax(dim=2, keepdim=True), 1.0 - output.alpha
    )
    (output.color_transmission[:, :, 0] - 0.5).square().mean().backward()
    assert model.head.bias.grad is not None
    assert torch.isfinite(model.head.bias.grad[7:10]).all()
    assert model.head.bias.grad[7:10].abs().sum() > 0


def test_legacy_rgb_transmission_remains_explicitly_available() -> None:
    config = MatterConfig(
        feature_channels=8, width=8, canonical_rgb_transmission=False,
        neutral_transmission_bias=4.0,
    )
    model = PhysicsAwareMatter(config)
    assert "canonical-rgb-transmission" not in model.ARCHITECTURE
    assert torch.allclose(model.head.bias[7:10], torch.full((3,), 4.0))


def test_deformable_refractive_kernel_is_normalized_and_matches_mean_flow() -> None:
    config = MatterConfig(
        feature_channels=8,
        width=8,
        encoder_depths=(1, 1, 1),
        temporal_blocks=1,
        hard_support_at_inference=False,
        refractive_kernel_size=3,
    )
    model = PhysicsAwareMatter(config).eval()
    frames = torch.rand(2, 3, 3, 12, 16)
    trimap = torch.zeros(2, 3, 3, 3, 4)
    trimap[:, :, 1] = 10
    output = model(
        frames,
        torch.rand_like(frames),
        trimap,
        torch.rand(2, 3, 8, 3, 4),
        torch.zeros(2, 3, 1, 12, 16),
    )
    assert output.refractive_kernel_weights is not None
    assert output.refractive_kernel_flows is not None
    assert output.refractive_kernel_weights.shape == (2, 3, 9, 12, 16)
    assert output.refractive_kernel_flows.shape == (2, 3, 9, 2, 12, 16)
    assert torch.allclose(
        output.refractive_kernel_weights.sum(dim=2),
        torch.ones(2, 3, 12, 16),
        atol=1e-6,
    )
    expected = (
        output.refractive_kernel_weights[:, :, :, None]
        * output.refractive_kernel_flows
    ).sum(dim=2)
    assert torch.allclose(output.refractive_flow, expected, atol=1e-6)


def test_optional_geometry_head_is_normalized_positive_and_supported() -> None:
    config = MatterConfig(
        feature_channels=8,
        width=8,
        encoder_depths=(1, 1, 1),
        temporal_blocks=0,
        hard_support_at_inference=True,
        predict_geometry=True,
        geometry_min_depth=0.01,
    )
    model = PhysicsAwareMatter(config).eval()
    frames = torch.rand(1, 2, 3, 8, 8)
    trimap = torch.zeros(1, 2, 3, 4, 4)
    trimap[:, :, 0] = 20.0
    trimap[:, 0, 0] = 0.0
    trimap[:, 0, 1] = 20.0
    output = model(
        frames,
        torch.rand_like(frames),
        trimap,
        torch.rand(1, 2, 8, 4, 4),
        torch.zeros(1, 2, 1, 8, 8),
    )

    assert output.surface_normal is not None
    assert output.depth is not None
    assert output.geometry_confidence is not None
    normal_norm = torch.linalg.vector_norm(output.surface_normal[:, 0], dim=1)
    assert torch.allclose(normal_norm, torch.ones_like(normal_norm), atol=1e-5)
    assert output.depth[:, 0].min() >= config.geometry_min_depth
    assert torch.count_nonzero(output.surface_normal[:, 1]) == 0
    assert torch.count_nonzero(output.depth[:, 1]) == 0
    assert torch.count_nonzero(output.geometry_confidence[:, 1]) == 0
