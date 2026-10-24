import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from refractive_mam2 import (
    BackgroundConfig,
    DirectBackgroundEvidence,
    MaskedTemporalBackground,
    RefractiveBackgroundEvidence,
    GLaMaCompletionNet,
)


class StubDiffusion(nn.Module):
    def __init__(self, value: float = 0.9) -> None:
        super().__init__()
        self.value = value
        self.calls = 0
        self.last_hole = None
        self.last_background = None
        self.last_coverage = None

    def forward(self, background, coverage, true_hole):
        self.calls += 1
        self.last_hole = true_hole.clone()
        self.last_background = background.clone()
        self.last_coverage = coverage.clone()
        return torch.full_like(background, self.value)


class StubCompletion(nn.Module):
    def __init__(self, value: float = 0.4) -> None:
        super().__init__()
        self.value = value
        self.last_mask = None
        self.last_background = None

    def forward(self, background, coverage, editable_mask):
        del coverage
        self.last_mask = editable_mask.clone()
        self.last_background = background.clone()
        return torch.full_like(background, self.value)


def test_observed_background_is_not_blended_with_completion() -> None:
    frames = torch.rand(1, 1, 3, 8, 10)
    exclusion = torch.zeros(1, 1, 1, 8, 10)
    model = MaskedTemporalBackground(
        BackgroundConfig(exclusion_dilation=1, use_temporal_completion=True)
    )
    output = model(frames, exclusion)
    assert output.background.shape == (1, 3, 8, 10)
    assert torch.allclose(output.background, frames[:, 0], atol=1e-6)
    assert torch.allclose(output.observed_background, frames[:, 0], atol=1e-6)


def test_disagreeing_inverse_samples_have_lower_reliable_coverage() -> None:
    model = MaskedTemporalBackground(
        BackgroundConfig(
            completion_backbone="dilated", completion_width=8,
            completion_dilations=(1,), use_temporal_completion=False,
            inverse_variance_scale=12.0,
        )
    )
    direct = DirectBackgroundEvidence(
        observed_background=torch.zeros(1, 3, 1, 1),
        weight=torch.zeros(1, 1, 1, 1),
        coverage=torch.zeros(1, 1, 1, 1),
        uncertainty=torch.ones(1, 1, 1, 1),
    )
    consistent = RefractiveBackgroundEvidence(
        background=torch.full((1, 3, 1, 1), 0.5),
        weight=torch.ones(1, 1, 1, 1),
        variance=torch.zeros(1, 1, 1, 1),
    )
    conflicting = RefractiveBackgroundEvidence(
        background=consistent.background,
        weight=consistent.weight,
        variance=torch.full((1, 1, 1, 1), 0.09),
    )
    assert model.fuse(direct, conflicting).inverse_coverage.item() < model.fuse(
        direct, consistent
    ).inverse_coverage.item()


def test_unobserved_area_reports_full_uncertainty() -> None:
    frames = torch.rand(1, 1, 3, 6, 6)
    exclusion = torch.ones(1, 1, 1, 6, 6)
    model = MaskedTemporalBackground(BackgroundConfig(exclusion_dilation=1))
    output = model(frames, exclusion)
    assert torch.all(output.uncertainty == 1)
    assert torch.all(output.coverage == 0)


def test_default_exclusion_does_not_expand_a_single_64px_object_pixel() -> None:
    frames = torch.ones(1, 1, 3, 64, 64)
    exclusion = torch.zeros(1, 1, 1, 64, 64)
    exclusion[..., 32, 32] = 1
    model = MaskedTemporalBackground(BackgroundConfig())

    evidence = model.observe(frames, exclusion)

    assert model.config.exclusion_dilation == 1
    assert evidence.coverage[0, 0, 32, 32] == 0
    assert evidence.coverage[0, 0, 32, 31] == 1
    assert evidence.coverage[0, 0, 31, 32] == 1


def test_flux_only_does_not_preserve_ambiguous_object_pixel_as_direct_background() -> None:
    frames = torch.full((1, 2, 3, 8, 8), 0.2)
    frames[..., 4, 4] = 0.8
    exclusion = torch.zeros(1, 2, 1, 8, 8)
    exclusion[..., 4, 4] = 0.3
    diffusion = StubDiffusion(value=0.4)
    model = MaskedTemporalBackground(
        BackgroundConfig(
            exclusion_dilation=1,
            completion_variant="diffusion",
            evidence_flux_only=True,
            flux_direct_observation_threshold=0.1,
        ),
        diffusion_completion=diffusion,
    )
    output = model(frames, exclusion)
    assert not output.direct_support[0, 0, 4, 4]
    assert torch.allclose(output.background[0, :, 4, 4], torch.full((3,), 0.4))
    assert output.direct_support[0, 0, 0, 0]
    assert torch.allclose(output.background[0, :, 0, 0], torch.full((3,), 0.2))


def test_temporal_direct_evidence_rejects_clean_object_tie() -> None:
    frames = torch.full((1, 2, 3, 4, 4), 0.2)
    frames[:, 1, :, 2, 2] = 0.8  # missed by the semantic mask
    exclusion = torch.zeros(1, 2, 1, 4, 4)
    diffusion = StubDiffusion(value=0.45)
    model = MaskedTemporalBackground(
        BackgroundConfig(
            completion_variant="diffusion",
            evidence_flux_only=True,
            exclusion_dilation=1,
        ),
        diffusion_completion=diffusion,
    )

    output = model(frames, exclusion)

    assert not output.direct_support[0, 0, 2, 2]
    assert output.direct_coverage[0, 0, 2, 2] == 0
    assert torch.allclose(output.background[0, :, 2, 2], torch.full((3,), 0.45))
    assert output.direct_support[0, 0, 0, 0]
    assert torch.allclose(output.background[0, :, 0, 0], torch.full((3,), 0.2))


def test_temporal_direct_evidence_keeps_majority_clean_observations() -> None:
    frames = torch.full((1, 3, 3, 4, 4), 0.2)
    frames[:, 2, :, 2, 2] = 0.8
    model = MaskedTemporalBackground(BackgroundConfig(exclusion_dilation=1))

    evidence = model.observe(frames, torch.zeros(1, 3, 1, 4, 4))

    assert evidence.weight[0, 0, 2, 2] > 0
    assert torch.allclose(
        evidence.observed_background[0, :, 2, 2],
        torch.full((3,), 0.2),
        atol=1e-6,
    )


def test_flux_does_not_freeze_majority_missed_object_color_as_background() -> None:
    frames = torch.full((1, 4, 3, 4, 4), 0.2)
    frames[:, :3, :, 2, 2] = 0.8  # object falsely marked visible in 3/4 frames
    diffusion = StubDiffusion(value=0.45)
    model = MaskedTemporalBackground(
        BackgroundConfig(
            completion_variant="diffusion",
            evidence_flux_only=True,
            exclusion_dilation=1,
        ),
        diffusion_completion=diffusion,
    )

    output = model(frames, torch.zeros(1, 4, 1, 4, 4))

    assert not output.direct_support[0, 0, 2, 2]
    assert diffusion.last_hole[0, 0, 2, 2]
    assert torch.allclose(output.background[0, :, 2, 2], torch.full((3,), 0.45))
    assert output.direct_support[0, 0, 0, 0]
    assert torch.allclose(output.background[0, :, 0, 0], torch.full((3,), 0.2))


def test_flux_requires_two_independent_direct_background_observations() -> None:
    frames = torch.full((1, 4, 3, 4, 4), 0.2)
    frames[:, 0, :, 2, 2] = 0.8  # the sole candidate is a missed object pixel
    exclusion = torch.zeros(1, 4, 1, 4, 4)
    exclusion[:, 1:, :, 2, 2] = 1.0
    diffusion = StubDiffusion(value=0.45)
    model = MaskedTemporalBackground(
        BackgroundConfig(
            completion_variant="diffusion",
            evidence_flux_only=True,
            exclusion_dilation=1,
        ),
        diffusion_completion=diffusion,
    )

    output = model(frames, exclusion)

    assert not output.direct_support[0, 0, 2, 2]
    assert diffusion.last_hole[0, 0, 2, 2]
    assert torch.allclose(output.background[0, :, 2, 2], torch.full((3,), 0.45))
    assert output.direct_support[0, 0, 0, 0]
    assert torch.allclose(output.background[0, :, 0, 0], torch.full((3,), 0.2))


def test_all_frames_produce_one_shared_background_asset() -> None:
    base = torch.rand(1, 3, 6, 7)
    frames = base[:, None].expand(-1, 4, -1, -1, -1).clone()
    exclusion = torch.zeros(1, 4, 1, 6, 7)
    output = MaskedTemporalBackground(
        BackgroundConfig(exclusion_dilation=1)
    )(frames, exclusion)
    assert output.background.shape == base.shape
    assert torch.allclose(output.background, base, atol=1e-6)
    assert output.video(4).shape == frames.shape


def test_glama_completion_supports_odd_sizes_and_backpropagation() -> None:
    model = GLaMaCompletionNet(
        width=8,
        down_blocks=2,
        residual_blocks=2,
        max_channels=32,
    )
    evidence = torch.rand(1, 3, 13, 17, requires_grad=True)
    coverage = torch.rand(1, 1, 13, 17)
    true_hole = coverage < 0.25
    output = model(evidence, coverage, true_hole)
    assert output.shape == evidence.shape
    assert torch.isfinite(output).all()
    output.mean().backward()
    assert evidence.grad is not None
    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_default_background_uses_glama_style_ffc_completion() -> None:
    model = MaskedTemporalBackground(BackgroundConfig(exclusion_dilation=1))
    assert isinstance(model.completion, GLaMaCompletionNet)


def test_diffusion_can_modify_only_true_holes() -> None:
    direct_background = torch.zeros(1, 3, 1, 3)
    direct_background[..., 0] = 0.2
    direct_weight = torch.zeros(1, 1, 1, 3)
    direct_weight[..., 0] = 1
    direct = DirectBackgroundEvidence(
        observed_background=direct_background,
        weight=direct_weight,
        coverage=direct_weight.clone(),
        uncertainty=1.0 - direct_weight,
    )
    inverse_background = torch.zeros(1, 3, 1, 3)
    inverse_background[..., 1] = 0.6
    inverse_weight = torch.zeros(1, 1, 1, 3)
    inverse_weight[..., 1] = 1
    inverse = RefractiveBackgroundEvidence(
        background=inverse_background,
        weight=inverse_weight,
    )
    diffusion = StubDiffusion()
    model = MaskedTemporalBackground(
        BackgroundConfig(completion_variant="diffusion", exclusion_dilation=1),
        diffusion_completion=diffusion,
    )
    model.completion = StubCompletion()
    output = model.fuse(direct, inverse)
    assert torch.allclose(output.background[..., 0], torch.full((1, 3, 1), 0.2))
    assert torch.allclose(output.background[..., 1], torch.full((1, 3, 1), 0.4))
    assert torch.allclose(output.background[..., 2], torch.full((1, 3, 1), 0.9))
    assert diffusion.calls == 1
    assert torch.equal(
        model.completion.last_mask,
        torch.tensor([[[[False, True, True]]]]),
    )
    assert torch.equal(
        diffusion.last_hole,
        torch.tensor([[[[False, False, True]]]]),
    )


def test_evidence_flux_replaces_inverse_only_pixels_but_preserves_direct() -> None:
    direct_background = torch.zeros(1, 3, 1, 3)
    direct_background[..., 0] = 0.2
    direct_weight = torch.zeros(1, 1, 1, 3)
    direct_weight[..., 0] = 1
    direct = DirectBackgroundEvidence(
        observed_background=direct_background,
        weight=direct_weight,
        coverage=direct_weight.clone(),
        uncertainty=1.0 - direct_weight,
    )
    inverse_background = torch.zeros_like(direct_background)
    inverse_background[..., 1] = 0.6
    inverse_weight = torch.zeros_like(direct_weight)
    inverse_weight[..., 1] = 1
    inverse = RefractiveBackgroundEvidence(
        background=inverse_background,
        weight=inverse_weight,
    )
    diffusion = StubDiffusion()
    model = MaskedTemporalBackground(
        BackgroundConfig(
            completion_variant="diffusion",
            evidence_flux_only=True,
            exclusion_dilation=1,
        ),
        diffusion_completion=diffusion,
    )
    model.completion = StubCompletion()
    intermediate = model.fuse(direct, inverse, completion_variant="base")
    assert model.completion.last_mask is None
    assert diffusion.calls == 0
    assert torch.allclose(intermediate.background[..., 0], torch.full((1, 3, 1), 0.2))
    assert torch.allclose(intermediate.background[..., 1], torch.full((1, 3, 1), 0.6))
    assert torch.all(intermediate.background[..., 2] == 0)
    final = model.fuse(direct, inverse, completion_variant="diffusion")
    assert model.completion.last_mask is None
    assert diffusion.calls == 1
    assert torch.equal(diffusion.last_hole, torch.tensor([[[[False, True, True]]]]))
    assert torch.all(diffusion.last_background[..., 1:] == 0)
    assert torch.all(diffusion.last_coverage[..., 1:] == 0)
    assert torch.allclose(final.background[..., 0], torch.full((1, 3, 1), 0.2))
    # The flow-guided inverse contributes a bounded update, but cannot become
    # an immutable object-colored patch or override a direct observation.
    assert torch.all(final.background[..., 1] < 0.9)
    assert torch.all(final.background[..., 1] > 0.85)
    assert torch.allclose(final.background[..., 2], torch.full((1, 3, 1), 0.9))
    assert torch.equal(final.uncertainty[..., 1:], torch.ones(1, 1, 1, 2))


def test_flux_inverse_feedback_is_differentiable_and_rejects_large_disagreement() -> None:
    direct_weight = torch.tensor([[[[1.0, 0.0, 0.0]]]])
    direct = DirectBackgroundEvidence(
        observed_background=torch.full((1, 3, 1, 3), 0.2),
        weight=direct_weight,
        coverage=direct_weight,
        uncertainty=1.0 - direct_weight,
    )
    inverse_color = torch.tensor([[[[0.2, 0.8, 0.1]]]]).expand(1, 3, 1, 3).clone()
    inverse_color.requires_grad_()
    inverse = RefractiveBackgroundEvidence(
        background=inverse_color,
        weight=torch.tensor([[[[1.0, 1.0, 1.0]]]]),
    )
    model = MaskedTemporalBackground(
        BackgroundConfig(
            evidence_flux_only=True,
            completion_variant="diffusion",
            diffusion_inverse_feedback_agreement_scale=8.0,
        ),
        diffusion_completion=StubDiffusion(value=0.9),
    )

    background = model.fuse(direct, inverse).background
    assert torch.allclose(background[..., 0], torch.full((1, 3, 1), 0.2))
    # A matching direct/inverse overlap gives a small correction even when
    # FLUX disagrees strongly; it must not replace the FLUX result wholesale.
    assert torch.all(background[..., 2] > 0.88)
    assert torch.all(background[..., 2] < 0.9)
    assert background[..., 1].sub(0.9).abs().max() <= 0.05
    background[..., 1].sum().backward()
    assert inverse_color.grad is not None
    assert inverse_color.grad[..., 1].abs().sum() > 0
    assert inverse_color.grad[..., 0].abs().sum() == 0


def test_direct_inverse_overlap_calibrates_flow_guided_flux_feedback() -> None:
    direct_weight = torch.tensor([[[[1.0, 0.0, 0.0]]]])
    direct = DirectBackgroundEvidence(
        observed_background=torch.full((1, 3, 1, 3), 0.5),
        weight=direct_weight,
        coverage=direct_weight,
        uncertainty=1.0 - direct_weight,
    )
    matching_inverse = RefractiveBackgroundEvidence(
        background=torch.full((1, 3, 1, 3), 0.5),
        weight=torch.ones_like(direct_weight),
    )
    conflicting_inverse = RefractiveBackgroundEvidence(
        background=torch.full((1, 3, 1, 3), 0.9),
        weight=torch.ones_like(direct_weight),
    )
    diffusion = StubDiffusion(value=0.2)
    calibrated = MaskedTemporalBackground(
        BackgroundConfig(evidence_flux_only=True, completion_variant="diffusion"),
        diffusion_completion=diffusion,
    )
    conservative = MaskedTemporalBackground(
        BackgroundConfig(
            evidence_flux_only=True,
            completion_variant="diffusion",
            diffusion_inverse_overlap_min_support=1.0,
        ),
        diffusion_completion=diffusion,
    )

    calibrated_result = calibrated.fuse(direct, matching_inverse).background
    conservative_result = conservative.fuse(direct, matching_inverse).background
    contaminated_result = calibrated.fuse(direct, conflicting_inverse).background

    assert torch.allclose(calibrated_result[..., 0], torch.full((1, 3, 1), 0.5))
    assert torch.all(calibrated_result[..., 1:] > conservative_result[..., 1:])
    assert torch.all(calibrated_result[..., 1:] < 0.5)
    assert torch.all(contaminated_result[..., 1:] < calibrated_result[..., 1:])
    assert torch.all(contaminated_result[..., 1:] < 0.3)


def test_flux_inverse_refinement_can_be_disabled_for_ablation() -> None:
    direct = DirectBackgroundEvidence(
        observed_background=torch.zeros(1, 3, 1, 2),
        weight=torch.zeros(1, 1, 1, 2),
        coverage=torch.zeros(1, 1, 1, 2),
        uncertainty=torch.ones(1, 1, 1, 2),
    )
    inverse = RefractiveBackgroundEvidence(
        background=torch.full((1, 3, 1, 2), 0.6),
        weight=torch.tensor([[[[1.0, 0.0]]]]),
    )
    diffusion = StubDiffusion()
    model = MaskedTemporalBackground(
        BackgroundConfig(
            completion_variant="diffusion",
            evidence_flux_only=True,
            refine_inverse_evidence=False,
        ),
        diffusion_completion=diffusion,
    )

    output = model.fuse(direct, inverse)

    assert torch.equal(diffusion.last_hole, torch.tensor([[[[False, True]]]]))
    assert torch.allclose(output.background[..., 0], torch.full((1, 3, 1), 0.6))
    assert torch.allclose(output.background[..., 1], torch.full((1, 3, 1), 0.9))


def test_weak_inverse_splat_is_treated_as_a_true_hole() -> None:
    direct = DirectBackgroundEvidence(
        observed_background=torch.zeros(1, 3, 1, 1),
        weight=torch.zeros(1, 1, 1, 1),
        coverage=torch.zeros(1, 1, 1, 1),
        uncertainty=torch.ones(1, 1, 1, 1),
    )
    inverse = RefractiveBackgroundEvidence(
        background=torch.ones(1, 3, 1, 1),
        weight=torch.full((1, 1, 1, 1), 0.01),
    )
    model = MaskedTemporalBackground(
        BackgroundConfig(completion_variant="diffusion", evidence_flux_only=True),
        diffusion_completion=StubDiffusion(),
    )
    model.completion = StubCompletion()
    output = model.fuse(direct, inverse)
    assert output.true_hole.item()
    assert not output.inverse_support.item()
    assert model.diffusion_completion.calls == 1


def test_flux_only_hole_does_not_invoke_trainable_completion() -> None:
    direct = DirectBackgroundEvidence(
        observed_background=torch.zeros(1, 3, 8, 8),
        weight=torch.zeros(1, 1, 8, 8),
        coverage=torch.zeros(1, 1, 8, 8),
        uncertainty=torch.ones(1, 1, 8, 8),
    )
    model = MaskedTemporalBackground(
        BackgroundConfig(
            completion_backbone="dilated",
            completion_width=8,
            completion_variant="diffusion",
            evidence_flux_only=True,
        ),
        diffusion_completion=StubDiffusion(),
    )
    calls = []
    handle = model.completion.register_forward_hook(
        lambda _module, _inputs, _output: calls.append(1)
    )
    output = model.fuse(direct)
    handle.remove()
    assert calls == []
    assert torch.allclose(output.background, torch.full_like(output.background, 0.9))
    assert not list(model.diffusion_completion.parameters())


def test_inverse_evidence_is_a_correctable_hint_not_immutable_background() -> None:
    direct_background = torch.zeros(1, 3, 1, 3)
    direct_background[..., 0] = 0.2
    direct_weight = torch.zeros(1, 1, 1, 3)
    direct_weight[..., 0] = 1
    direct = DirectBackgroundEvidence(
        observed_background=direct_background,
        weight=direct_weight,
        coverage=direct_weight.clone(),
        uncertainty=1.0 - direct_weight,
    )
    inverse_background = torch.zeros(1, 3, 1, 3)
    inverse_background[..., 1] = 0.8  # Contaminated object color.
    inverse_weight = torch.zeros(1, 1, 1, 3)
    inverse_weight[..., 1] = 1
    model = MaskedTemporalBackground(BackgroundConfig(exclusion_dilation=1))
    model.completion = StubCompletion(0.4)

    output = model.fuse(
        direct,
        RefractiveBackgroundEvidence(inverse_background, inverse_weight),
    )

    assert torch.allclose(output.background[..., 0], torch.full((1, 3, 1), 0.2))
    assert torch.allclose(output.evidence_background[..., 1], torch.full((1, 3, 1), 0.8))
    assert torch.allclose(output.background[..., 1], torch.full((1, 3, 1), 0.4))
    assert torch.allclose(output.background[..., 2], torch.full((1, 3, 1), 0.4))
    assert torch.equal(
        model.completion.last_mask,
        torch.tensor([[[[False, True, True]]]]),
    )
    assert torch.equal(
        output.true_hole,
        torch.tensor([[[[False, False, True]]]]),
    )


def test_inverse_refinement_can_be_disabled_for_ablation() -> None:
    direct = DirectBackgroundEvidence(
        observed_background=torch.zeros(1, 3, 1, 1),
        weight=torch.zeros(1, 1, 1, 1),
        coverage=torch.zeros(1, 1, 1, 1),
        uncertainty=torch.ones(1, 1, 1, 1),
    )
    inverse = RefractiveBackgroundEvidence(
        background=torch.full((1, 3, 1, 1), 0.8),
        weight=torch.ones(1, 1, 1, 1),
    )
    model = MaskedTemporalBackground(
        BackgroundConfig(exclusion_dilation=1, refine_inverse_evidence=False)
    )
    model.completion = StubCompletion(0.4)

    output = model.fuse(direct, inverse)

    assert torch.allclose(output.background, inverse.background)
    assert torch.equal(model.completion.last_mask, output.true_hole)
