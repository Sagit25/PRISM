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

    def forward(self, background, coverage, true_hole):
        del coverage
        self.calls += 1
        self.last_hole = true_hole.clone()
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


def test_evidence_flux_refines_non_direct_pixels_after_filling_true_holes() -> None:
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
    assert torch.equal(
        model.completion.last_mask,
        torch.tensor([[[[False, True, True]]]]),
    )
    assert diffusion.calls == 1
    assert torch.equal(diffusion.last_hole, torch.tensor([[[[False, False, True]]]]))
    assert torch.allclose(
        model.completion.last_background[..., 2], torch.full((1, 3, 1), 0.9)
    )
    assert torch.allclose(final.background[..., 0], torch.full((1, 3, 1), 0.2))
    assert torch.allclose(final.background[..., 1], torch.full((1, 3, 1), 0.4))
    assert torch.allclose(final.background[..., 2], torch.full((1, 3, 1), 0.4))


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


def test_flux_hole_supervision_updates_only_trainable_completion() -> None:
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
    output = model.fuse(direct)
    output.background.square().mean().backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.completion.parameters()
    )
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
