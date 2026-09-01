import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from refractive_mam2 import (
    BackgroundConfig,
    DirectBackgroundEvidence,
    MaskedTemporalBackground,
    RefractiveBackgroundEvidence,
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
    output = model.fuse(direct, inverse)
    assert torch.allclose(output.background[..., 0], torch.full((1, 3, 1), 0.2))
    assert torch.allclose(output.background[..., 1], torch.full((1, 3, 1), 0.6))
    assert torch.allclose(output.background[..., 2], torch.full((1, 3, 1), 0.9))
    assert diffusion.calls == 1
    assert torch.equal(
        diffusion.last_hole,
        torch.tensor([[[[False, False, True]]]]),
    )
