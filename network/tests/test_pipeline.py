import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")
F = pytest.importorskip("torch.nn.functional")

from refractive_mam2 import (
    BackgroundConfig,
    MAM2BackboneOutput,
    MaskedTemporalBackground,
    MatterConfig,
    PipelineConfig,
    RefractiveGroundTruth,
    RefractiveLoss,
    RefractiveMAM2,
)
from refractive_mam2.train import _batch_metrics


class StubDiffusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, background, coverage, true_hole):
        del coverage, true_hole
        self.calls += 1
        return torch.full_like(background, 0.5)


class DummyBackbone(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Conv2d(3, channels, 3, padding=1)
        self.mask = nn.Conv2d(channels, 1, 1)
        self.trimap = nn.Conv2d(channels, 3, 1)

    def forward(self, frames, prompts=None):
        del prompts
        b, t, _, h, w = frames.shape
        feature = self.body(frames.flatten(0, 1))
        return MAM2BackboneOutput(
            mask_logits=self.mask(feature).reshape(b, t, 1, h, w),
            trimap_logits=self.trimap(feature).reshape(b, t, 3, h, w),
            non_memory_features=F.avg_pool2d(feature, 2).reshape(
                b, t, feature.shape[1], h // 2, w // 2
            ),
        )


def test_pipeline_shapes_and_gradient() -> None:
    frames = torch.rand(1, 2, 3, 32, 40)
    config = PipelineConfig(matter=MatterConfig(feature_channels=16, width=16))
    model = RefractiveMAM2(DummyBackbone(16), config)
    output = model(frames)

    assert output.matter.alpha.shape == (1, 2, 1, 32, 40)
    assert output.matter.premultiplied_foreground.shape == frames.shape
    assert output.matter.transmittance.shape == frames.shape
    assert output.matter.residual.shape == frames.shape
    assert output.matter.refractive_flow.shape == (1, 2, 2, 32, 40)
    assert output.background.background.shape == (1, 3, 32, 40)
    assert output.background.video(frames.shape[1]).shape == frames.shape
    assert output.reconstructed_frames.shape == frames.shape

    loss = RefractiveLoss()(output, RefractiveGroundTruth(frames=frames))["total"]
    loss.backward()
    assert model.matter.head.weight.grad is not None


def test_teacher_forced_background() -> None:
    frames = torch.rand(1, 2, 3, 24, 24)
    background_gt = torch.rand(1, 3, 24, 24)
    config = PipelineConfig(matter=MatterConfig(feature_channels=8, width=16))
    model = RefractiveMAM2(DummyBackbone(8), config)
    output = model(
        frames,
        counterfactual_background_gt=background_gt,
        use_ground_truth_background=True,
    )
    assert output.reconstructed_frames.shape == frames.shape


def test_forward_from_precomputed_official_sam2_outputs() -> None:
    frames = torch.rand(1, 3, 3, 24, 32)
    features = torch.rand(1, 3, 8, 6, 8)
    precomputed = MAM2BackboneOutput(
        mask_logits=torch.rand(1, 3, 1, 6, 8),
        trimap_logits=torch.rand(1, 3, 3, 6, 8),
        non_memory_features=features,
    )
    config = PipelineConfig(matter=MatterConfig(feature_channels=8, width=16))
    model = RefractiveMAM2(DummyBackbone(8), config)
    output = model.forward_from_backbone(frames, precomputed)
    assert output.backbone is precomputed
    assert output.reconstructed_frames.shape == frames.shape


def test_diffusion_completion_runs_once_after_fixed_point_refinement() -> None:
    frames = torch.rand(1, 2, 3, 16, 16)
    diffusion = StubDiffusion()
    config = PipelineConfig(
        background=BackgroundConfig(
            completion_variant="diffusion",
            exclusion_dilation=1,
        ),
        matter=MatterConfig(feature_channels=8, width=16),
        joint_refinement_steps=3,
    )
    background = MaskedTemporalBackground(
        config.background,
        diffusion_completion=diffusion,
    )
    model = RefractiveMAM2(
        DummyBackbone(8),
        config,
        background_model=background,
    )
    output = model(frames)
    assert output.background.background.shape == (1, 3, 16, 16)
    assert diffusion.calls == 1


def test_zero_step_ablation_runs_without_inverse_refinement() -> None:
    frames = torch.rand(1, 2, 3, 16, 16)
    config = PipelineConfig(
        matter=MatterConfig(feature_channels=8, width=16),
        joint_refinement_steps=0,
    )
    output = RefractiveMAM2(DummyBackbone(8), config)(frames)
    assert output.reconstructed_frames.shape == frames.shape
    assert torch.all(output.background.inverse_coverage == 0)


def test_paper_metrics_include_region_and_boundary_breakdowns() -> None:
    pytest.importorskip("cv2")
    frames = torch.rand(1, 2, 3, 16, 16)
    background_gt = torch.rand(1, 3, 16, 16)
    config = PipelineConfig(matter=MatterConfig(feature_channels=8, width=16))
    output = RefractiveMAM2(DummyBackbone(8), config)(frames)
    metrics = _batch_metrics(
        output,
        RefractiveGroundTruth(
            frames=frames,
            alpha=torch.rand_like(output.matter.alpha),
            counterfactual_background=background_gt,
            refractive_flow=torch.zeros_like(output.matter.refractive_flow),
            refractive_validity=torch.ones_like(output.matter.alpha),
        ),
    )
    assert "background_ssim" in metrics
    assert "background_true_hole_fraction" in metrics
    assert "evidence_preservation_l1" in metrics
    assert "alpha_connectivity_error" in metrics
    assert "alpha_boundary_f1" in metrics
    assert "flow_bad_pixel" in metrics
