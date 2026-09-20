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
from refractive_mam2.train import (
    _batch_metrics,
    _checkpointed_paired_predict,
    _predict,
)
from refractive_mam2.training import joint_stage_loss
from refractive_mam2.losses import _probability_binary_cross_entropy


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
        mask_logits = self.mask(feature).reshape(b, t, 1, h, w)
        trimap_logits = self.trimap(feature).reshape(b, t, 3, h, w)
        return MAM2BackboneOutput(
            mask_logits=mask_logits,
            trimap_logits=trimap_logits,
            alpha_matte=mask_logits.sigmoid(),
            non_memory_features=F.avg_pool2d(feature, 2).reshape(
                b, t, feature.shape[1], h // 2, w // 2
            ),
        )


class DummyClipPredictor(DummyBackbone):
    image_size = 16

    def forward_mam2_clip(
        self,
        frames,
        *,
        first_frame_mask_inputs=None,
        first_frame_point_inputs=None,
        compute_alpha=True,
    ):
        del first_frame_mask_inputs, first_frame_point_inputs, compute_alpha
        return self(frames)


def test_confidence_bce_stays_fp32_inside_bfloat16_autocast() -> None:
    logits = torch.randn(8, requires_grad=True)
    target = torch.rand(8)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        probability = logits.sigmoid()
        loss = _probability_binary_cross_entropy(probability, target).mean()

    reference = F.binary_cross_entropy(logits.sigmoid(), target)
    assert loss.dtype == torch.float32
    assert torch.allclose(loss, reference)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


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


def test_paired_microbatch_checkpoint_preserves_outputs_and_gradients() -> None:
    torch.manual_seed(4)
    target = RefractiveGroundTruth(
        frames=torch.rand(2, 2, 3, 16, 16),
        object_mask=torch.ones(2, 2, 1, 16, 16),
        counterfactual_background=torch.rand(2, 3, 16, 16),
    )
    config = PipelineConfig(
        background=BackgroundConfig(
            completion_backbone="dilated",
            completion_width=8,
            completion_dilations=(1,),
            exclusion_dilation=1,
        ),
        matter=MatterConfig(feature_channels=8, width=8),
        joint_refinement_steps=1,
    )
    direct_predictor = DummyClipPredictor(8)
    direct_pipeline = RefractiveMAM2(direct_predictor, config)
    checkpoint_predictor = DummyClipPredictor(8)
    checkpoint_pipeline = RefractiveMAM2(checkpoint_predictor, config)
    checkpoint_predictor.load_state_dict(direct_predictor.state_dict())
    checkpoint_pipeline.load_state_dict(direct_pipeline.state_dict())

    direct = _predict(
        direct_predictor,
        direct_pipeline,
        target,
        teacher_forcing=False,
        prompt_mode="point",
    )
    checkpointed = _checkpointed_paired_predict(
        checkpoint_predictor,
        checkpoint_pipeline,
        target,
        teacher_forcing=False,
        prompt_mode="point",
    )
    assert torch.allclose(
        checkpointed.reconstructed_frames,
        direct.reconstructed_frames,
        atol=1e-6,
    )
    direct_loss = joint_stage_loss(
        direct,
        target,
        paired_background_group_ids=["pair", "pair"],
        operator_support=target.object_mask,
    )["total"]
    checkpointed_loss = joint_stage_loss(
        checkpointed,
        target,
        paired_background_group_ids=["pair", "pair"],
        operator_support=target.object_mask,
    )["total"]
    assert torch.allclose(checkpointed_loss, direct_loss, atol=1e-6)
    direct_loss.backward()
    checkpointed_loss.backward()
    assert torch.allclose(
        checkpoint_pipeline.matter.head.weight.grad,
        direct_pipeline.matter.head.weight.grad,
        atol=1e-5,
    )


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


def test_physics_alpha_is_a_bounded_unknown_region_refinement() -> None:
    frames = torch.rand(1, 2, 3, 16, 16)
    config = PipelineConfig(
        matter=MatterConfig(
            feature_channels=8,
            width=8,
            hard_support_at_inference=False,
            alpha_refinement_scale=0.1,
        )
    )
    output = RefractiveMAM2(DummyBackbone(8), config)(frames)
    delta = (output.matter.alpha - output.backbone.alpha_matte).abs()
    assert float(delta.detach().max()) <= 0.100001


def test_forward_from_precomputed_official_sam2_outputs() -> None:
    frames = torch.rand(1, 3, 3, 24, 32)
    features = torch.rand(1, 3, 8, 6, 8)
    precomputed = MAM2BackboneOutput(
        mask_logits=torch.rand(1, 3, 1, 6, 8),
        trimap_logits=torch.rand(1, 3, 3, 6, 8),
        alpha_matte=torch.rand(1, 3, 1, 24, 32),
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


def test_ffc_completion_runs_and_receives_gradients_at_every_iteration() -> None:
    frames = torch.rand(1, 2, 3, 16, 16)
    config = PipelineConfig(
        background=BackgroundConfig(
            completion_backbone="ffc",
            completion_width=8,
            completion_down_blocks=2,
            completion_residual_blocks=1,
            completion_max_channels=32,
            exclusion_dilation=1,
        ),
        matter=MatterConfig(feature_channels=8, width=16),
        joint_refinement_steps=2,
    )
    model = RefractiveMAM2(DummyBackbone(8), config)
    calls = []
    handle = model.background_model.completion.register_forward_hook(
        lambda _module, _inputs, _output: calls.append(1)
    )
    output = model(frames)
    handle.remove()
    # One direct-evidence initialization followed by every unrolled update.
    assert len(calls) == config.joint_refinement_steps + 1
    output.background.background.mean().backward()
    assert any(
        parameter.grad is not None
        for parameter in model.background_model.completion.parameters()
        if parameter.requires_grad
    )


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
