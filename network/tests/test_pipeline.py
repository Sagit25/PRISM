import pytest
from types import SimpleNamespace

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
    _loss_terms,
    _loss_weights_for_profile,
    _merge_oracle_component_metrics,
    _operator_head_gradient_metrics,
    _predict,
    evaluate,
)
from refractive_mam2.dataset import RCTransBatch
from refractive_mam2.training import joint_stage_loss
from refractive_mam2.losses import (
    _edge_aware_flow_smoothness,
    _far_background_support,
    _identifiable_color_transmission,
    _leave_one_out_background,
    _masked_gradient_loss,
    _masked_laplacian_loss,
    _straight_foreground_reliability,
    _past_visible_refracted_background,
    _probability_binary_cross_entropy,
    _temporal_loss,
)
from refractive_mam2.renderer import warp_background
from refractive_mam2.matter import PhysicsMatterOutput
from refractive_mam2.training import configure_stage3


def test_foreground_recovery_loss_trains_support_and_f_with_frozen_tracker() -> None:
    frames = torch.rand(1, 1, 3, 16, 16)
    truth_mask = torch.zeros(1, 1, 1, 16, 16)
    truth_mask[..., 4:12, 4:12] = 1.0
    predictor = DummyClipPredictor(8)
    with torch.no_grad():
        predictor.mask.weight.zero_()
        predictor.mask.bias.fill_(-12.0)
    predictor.requires_grad_(False)
    pipeline = RefractiveMAM2(
        predictor,
        PipelineConfig(matter=MatterConfig(
            feature_channels=8, width=16, max_channels=64,
            learned_object_support=True,
        )),
    )
    prediction = pipeline.forward_from_backbone(
        frames, predictor(frames),
        counterfactual_background_gt=torch.rand(1, 3, 16, 16),
        use_ground_truth_background=True,
        skip_background_estimation=True,
    )
    target = RefractiveGroundTruth(
        frames=frames,
        object_mask=truth_mask,
        alpha=truth_mask * 0.6,
        straight_foreground=truth_mask * 0.3,
        premultiplied_foreground=truth_mask * 0.18,
    )
    losses = RefractiveLoss(_loss_weights_for_profile("foreground_recovery"))(
        prediction, target
    )
    assert "matter_object_support" in losses
    assert "foreground_candidate" in losses
    losses["total"].backward()
    assert pipeline.matter.object_support_refiner is not None
    support_grad = pipeline.matter.object_support_refiner.head.weight.grad
    assert support_grad is not None and support_grad.abs().sum() > 0
    assert pipeline.matter.head.weight.grad is not None
    assert pipeline.matter.head.weight.grad[1:4].abs().sum() > 0
    assert predictor.mask.weight.grad is None


def test_observed_pixels_outside_object_are_exactly_preserved_in_render() -> None:
    frames = torch.rand(1, 1, 3, 16, 16)
    predictor = DummyClipPredictor(8)
    model = RefractiveMAM2(
        predictor,
        PipelineConfig(matter=MatterConfig(feature_channels=8, width=8)),
    ).eval()
    semantics = predictor(frames)
    semantics.mask_logits.fill_(-20.0)
    semantics.alpha_matte.zero_()
    output = model.forward_from_backbone(
        frames, semantics,
        counterfactual_background_gt=torch.zeros(1, 3, 16, 16),
        use_ground_truth_background=True,
        skip_background_estimation=True,
    )
    torch.testing.assert_close(output.reconstructed_frames, frames)


def test_detail_recovery_supervises_ungated_foreground_edges() -> None:
    weights = _loss_weights_for_profile("detail_recovery")
    assert weights.foreground_candidate_gradient > 0
    assert weights.foreground_candidate_laplacian > 0
    assert weights.matter_object_support_recall > 0
    assert weights.background_true_hole_laplacian > 0


def test_flow_recovery_adds_direct_refinement_gain_supervision() -> None:
    frames = torch.rand(1, 1, 3, 16, 16)
    predictor = DummyClipPredictor(8)
    pipeline = RefractiveMAM2(
        predictor,
        PipelineConfig(matter=MatterConfig(
            feature_channels=8, width=16, max_channels=64,
            refine_refractive_flow=True,
        )),
    )
    prediction = pipeline.forward_from_backbone(
        frames, predictor(frames),
        counterfactual_background_gt=torch.rand(1, 3, 16, 16),
        use_ground_truth_background=True,
        skip_background_estimation=True,
    )
    target = RefractiveGroundTruth(
        frames=frames,
        object_mask=torch.ones(1, 1, 1, 16, 16),
        refractive_flow=torch.ones(1, 1, 2, 16, 16),
        refractive_validity=torch.ones(1, 1, 1, 16, 16),
    )
    losses = RefractiveLoss(_loss_weights_for_profile("flow_recovery"))(
        prediction, target,
    )
    assert "flow_refinement_gain" in losses
    assert torch.isfinite(losses["flow_refinement_gain"])
    assert losses["flow_refinement_gain"] > 0


def test_evidence_flux_stage_trains_pixel_refiner_with_frozen_prior() -> None:
    class FakeFrozenPrior(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.flow_latent_adapter = nn.Conv2d(1, 1, 1)
            self.pixel_refiner = nn.Conv2d(1, 1, 1)
            self.frozen_prior = nn.Conv2d(1, 1, 1)

    predictor = DummyClipPredictor(8)
    prior = FakeFrozenPrior()
    config = PipelineConfig(
        background=BackgroundConfig(
            completion_variant="diffusion", evidence_flux_only=True,
        ),
        matter=MatterConfig(feature_channels=8, width=16),
    )
    pipeline = RefractiveMAM2(
        predictor, config,
        background_model=MaskedTemporalBackground(config.background, prior),
    )
    trainable = configure_stage3(predictor, pipeline)
    assert prior.pixel_refiner.weight.requires_grad
    assert any(parameter is prior.pixel_refiner.weight for parameter in trainable)
    assert not prior.frozen_prior.weight.requires_grad


def test_oracle_background_reinference_diagnoses_foreground_without_second_backbone() -> None:
    frames = torch.rand(1, 2, 3, 16, 16)
    target = RefractiveGroundTruth(
        frames=frames,
        counterfactual_background=torch.rand(1, 3, 16, 16),
        object_mask=torch.ones(1, 2, 1, 16, 16),
        alpha=torch.full((1, 2, 1, 16, 16), 0.5),
        straight_foreground=torch.rand(1, 2, 3, 16, 16),
    )
    batch = RCTransBatch(
        ground_truth=target,
        sequence_ids=["oracle_foreground_test"],
        paired_background_group_ids=["unique"],
        metadata=[{}],
    )
    pipeline = RefractiveMAM2(
        DummyClipPredictor(8),
        PipelineConfig(matter=MatterConfig(feature_channels=8, width=16)),
    )
    metrics = evaluate(
        pipeline.backbone,
        pipeline,
        [batch],
        torch.device("cpu"),
        stage="3",
        prompt_mode="point",
    )

    assert "straight_foreground_valid_mae" in metrics
    assert "estimated_background_probe_straight_foreground_valid_mae" in metrics
    assert "oracle_background_probe_straight_foreground_valid_mae" in metrics
    assert "oracle_background_probe_straight_foreground_gradient_energy_ratio" in metrics
    assert "oracle_background_probe_render_object_psnr" in metrics
    assert "render_oracle_background_object_psnr" in metrics
    assert metrics["background_probe_batches"] == 1.0
    assert pipeline.backbone.clip_calls == 1


def test_oracle_component_diagnostic_exposes_alpha_error() -> None:
    background = torch.full((1, 3, 2, 2), 0.4)
    alpha_gt = torch.full((1, 1, 1, 2, 2), 0.25)
    alpha_pred = torch.full_like(alpha_gt, 0.5)
    foreground = torch.full((1, 1, 3, 2, 2), 0.8)
    frames = alpha_gt * foreground + (1.0 - alpha_gt) * background[:, None]
    matter = PhysicsMatterOutput(
        alpha=alpha_pred,
        straight_foreground=foreground,
        premultiplied_foreground=alpha_pred * foreground,
        color_transmission=torch.ones_like(foreground),
        transmittance=1.0 - alpha_pred,
        refractive_flow=torch.zeros(1, 1, 2, 2, 2),
        residual=torch.zeros_like(foreground),
        confidence=torch.ones_like(alpha_pred),
    )
    target = RefractiveGroundTruth(
        frames=frames, counterfactual_background=background,
        object_mask=torch.ones_like(alpha_gt), alpha=alpha_gt,
        straight_foreground=foreground,
        residual=torch.zeros_like(foreground),
    )
    metrics = {}
    _merge_oracle_component_metrics(
        metrics, SimpleNamespace(matter=matter), target,
        torch.ones_like(alpha_gt, dtype=torch.bool),
    )
    assert metrics["render_oracle_alpha_object_psnr"][0] > metrics[
        "render_oracle_background_object_psnr"
    ][0]
    assert "render_oracle_foreground_F_residual_R_object_psnr" in metrics
    assert "render_oracle_alpha_foreground_F_object_psnr" in metrics


def test_oracle_diagnostic_separates_background_flow_and_color_t() -> None:
    background = torch.full((1, 3, 4, 4), 0.5)
    predicted_background = torch.full_like(background, 0.2)
    alpha = torch.zeros(1, 1, 1, 4, 4)
    color_t = torch.full((1, 1, 3, 4, 4), 0.6)
    frames = color_t * background[:, None]
    matter = PhysicsMatterOutput(
        alpha=alpha,
        straight_foreground=torch.zeros_like(frames),
        premultiplied_foreground=torch.zeros_like(frames),
        color_transmission=torch.ones_like(frames),
        transmittance=torch.ones_like(frames),
        refractive_flow=torch.zeros(1, 1, 2, 4, 4),
        residual=torch.zeros_like(frames),
        confidence=torch.ones_like(alpha),
    )
    target = RefractiveGroundTruth(
        frames=frames, counterfactual_background=background,
        object_mask=torch.ones_like(alpha),
        alpha=alpha,
        premultiplied_foreground=torch.zeros_like(frames),
        color_transmission=color_t,
        transmittance=color_t,
        refractive_flow=matter.refractive_flow,
        residual=torch.zeros_like(frames),
    )
    metrics = {}
    _merge_oracle_component_metrics(
        metrics,
        SimpleNamespace(
            matter=matter, background=SimpleNamespace(background=predicted_background),
        ),
        target, torch.ones_like(alpha, dtype=torch.bool),
    )
    assert metrics["render_oracle_gt_flow_color_T_object_psnr"][0] > metrics[
        "render_oracle_background_object_psnr"
    ][0]
    assert "render_predicted_background_gt_flow_object_psnr" in metrics
    assert metrics["render_oracle_all_gt_object_psnr"][0] > metrics[
        "render_oracle_background_object_psnr"
    ][0]
    assert "render_predicted_all_gt_operator_object_psnr" in metrics


def test_past_visible_source_reveals_current_occlusion_at_refracted_coordinate() -> None:
    frames = torch.zeros(1, 2, 3, 4, 4)
    frames[:, 0, :, 1, 2] = torch.tensor([0.4, 0.6, 0.8])
    mask = torch.zeros(1, 2, 1, 4, 4)
    mask[:, 1, :, 1, 1] = 1.0
    mask[:, 1, :, 1, 2] = 1.0  # The source is covered *now*, not previously.
    flow = torch.zeros(1, 2, 2, 4, 4)
    flow[:, 1, 0, 1, 1] = 1.0
    tau = torch.ones_like(frames)
    tau[:, 1, :, 1, 1] = 0.5
    target = RefractiveGroundTruth(
        frames=frames,
        object_mask=mask,
        refractive_flow=flow,
        refractive_validity=torch.ones_like(mask),
        transmittance=tau,
    )

    evidence = _past_visible_refracted_background(target)
    assert evidence is not None
    sampled, support = evidence
    assert support[0, 0].sum() == 0
    assert support[0, 1, 0, 1, 1] == 1
    assert torch.allclose(sampled[0, 1, :, 1, 1], frames[0, 0, :, 1, 2])

    # A source hidden in every earlier frame must not be called direct evidence.
    target.object_mask[:, 0, :, 1, 2] = 1.0
    _, support = _past_visible_refracted_background(target)
    assert support[0, 1, 0, 1, 1] == 0


def test_leave_one_out_render_supervises_tau_and_flow_without_self_copy() -> None:
    frames = torch.zeros(1, 2, 3, 4, 4)
    for x in range(4):
        frames[:, :, :, :, x] = 0.1 + 0.1 * x
    mask = torch.zeros(1, 2, 1, 4, 4)
    mask[0, 1, 0, 1, 1] = 1
    frames[0, 1, :, 1, 1] = 0.6  # object appearance must not be used as B
    target = RefractiveGroundTruth(
        frames=frames,
        object_mask=mask,
        alpha=torch.zeros_like(mask),
        premultiplied_foreground=torch.zeros_like(frames),
        residual=torch.zeros_like(frames),
        transmittance=torch.ones_like(frames),
        refractive_validity=torch.ones_like(mask),
    )
    observed, coverage = _leave_one_out_background(target)
    assert torch.allclose(observed[0, 1, :, 1, 1], torch.full((3,), 0.2))
    assert coverage[0, 1, 0, 1, 1] == 1

    predicted_tau = torch.full_like(frames, 0.5, requires_grad=True)
    predicted_flow = torch.zeros(1, 2, 2, 4, 4)
    predicted_flow[0, 1, 0, 1, 1] = 0.25
    predicted_flow.requires_grad_()
    matter = PhysicsMatterOutput(
        alpha=torch.zeros_like(mask),
        premultiplied_foreground=torch.zeros_like(frames),
        straight_foreground=torch.zeros_like(frames),
        color_transmission=torch.ones_like(frames),
        transmittance=predicted_tau,
        refractive_flow=predicted_flow,
        residual=torch.zeros_like(frames),
        confidence=torch.ones_like(mask),
    )
    backbone = MAM2BackboneOutput(
        mask_logits=torch.zeros_like(mask),
        trimap_logits=torch.zeros(1, 2, 3, 4, 4),
        alpha_matte=torch.zeros_like(mask),
        non_memory_features=torch.zeros(1, 2, 1, 2, 2),
    )
    prediction = SimpleNamespace(
        backbone=backbone, matter=matter, reconstructed_frames=frames,
    )
    loss = RefractiveLoss()(prediction, target)["temporal_holdout_render"]
    assert loss > 0
    loss.backward()
    assert predicted_tau.grad[0, 1, :, 1, 1].abs().sum() > 0
    assert predicted_flow.grad[0, 1, 0, 1, 1].abs() > 0

    matter.refractive_kernel_weights = torch.full((1, 2, 2, 4, 4), 0.5)
    matter.refractive_kernel_flows = torch.stack(
        (predicted_flow.detach(), predicted_flow.detach()), dim=2
    )
    kernel_loss = RefractiveLoss()(prediction, target)["temporal_holdout_render"]
    assert torch.allclose(kernel_loss, loss.detach(), atol=1e-6)

    target.object_mask[0, 0, 0, 1, 1] = 1
    hidden, hidden_coverage = _leave_one_out_background(target)
    assert hidden_coverage[0, 1, 0, 1, 1] == 0
    assert hidden[0, 1, :, 1, 1].abs().sum() == 0


def test_temporal_surface_loss_uses_occlusion_support() -> None:
    video = torch.zeros(1, 2, 1, 4, 4)
    video[0, 1, 0, 1, 1] = 1
    flow = torch.zeros(1, 1, 2, 4, 4)
    support = torch.zeros(1, 1, 1, 4, 4)
    assert _temporal_loss(video, flow, support) == 0
    support[0, 0, 0, 1, 1] = 1
    assert _temporal_loss(video, flow, support) > 0


def test_past_visible_source_rejects_antialiased_foreground_contamination() -> None:
    frames = torch.zeros(1, 2, 3, 4, 4)
    mask = torch.zeros(1, 2, 1, 4, 4)
    alpha = torch.zeros_like(mask)
    alpha[:, 0, :, 1, 2] = 0.2
    mask[:, 1, :, 1, 1] = 1.0
    flow = torch.zeros(1, 2, 2, 4, 4)
    flow[:, 1, 0, 1, 1] = 1.0
    target = RefractiveGroundTruth(
        frames=frames,
        object_mask=mask,
        alpha=alpha,
        refractive_flow=flow,
    )
    _, support = _past_visible_refracted_background(target)
    assert support[0, 1, 0, 1, 1] == 0


def test_far_background_excludes_one_pixel_antialiased_silhouette() -> None:
    mask = torch.zeros(1, 1, 1, 7, 7)
    mask[:, :, :, 3, 3] = 1.0
    background = _far_background_support(mask)
    assert background[0, 0, 0, 3, 3] == 0
    assert background[0, 0, 0, 2, 3] == 0
    assert background[0, 0, 0, 3, 2] == 0
    assert background[0, 0, 0, 0, 0] == 1


def test_leakage_losses_penalize_object_terms_in_certain_background() -> None:
    frames = torch.rand(1, 1, 3, 8, 8)
    backbone = DummyBackbone(8)
    model = RefractiveMAM2(
        backbone,
        PipelineConfig(matter=MatterConfig(feature_channels=8, width=8)),
    )
    with torch.no_grad():
        backbone.mask.weight.zero_()
        backbone.mask.bias.fill_(10.0)
        model.matter.head.weight.zero_()
        model.matter.head.bias.zero_()
        model.matter.head.bias[4:6].fill_(1.0)
        model.matter.head.bias[10:13].fill_(1.0)
    semantics = backbone(frames)
    prediction = model.forward_from_backbone(
        frames,
        semantics,
        counterfactual_background_gt=frames[:, 0],
        use_ground_truth_background=True,
        skip_background_estimation=True,
    )
    target = RefractiveGroundTruth(
        frames=frames,
        object_mask=torch.zeros(1, 1, 1, 8, 8),
    )
    losses = RefractiveLoss()(prediction, target)
    assert losses["foreground_background_leakage"] > 0
    assert losses["flow_background_leakage"] > 0
    assert losses["residual_background_leakage"] > 0
    metrics = _batch_metrics(prediction, target)
    assert metrics["foreground_far_background_abs_mean"][0] > 0
    assert metrics["flow_far_background_epe"][0] > 0


def test_color_T_identifiability_and_zero_tau_have_different_support() -> None:
    mask = torch.ones(1, 1, 1, 2, 2)
    tau = torch.ones(1, 1, 3, 2, 2)
    tau[..., 0, 0] = 0.0
    target = RefractiveGroundTruth(
        frames=torch.zeros_like(tau), object_mask=mask, transmittance=tau,
    )
    valid_T = _identifiable_color_transmission(target)
    assert valid_T[0, 0, 0, 0, 0] == 0
    assert valid_T[0, 0, 0, 1, 1] == 1


def test_past_visible_transmission_loss_updates_tau_without_dark_pixel_division() -> None:
    frames = torch.zeros(1, 2, 3, 4, 4)
    source_rgb = torch.tensor([0.4, 0.6, 0.8])
    frames[0, 0, :, 1, 2] = source_rgb
    frames[0, 1, :, 1, 1] = 0.1 + 0.5 * source_rgb
    mask = torch.zeros(1, 2, 1, 4, 4)
    mask[0, 1, 0, 1, 1:3] = 1.0
    flow = torch.zeros(1, 2, 2, 4, 4)
    flow[0, 1, 0, 1, 1] = 1.0
    alpha = torch.zeros_like(mask)
    alpha[0, 1, 0, 1, 1] = 0.5
    alpha[0, 1, 0, 1, 2] = 1.0
    tau_gt = torch.ones_like(frames)
    tau_gt[0, 1, :, 1, 1] = 0.5
    tau_gt[0, 1, :, 1, 2] = 0.0
    foreground = torch.zeros_like(frames)
    foreground[0, 1, :, 1, 1] = 0.1
    target = RefractiveGroundTruth(
        frames=frames, object_mask=mask, alpha=alpha,
        premultiplied_foreground=foreground,
        color_transmission=torch.ones_like(frames),
        transmittance=tau_gt, residual=torch.zeros_like(frames),
        refractive_flow=flow, refractive_validity=torch.ones_like(mask),
    )
    predicted_tau = torch.full_like(frames, 0.4, requires_grad=True)
    predicted_flow = flow.clone().detach()
    predicted_flow[0, 1, 0, 1, 1] = 1.25
    predicted_flow.requires_grad_()
    matter = PhysicsMatterOutput(
        alpha=alpha,
        premultiplied_foreground=foreground,
        straight_foreground=foreground,
        color_transmission=torch.ones_like(frames),
        transmittance=predicted_tau,
        refractive_flow=predicted_flow,
        residual=torch.zeros_like(frames),
        confidence=torch.full_like(mask, 0.5),
    )
    backbone = MAM2BackboneOutput(
        mask_logits=torch.zeros_like(mask),
        trimap_logits=torch.zeros(1, 2, 3, 4, 4),
        alpha_matte=alpha,
        non_memory_features=torch.zeros(1, 2, 1, 2, 2),
    )
    prediction = SimpleNamespace(
        backbone=backbone, matter=matter, reconstructed_frames=frames,
    )
    losses = RefractiveLoss()(prediction, target)
    assert losses["temporal_visible_transmission"] > 0
    isolated_tau_gradient = torch.autograd.grad(
        losses["temporal_visible_transmission"], predicted_tau, retain_graph=True
    )[0]
    assert isolated_tau_gradient[0, 1, :, 1, 1].abs().sum() > 0
    assert losses["temporal_visible_joint_recomposition"] > 0
    losses["temporal_visible_joint_recomposition"].backward()
    assert predicted_tau.grad is not None
    assert predicted_flow.grad is not None
    assert predicted_tau.grad[0, 1, :, 1, 1].abs().sum() > 0
    assert predicted_tau.grad[0, 1, :, 1, 2].abs().sum() == 0
    assert predicted_flow.grad[0, 1, 0, 1, 1].abs() > 0
    assert predicted_flow.grad[0, 1, 0, 1, 2].abs() == 0


def test_supervised_flow_smoothness_preserves_valid_gt_discontinuity() -> None:
    frames = torch.zeros(1, 1, 3, 4, 4)
    ground_truth = torch.zeros(1, 1, 2, 4, 4)
    ground_truth[..., 2:] = 3.0
    support = torch.ones(1, 1, 1, 4, 4)
    exact = ground_truth.clone().requires_grad_()
    flat = torch.zeros_like(ground_truth).requires_grad_()

    exact_loss = _edge_aware_flow_smoothness(
        exact, frames, ground_truth, support
    )
    flat_loss = _edge_aware_flow_smoothness(
        flat, frames, ground_truth, support
    )

    assert exact_loss == 0
    assert flat_loss > 0
    flat_loss.backward()
    assert flat.grad is not None
    assert torch.isfinite(flat.grad).all()


def test_supervised_flow_smoothness_ignores_invalid_pixel_pairs() -> None:
    frames = torch.zeros(1, 1, 3, 4, 4)
    ground_truth = torch.zeros(1, 1, 2, 4, 4)
    prediction = ground_truth.clone()
    prediction[..., 0, 0] = 10.0
    support = torch.ones(1, 1, 1, 4, 4)
    support[..., 0, 0] = 0

    loss = _edge_aware_flow_smoothness(
        prediction, frames, ground_truth, support
    )

    assert loss == 0


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

    def __init__(self, channels: int) -> None:
        super().__init__(channels)
        self.clip_calls = 0

    def clear_mam2_cache(self):
        pass

    def forward_mam2_clip(
        self,
        frames,
        *,
        first_frame_mask_inputs=None,
        first_frame_point_inputs=None,
        compute_alpha=True,
    ):
        del first_frame_mask_inputs, first_frame_point_inputs, compute_alpha
        self.clip_calls += 1
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


@pytest.mark.parametrize(
    ("probability", "target", "message"),
    (
        ([float("nan")], [0.5], "prediction contains NaN or Inf"),
        ([0.5], [float("inf")], "target contains NaN or Inf"),
        ([1.1], [0.5], "prediction is outside"),
        ([0.5], [-0.1], "target is outside"),
    ),
)
def test_confidence_bce_rejects_invalid_values_before_kernel(
    probability, target, message
) -> None:
    with pytest.raises((FloatingPointError, ValueError), match=message):
        _probability_binary_cross_entropy(
            torch.tensor(probability), torch.tensor(target)
        )


def test_pipeline_shapes_and_gradient() -> None:
    frames = torch.rand(1, 2, 3, 32, 40)
    config = PipelineConfig(matter=MatterConfig(feature_channels=16, width=16))
    model = RefractiveMAM2(DummyBackbone(16), config)
    with torch.no_grad():
        model.backbone.mask.weight.zero_()
        model.backbone.mask.bias.fill_(4.0)
    output = model(frames)

    assert output.matter.alpha.shape == (1, 2, 1, 32, 40)
    assert output.matter.premultiplied_foreground.shape == frames.shape
    torch.testing.assert_close(
        output.matter.premultiplied_foreground,
        output.matter.alpha * output.matter.straight_foreground,
    )
    assert output.matter.transmittance.shape == frames.shape
    assert output.matter.residual.shape == frames.shape
    assert output.matter.refractive_flow.shape == (1, 2, 2, 32, 40)
    assert output.matter.refractive_kernel_weights is None
    assert output.matter.refractive_kernel_flows is None
    assert output.background.background.shape == (1, 3, 32, 40)
    assert output.background.video(frames.shape[1]).shape == frames.shape
    assert output.reconstructed_frames.shape == frames.shape

    loss = RefractiveLoss()(output, RefractiveGroundTruth(frames=frames))["total"]
    loss.backward()
    assert model.matter.head.weight.grad is not None
    assert model.matter.head.out_channels == 13
    assert torch.count_nonzero(model.matter.head.weight.grad[4:6]) > 0


def test_operator_head_gradient_metrics_track_flow_and_color_heads() -> None:
    model = RefractiveMAM2(
        DummyBackbone(8),
        PipelineConfig(
            matter=MatterConfig(
                feature_channels=8,
                width=8,
                flow_guided_latent_bridge=True,
            )
        ),
    )
    model.matter.head.weight.grad = torch.ones_like(model.matter.head.weight)
    bridge_head = model.matter.latent_bridge.head
    bridge_head.weight.grad = torch.ones_like(bridge_head.weight)
    metrics = _operator_head_gradient_metrics(model)

    assert set(metrics) == {
        "train/pam_head_grad_f",
        "train/pam_head_grad_flow",
        "train/pam_head_grad_t",
        "train/bridge_head_grad_f",
        "train/bridge_head_grad_flow",
        "train/bridge_head_grad_t",
    }
    assert all(torch.isfinite(value) and value > 0 for value in metrics.values())


def test_straight_foreground_supervision_and_metric_ignore_low_alpha() -> None:
    frames = torch.rand(1, 1, 3, 8, 8)
    model = RefractiveMAM2(
        DummyBackbone(8),
        PipelineConfig(matter=MatterConfig(feature_channels=8, width=8)),
    ).eval()
    prediction = model(frames)
    alpha_gt = torch.zeros(1, 1, 1, 8, 8)
    alpha_gt[..., 3, 3] = 0.5
    foreground_gt = prediction.matter.straight_foreground.detach().clone()
    target = RefractiveGroundTruth(
        frames=frames,
        object_mask=torch.ones_like(alpha_gt),
        alpha=alpha_gt,
        straight_foreground=foreground_gt,
    )
    baseline = RefractiveLoss()(prediction, target)["straight_foreground"]
    assert baseline == 0
    target.straight_foreground[..., 0, 0] += 0.5
    assert RefractiveLoss()(prediction, target)["straight_foreground"] == baseline
    target.straight_foreground[..., 3, 3] += 0.5
    assert RefractiveLoss()(prediction, target)["straight_foreground"] > baseline
    metrics = _batch_metrics(prediction, target)
    assert metrics["straight_foreground_valid_mae"][1] == 1
    assert metrics["straight_foreground_valid_mae"][0] > 0


def test_foreground_counterfactual_metrics_separate_alpha_and_color_error() -> None:
    frames = torch.rand(1, 1, 3, 8, 8)
    model = RefractiveMAM2(
        DummyBackbone(8),
        PipelineConfig(matter=MatterConfig(feature_channels=8, width=8)),
    ).eval()
    prediction = model(frames)
    prediction.matter.alpha = torch.full((1, 1, 1, 8, 8), 0.5)
    prediction.matter.straight_foreground = torch.full((1, 1, 3, 8, 8), 0.8)
    prediction.matter.premultiplied_foreground = (
        prediction.matter.alpha * prediction.matter.straight_foreground
    )

    alpha_only = RefractiveGroundTruth(
        frames=frames,
        alpha=torch.full_like(prediction.matter.alpha, 0.6),
        straight_foreground=prediction.matter.straight_foreground.clone(),
    )
    alpha_metrics = _batch_metrics(prediction, alpha_only)
    assert alpha_metrics["foreground_gt_alpha_probe_mae"][0] == pytest.approx(0)
    alpha_probe = alpha_metrics["foreground_gt_f_probe_mae"]
    assert alpha_probe[0] / alpha_probe[1] == pytest.approx(0.08)

    color_only = RefractiveGroundTruth(
        frames=frames,
        alpha=prediction.matter.alpha.clone(),
        straight_foreground=torch.full_like(prediction.matter.straight_foreground, 0.9),
    )
    color_metrics = _batch_metrics(prediction, color_only)
    assert color_metrics["foreground_gt_f_probe_mae"][0] == pytest.approx(0)
    color_probe = color_metrics["foreground_gt_alpha_probe_mae"]
    assert color_probe[0] / color_probe[1] == pytest.approx(0.05)


def test_foreground_laplacian_penalizes_blurred_interior_not_outside_support() -> None:
    sharp = torch.zeros(1, 1, 3, 7, 7)
    sharp[..., 3, 3] = 1.0
    blurred = F.avg_pool2d(sharp.flatten(0, 1), 3, stride=1, padding=1).unflatten(0, (1, 1))
    support = torch.zeros(1, 1, 1, 7, 7)
    support[..., 1:6, 1:6] = 1.0
    loss = _masked_laplacian_loss(blurred, sharp, support)
    assert loss > _masked_laplacian_loss(sharp, sharp, support)
    outside = sharp.clone()
    outside[..., 0, 0] = 1.0
    assert _masked_laplacian_loss(outside, sharp, support) == 0


def test_foreground_gradient_metric_detects_lost_interior_detail() -> None:
    frames = torch.rand(1, 1, 3, 8, 8)
    model = RefractiveMAM2(
        DummyBackbone(8),
        PipelineConfig(matter=MatterConfig(feature_channels=8, width=8)),
    ).eval()
    prediction = model(frames)
    alpha = torch.zeros(1, 1, 1, 8, 8)
    alpha[..., 2:6, 2:6] = 1.0
    foreground = prediction.matter.straight_foreground.detach().clone()
    baseline = RefractiveGroundTruth(
        frames=frames, alpha=alpha, straight_foreground=foreground.clone(),
    )
    assert abs(_batch_metrics(prediction, baseline)["straight_foreground_gradient_mae"][0]) < 1e-7
    foreground[..., 4, 4] += 0.5
    target = RefractiveGroundTruth(
        frames=frames, alpha=alpha, straight_foreground=foreground,
    )
    assert _batch_metrics(prediction, target)["straight_foreground_gradient_mae"][0] > 0


def test_foreground_detail_losses_penalize_blurring_not_sharp_gt() -> None:
    """The high-frequency losses must not reward a smooth foreground."""

    sharp = torch.zeros(1, 1, 3, 9, 9)
    sharp[..., 3:6, 3:6] = 1.0
    blurred = F.avg_pool2d(sharp.flatten(0, 1), 3, stride=1, padding=1)
    blurred = blurred.reshape_as(sharp)
    support = torch.ones(1, 1, 1, 9, 9)

    sharp_gradient_loss = _masked_gradient_loss(sharp, sharp, support)
    blurred_gradient_loss = _masked_gradient_loss(blurred, sharp, support)
    sharp_laplacian_loss = _masked_laplacian_loss(sharp, sharp, support)
    blurred_laplacian_loss = _masked_laplacian_loss(blurred, sharp, support)
    assert blurred_gradient_loss > sharp_gradient_loss
    assert blurred_laplacian_loss > sharp_laplacian_loss


def test_joint_sharp_loss_connects_refracted_detail_to_flow_and_background() -> None:
    torch.manual_seed(31)
    frames = torch.rand(1, 2, 3, 16, 16)
    model = RefractiveMAM2(
        DummyBackbone(8),
        PipelineConfig(
            matter=MatterConfig(feature_channels=8, width=8),
            joint_refinement_steps=1,
        ),
    )
    with torch.no_grad():
        model.backbone.mask.weight.zero_()
        model.backbone.mask.bias.fill_(8.0)
    output = model(frames)
    grid = torch.arange(16)
    checkerboard = ((grid[:, None] + grid[None, :]) % 2).float()
    background_gt = checkerboard[None, None].expand(1, 3, -1, -1)
    target = RefractiveGroundTruth(
        frames=frames,
        object_mask=torch.ones(1, 2, 1, 16, 16),
        refractive_validity=torch.ones(1, 2, 1, 16, 16),
        refractive_flow=torch.zeros(1, 2, 2, 16, 16),
        counterfactual_background=background_gt,
    )
    weights = _loss_weights_for_profile("joint_sharp")
    assert weights.refracted_background_gradient > 0
    assert _loss_weights_for_profile("optical_priority").refracted_background_gradient == 0
    terms = RefractiveLoss(weights)(output, target)
    assert terms["refracted_background_gradient"] > 0
    assert terms["render_object_gradient"] > 0
    assert terms["background_true_hole_gradient"] >= 0

    background_grad, flow_grad = torch.autograd.grad(
        terms["refracted_background_gradient"],
        (output.background.background, output.matter.refractive_flow),
        retain_graph=True,
    )
    assert torch.isfinite(background_grad).all() and background_grad.abs().sum() > 0
    assert torch.isfinite(flow_grad).all() and flow_grad.abs().sum() > 0
    foreground_grad = torch.autograd.grad(
        terms["render_object_gradient"], output.matter.premultiplied_foreground,
        retain_graph=True,
    )[0]
    assert torch.isfinite(foreground_grad).all() and foreground_grad.abs().sum() > 0


def test_low_alpha_straight_foreground_is_less_reliable_than_opaque_foreground() -> None:
    alpha = torch.tensor([[[[[0.01, 0.1, 1.0]]]]])
    target = RefractiveGroundTruth(
        frames=torch.zeros(1, 1, 3, 1, 3),
        alpha=alpha,
        straight_foreground=torch.ones(1, 1, 3, 1, 3),
    )
    reliability = _straight_foreground_reliability(target)
    assert reliability is not None
    assert torch.allclose(reliability.flatten(), torch.tensor([0.0, 0.1, 1.0]))


def test_antialiased_foreground_outside_center_mask_remains_supervised() -> None:
    frames = torch.rand(1, 1, 3, 8, 8)
    model = RefractiveMAM2(
        DummyBackbone(8),
        PipelineConfig(matter=MatterConfig(feature_channels=8, width=8)),
    ).eval()
    prediction = model(frames)
    center_mask = torch.zeros(1, 1, 1, 8, 8)
    center_mask[..., 3, 3] = 1.0
    alpha = torch.zeros_like(center_mask)
    alpha[..., 3, 4] = 0.5  # Antialiased silhouette outside center-ray mask.
    foreground = prediction.matter.straight_foreground.detach().clone()
    foreground[..., 3, 4] = 1.0 - foreground[..., 3, 4]
    target = RefractiveGroundTruth(
        frames=frames,
        object_mask=center_mask,
        alpha=alpha,
        straight_foreground=foreground,
        premultiplied_foreground=alpha * foreground,
    )
    terms = RefractiveLoss()(prediction, target)
    assert terms["straight_foreground"] > 0
    assert terms["premultiplied_foreground"] > 0
    assert "straight_foreground_gradient" in terms
    assert "premultiplied_foreground_gradient" in terms
    assert "render_object" in terms


def test_flow_refiner_backpropagates_through_joint_background_loop() -> None:
    frames = torch.rand(1, 2, 3, 16, 16)
    config = PipelineConfig(
        background=BackgroundConfig(
            completion_backbone="dilated",
            completion_width=8,
            completion_dilations=(1,),
            exclusion_dilation=1,
        ),
        matter=MatterConfig(
            feature_channels=8,
            width=8,
            encoder_depths=(1, 1, 1),
            temporal_blocks=0,
            refine_refractive_flow=True,
            # This test isolates the refiner gradient, independently of an
            # untrained dummy backbone's arbitrary binary mask prediction.
            hard_object_support=False,
        ),
        joint_refinement_steps=1,
    )
    model = RefractiveMAM2(DummyBackbone(8), config)
    with torch.no_grad():
        model.backbone.mask.weight.zero_()
        model.backbone.mask.bias.fill_(10.0)
    output = model(frames)
    assert output.matter.coarse_refractive_flow is not None
    metrics = _batch_metrics(
        output,
        RefractiveGroundTruth(
            frames=frames,
            refractive_flow=torch.zeros_like(output.matter.refractive_flow),
        ),
    )
    assert "flow_epe" in metrics
    assert "coarse_flow_epe" in metrics
    loss = (
        (output.reconstructed_frames - frames).square().mean()
        + output.matter.refractive_flow.square().mean()
    )
    loss.backward()
    assert model.matter.flow_refiner is not None
    gradient = model.matter.flow_refiner.head.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_pam_runs_in_fp32_under_bfloat16_autocast() -> None:
    frames = torch.rand(1, 1, 3, 16, 16)
    model = RefractiveMAM2(
        DummyBackbone(8),
        PipelineConfig(matter=MatterConfig(feature_channels=8, width=8)),
    )
    observed_dtypes = []
    hook = model.matter.image_stem.register_forward_hook(
        lambda _module, inputs, output: observed_dtypes.append(
            (inputs[0].dtype, output.dtype)
        )
    )
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            backbone_output = model.backbone(frames)
            output = model._matter_from_background(
                frames,
                torch.rand(1, 3, 16, 16),
                torch.zeros(1, 1, 16, 16),
                backbone_output,
            )
        assert observed_dtypes == [(torch.float32, torch.float32)]
        assert output.alpha.dtype == torch.float32
        output.alpha.mean().backward()
        assert model.matter.image_stem[0].weight.grad is not None
        assert torch.isfinite(model.matter.image_stem[0].weight.grad).all()
    finally:
        hook.remove()


def test_geometry_supervision_reaches_optional_pam_head() -> None:
    frames = torch.rand(1, 2, 3, 16, 16)
    config = PipelineConfig(
        matter=MatterConfig(
            feature_channels=8,
            width=8,
            encoder_depths=(1, 1, 1),
            temporal_blocks=0,
            predict_geometry=True,
        )
    )
    model = RefractiveMAM2(DummyBackbone(8), config)
    with torch.no_grad():
        model.backbone.mask.weight.zero_()
        model.backbone.mask.bias.fill_(4.0)
    output = model(frames)
    normal = torch.zeros(1, 2, 3, 16, 16)
    normal[:, :, 2] = 1.0
    validity = torch.ones(1, 2, 1, 16, 16)
    target = RefractiveGroundTruth(
        frames=frames,
        object_mask=validity,
        surface_normal=normal,
        depth=torch.full_like(validity, 2.0),
        normal_validity=validity,
        depth_validity=validity,
    )

    losses = RefractiveLoss()(output, target)

    assert {
        "surface_normal",
        "depth",
        "depth_gradient",
        "geometry_confidence",
    }.issubset(losses)
    losses["total"].backward()
    assert model.matter.head.weight.grad is not None
    assert torch.count_nonzero(
        model.matter.head.weight.grad[model.matter.geometry_start :]
    ) > 0


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


def test_stage2_oracle_background_skips_completion() -> None:
    frames = torch.rand(1, 2, 3, 24, 24)
    background_gt = torch.rand(1, 3, 24, 24)
    config = PipelineConfig(matter=MatterConfig(feature_channels=8, width=16))
    model = RefractiveMAM2(DummyBackbone(8), config)
    calls = []
    handle = model.background_model.completion.register_forward_hook(
        lambda _module, _inputs, _output: calls.append(1)
    )
    output = model(
        frames,
        counterfactual_background_gt=background_gt,
        use_ground_truth_background=True,
        skip_background_estimation=True,
    )
    handle.remove()

    assert calls == []
    assert torch.equal(output.background.background, background_gt)
    assert not output.background.true_hole.any()


def test_stage2_paired_images_apply_operator_reuse_supervision() -> None:
    frames = torch.rand(2, 1, 3, 16, 16)
    background_gt = torch.rand(2, 3, 16, 16)
    support = torch.ones(2, 1, 1, 16, 16)
    config = PipelineConfig(
        matter=MatterConfig(
            feature_channels=8,
            width=8,
            encoder_depths=(1, 1, 1),
            temporal_blocks=0,
            predict_geometry=True,
        )
    )
    model = RefractiveMAM2(DummyBackbone(8), config)
    output = model(
        frames,
        counterfactual_background_gt=background_gt,
        use_ground_truth_background=True,
        skip_background_estimation=True,
    )
    target = RefractiveGroundTruth(
        frames=frames,
        object_mask=support,
        counterfactual_background=background_gt,
    )
    batch = SimpleNamespace(
        ground_truth=target,
        paired_background_group_ids=["same-operator", "same-operator"],
    )

    losses = _loss_terms("2", output, batch)

    assert "operator_reuse" in losses
    assert torch.isfinite(losses["operator_reuse"])
    assert losses["total"] >= losses["operator_reuse"] * 0.5


def test_physics_alpha_refinement_preserves_the_one_pixel_support_limit() -> None:
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
    predicted_support = (
        output.backbone.mask_logits.sigmoid() >= 0.5
    ).to(delta.dtype)
    assert float((delta * predicted_support).detach().max()) <= 0.100001
    permitted = F.max_pool2d(
        predicted_support.flatten(0, 1), 3, stride=1, padding=1,
    ).reshape_as(predicted_support)
    assert torch.count_nonzero(output.matter.alpha * (1 - permitted)) == 0


def test_joint_loss_uses_independent_background_and_reports_fixed_gt_regions() -> None:
    frames = torch.rand(1, 2, 3, 8, 8)
    background_gt = torch.rand(1, 3, 8, 8)
    object_mask = torch.zeros(1, 2, 1, 8, 8)
    object_mask[..., 2:6, 2:6] = 1.0
    model = RefractiveMAM2(
        DummyBackbone(8),
        PipelineConfig(
            matter=MatterConfig(feature_channels=8, width=8),
            joint_refinement_steps=0,
        ),
    )
    prediction = model(frames)
    # Simulate the degenerate inverse/recompose fixed point: ordinary render
    # matches I exactly even though its physical decomposition is wrong.
    prediction.reconstructed_frames = frames
    target = RefractiveGroundTruth(
        frames=frames,
        object_mask=object_mask,
        counterfactual_background=background_gt,
        refractive_flow=torch.full((1, 2, 2, 8, 8), 2.0),
        residual=torch.full_like(frames, 0.1),
    )
    losses = joint_stage_loss(prediction, target)
    assert losses["render_oracle_background"] > losses["render"]
    assert torch.isfinite(losses["render_oracle_background"])
    losses["render_oracle_background"].backward()
    assert model.matter.head.weight.grad is not None
    metrics = _batch_metrics(
        prediction,
        target,
        matter_config=MatterConfig(
            max_refractive_flow_fraction=0.01,
            residual_scale=0.01,
        ),
    )
    assert "render_object_psnr" in metrics
    assert "background_gt_occluded_psnr" in metrics
    assert metrics["flow_gt_outside_capacity_fraction"][0] > 0
    assert metrics["residual_gt_outside_capacity_fraction"][0] > 0
    assert metrics["temporal_gt_available"][0] == 0


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


def test_learned_object_support_rechecks_direct_background_evidence(monkeypatch) -> None:
    frames = torch.rand(1, 2, 3, 16, 16)
    config = PipelineConfig(
        matter=MatterConfig(
            feature_channels=8, width=16, max_channels=64,
            learned_object_support=True,
        ),
        joint_refinement_steps=2,
    )
    model = RefractiveMAM2(DummyBackbone(8), config)
    original = model.background_model.observe
    supports = []

    def observe(value, support):
        supports.append(support.detach().clone())
        return original(value, support)

    monkeypatch.setattr(model.background_model, "observe", observe)
    output = model(frames)
    assert output.matter.object_support_logits is not None
    assert len(supports) == 1 + config.joint_refinement_steps
    assert all(torch.isfinite(value).all() for value in supports)


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


def test_evidence_flux_pipeline_never_invokes_learned_completion() -> None:
    frames = torch.rand(1, 2, 3, 16, 16)
    diffusion = StubDiffusion()
    config = PipelineConfig(
        background=BackgroundConfig(
            completion_variant="diffusion",
            evidence_flux_only=True,
            exclusion_dilation=1,
        ),
        matter=MatterConfig(feature_channels=8, width=16),
        joint_refinement_steps=2,
    )
    model = RefractiveMAM2(
        DummyBackbone(8),
        config,
        background_model=MaskedTemporalBackground(
            config.background,
            diffusion_completion=diffusion,
        ),
    )
    completion_calls = []
    handle = model.background_model.completion.register_forward_hook(
        lambda _module, _inputs, _output: completion_calls.append(1)
    )
    output = model(frames)
    handle.remove()
    assert output.reconstructed_frames.shape == frames.shape
    assert completion_calls == []
    assert diffusion.calls == 1


def test_pretrained_feature_flux_bridge_runs_in_joint_pipeline() -> None:
    frames = torch.rand(1, 2, 3, 16, 16)
    config = PipelineConfig(
        background=BackgroundConfig(
            completion_variant="diffusion",
            evidence_flux_only=True,
            exclusion_dilation=1,
        ),
        matter=MatterConfig(
            feature_channels=8,
            width=16,
            max_channels=64,
            flow_guided_latent_bridge=True,
            latent_bridge_width=8,
        ),
        joint_refinement_steps=2,
    )
    diffusion = StubDiffusion()
    model = RefractiveMAM2(
        DummyBackbone(8),
        config,
        background_model=MaskedTemporalBackground(
            config.background,
            diffusion_completion=diffusion,
        ),
    )
    with torch.no_grad():
        model.backbone.mask.weight.zero_()
        model.backbone.mask.bias.fill_(4.0)
    result = model(frames)
    assert diffusion.calls == 1
    assert result.reconstructed_frames.shape == frames.shape
    assert result.matter.coarse_refractive_flow is not None
    result.reconstructed_frames.square().mean().backward()
    assert model.matter.latent_bridge is not None
    assert model.matter.latent_bridge.head.weight.grad is not None


def test_pretrained_foreground_prior_is_used_once_and_can_be_finetuned() -> None:
    class StubForegroundPrior(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.color = nn.Parameter(torch.tensor(0.4))
            self.calls = 0

        def forward(self, frames, backbone):
            del backbone
            self.calls += 1
            b, t, _, h, w = frames.shape
            return self.color.expand(b, t, 16, h, w)

        def configure_trainability(self):
            self.requires_grad_(True)

    config = PipelineConfig(
        background=BackgroundConfig(completion_variant="diffusion", evidence_flux_only=True),
        matter=MatterConfig(
            feature_channels=8,
            width=16,
            max_channels=64,
            flow_guided_latent_bridge=True,
            use_fba_foreground_prior=True,
            latent_bridge_width=8,
        ),
        joint_refinement_steps=2,
    )
    prior = StubForegroundPrior()
    model = RefractiveMAM2(
        DummyBackbone(8),
        config,
        background_model=MaskedTemporalBackground(
            config.background, diffusion_completion=StubDiffusion()
        ),
        foreground_prior_model=prior,
    )
    model.matter.latent_bridge.head.weight.data.normal_(std=0.01)
    output = model(torch.rand(1, 2, 3, 16, 16))
    assert prior.calls == 1
    assert output.reconstructed_frames.shape == (1, 2, 3, 16, 16)
    output.reconstructed_frames.square().mean().backward()
    assert prior.color.grad is not None
    assert torch.isfinite(prior.color.grad)


def test_frozen_filled_background_still_backpropagates_render_error_to_flow() -> None:
    fixed_background = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    fixed_background = fixed_background.repeat(1, 3, 1, 1).detach()
    flow = torch.zeros(1, 2, 4, 4, requires_grad=True)
    rendered = warp_background(fixed_background, flow)
    rendered[..., 1:3, 1:3].sum().backward()
    assert flow.grad is not None
    assert flow.grad[..., 1:3, 1:3].abs().sum() > 0


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
        compute_connectivity=True,
    )
    assert "background_ssim" in metrics
    assert "background_true_hole_fraction" in metrics
    assert "evidence_preservation_l1" in metrics
    assert "alpha_connectivity_error" in metrics
    assert "alpha_boundary_f1" in metrics
    assert "flow_bad_pixel" in metrics


def test_background_direct_metrics_count_pixels_not_sample_psnr() -> None:
    frames = torch.zeros(2, 1, 3, 8, 8)
    output = RefractiveMAM2(
        DummyBackbone(8),
        PipelineConfig(matter=MatterConfig(feature_channels=8, width=8)),
    )(frames)
    output.background.background = torch.zeros(2, 3, 8, 8)
    output.background.background[0, :, 0, 0] = 1.0
    output.background.direct_support = torch.zeros(2, 1, 8, 8, dtype=torch.bool)
    output.background.direct_support[0, :, 0, 0:2] = True
    output.background.direct_support[1] = True
    metrics = _batch_metrics(
        output,
        RefractiveGroundTruth(
            frames=frames,
            counterfactual_background=torch.zeros(2, 3, 8, 8),
        ),
    )
    mse_sum, rgb_count = metrics["background_direct_pixel_mse"]
    bad_count, pixel_count = metrics["background_direct_large_error_fraction"]
    assert mse_sum / rgb_count == pytest.approx(1 / 66)
    assert bad_count / pixel_count == pytest.approx(1 / 66)


def test_metrics_report_neutral_color_transmission_collapse() -> None:
    frames = torch.rand(1, 1, 3, 8, 8)
    output = RefractiveMAM2(
        DummyBackbone(8),
        PipelineConfig(matter=MatterConfig(feature_channels=8, width=8)),
    )(frames)
    output.matter.color_transmission = torch.ones_like(frames)
    target_t = torch.tensor((0.55, 0.7, 1.0)).view(1, 1, 3, 1, 1).expand_as(frames)
    metrics = _batch_metrics(
        output,
        RefractiveGroundTruth(
            frames=frames,
            object_mask=torch.ones_like(output.matter.alpha),
            color_transmission=target_t,
            transmittance=target_t * 0.5,
        ),
    )
    collapsed, count = metrics["colored_T_neutral_collapse_fraction"]
    assert collapsed / count == 1.0
    gt_chroma, gt_count = metrics["colored_T_gt_chroma"]
    assert gt_chroma / gt_count == pytest.approx(0.45)
    pred_chroma, pred_count = metrics["colored_T_pred_chroma"]
    assert pred_chroma / pred_count == 0.0


def test_optical_priority_profile_penalizes_neutral_color_transmission() -> None:
    frames = torch.rand(1, 1, 3, 8, 8)
    output = RefractiveMAM2(
        DummyBackbone(8),
        PipelineConfig(matter=MatterConfig(feature_channels=8, width=8)),
    )(frames)
    output.matter.color_transmission = torch.ones_like(frames)
    target_t = torch.tensor((0.55, 0.7, 1.0)).view(1, 1, 3, 1, 1).expand_as(frames)
    target = RefractiveGroundTruth(
        frames=frames,
        object_mask=torch.ones_like(output.matter.alpha),
        color_transmission=target_t,
        transmittance=target_t * 0.5,
    )
    standard = RefractiveLoss(_loss_weights_for_profile("standard"))(output, target)
    optical = RefractiveLoss(_loss_weights_for_profile("optical_priority"))(
        output, target
    )
    assert "color_transmission_chroma" not in standard
    assert optical["color_transmission_chroma"] > 0
    assert _loss_weights_for_profile("optical_priority").refractive_flow < 1.0
    assert optical["total"] > standard["total"]


def test_optical_priority_penalizes_residual_that_exceeds_ground_truth() -> None:
    frames = torch.rand(1, 1, 3, 8, 8)
    output = RefractiveMAM2(
        DummyBackbone(8),
        PipelineConfig(matter=MatterConfig(feature_channels=8, width=8)),
    )(frames)
    output.matter.residual = torch.full_like(frames, 0.2)
    target = RefractiveGroundTruth(
        frames=frames,
        object_mask=torch.ones_like(output.matter.alpha),
        residual=torch.zeros_like(frames),
    )
    optical = RefractiveLoss(_loss_weights_for_profile("optical_priority"))(
        output, target,
    )
    assert optical["residual_excess"] > 0.1
    assert "residual_excess" not in RefractiveLoss(
        _loss_weights_for_profile("standard")
    )(output, target)
