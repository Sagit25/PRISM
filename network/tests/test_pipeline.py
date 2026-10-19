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
    _predict,
)
from refractive_mam2.training import joint_stage_loss
from refractive_mam2.losses import (
    _edge_aware_flow_smoothness,
    _far_background_support,
    _identifiable_color_transmission,
    _past_visible_refracted_background,
    _probability_binary_cross_entropy,
)
from refractive_mam2.renderer import warp_background
from refractive_mam2.matter import PhysicsMatterOutput


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
    predicted_support = (
        output.backbone.mask_logits.sigmoid() >= 0.5
    ).to(delta.dtype)
    assert float((delta * predicted_support).detach().max()) <= 0.100001
    assert torch.count_nonzero(output.matter.alpha * (1 - predicted_support)) == 0


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


def test_evidence_flux_pipeline_never_runs_learned_completion() -> None:
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
