from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

import refractive_mam2.train as train_module

from refractive_mam2.training import (
    SemanticTargets,
    configure_stage1a,
    configure_stage1b,
    configure_stage2,
    configure_stage3,
    configure_stage4,
    selective_semantic_loss,
)
from refractive_mam2.config import BackgroundConfig
from refractive_mam2.background import MaskedTemporalBackground
from refractive_mam2.logger import WandbLogger
from refractive_mam2.losses import RefractiveGroundTruth
from refractive_mam2.train import (
    _alpha_gt_statistics,
    _alpha_gt_values,
    _loader,
    _preview_mask_overlay,
    _preview_temporal_gt,
    _semantic_montages,
    _supervised_prediction_montages,
    _training_teacher_forcing,
    _validate_diffusion_training_mode,
)


def test_frozen_flux_training_runs_on_every_applicable_stage_iteration(
    monkeypatch,
) -> None:
    monkeypatch.setattr(train_module.random, "random", lambda: 0.0)
    assert _training_teacher_forcing("2", 0.5, evidence_flux_only=True)
    assert _training_teacher_forcing("3", 0.5, evidence_flux_only=False)
    assert not _training_teacher_forcing("3", 1.0, evidence_flux_only=True)
    assert not _training_teacher_forcing("4", 1.0, evidence_flux_only=True)
    for stage in ("3", "4"):
        _validate_diffusion_training_mode(
            stage=stage, mode="train", completion_variant="diffusion",
            evidence_flux_only=True,
        )
    with pytest.raises(SystemExit, match="only with --evidence-flux-only"):
        _validate_diffusion_training_mode(
            stage="3", mode="train", completion_variant="diffusion",
            evidence_flux_only=False,
        )
    with pytest.raises(SystemExit, match="only with --evidence-flux-only"):
        _validate_diffusion_training_mode(
            stage="2", mode="train", completion_variant="diffusion",
            evidence_flux_only=True,
        )
from refractive_mam2.types import MAM2BackboneOutput


class _EmptyRemoteDataset(torch.utils.data.IterableDataset):
    def __iter__(self):
        return iter(())


class _LoRAModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = torch.nn.Linear(2, 2)
        self.lora_A = torch.nn.Parameter(torch.zeros(1, 2))
        self.lora_B = torch.nn.Parameter(torch.zeros(2, 1))


class _Predictor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sam2_base = _LoRAModule()
        self.mam2_mss = torch.nn.Linear(2, 2)
        self.mam2_matter = torch.nn.Linear(2, 2)


class _Pipeline(torch.nn.Module):
    def __init__(self, predictor: _Predictor) -> None:
        super().__init__()
        self.backbone = predictor
        self.matter = torch.nn.Linear(2, 2)
        self.background_model = torch.nn.Linear(2, 2)
        self.config = SimpleNamespace(background=BackgroundConfig())


def _is_trainable(module: torch.nn.Module) -> bool:
    return any(parameter.requires_grad for parameter in module.parameters())


def test_remote_loader_uses_one_prefetch_process_without_duplicate_owners() -> None:
    loader = _loader(
        _EmptyRemoteDataset(),
        batch_size=2,
        shuffle=False,
        workers=1,
        paired_backgrounds=False,
        seed=0,
    )
    assert loader.num_workers == 1
    assert loader.prefetch_factor == 2
    with pytest.raises(ValueError, match="only --workers 0 or 1"):
        _loader(
            _EmptyRemoteDataset(),
            batch_size=2,
            shuffle=False,
            workers=2,
            paired_backgrounds=False,
            seed=0,
        )


def test_explicit_curriculum_assigns_non_overlapping_responsibilities() -> None:
    predictor = _Predictor()
    pipeline = _Pipeline(predictor)

    configure_stage1a(predictor)
    assert _is_trainable(predictor.mam2_mss)
    assert not _is_trainable(predictor.mam2_matter)
    assert predictor.sam2_base.lora_A.requires_grad
    assert not predictor.sam2_base.base.weight.requires_grad

    configure_stage1b(predictor)
    assert not _is_trainable(predictor.mam2_mss)
    assert _is_trainable(predictor.mam2_matter)
    assert not predictor.sam2_base.lora_A.requires_grad

    configure_stage2(predictor, pipeline)
    assert _is_trainable(pipeline.matter)
    assert not _is_trainable(pipeline.background_model)
    assert not _is_trainable(predictor)

    configure_stage3(predictor, pipeline)
    assert _is_trainable(pipeline.matter)
    assert _is_trainable(pipeline.background_model)
    assert not _is_trainable(predictor)

    configure_stage4(predictor, pipeline)
    assert _is_trainable(predictor.mam2_mss)
    assert _is_trainable(predictor.mam2_matter)
    assert predictor.sam2_base.lora_A.requires_grad
    assert not predictor.sam2_base.base.weight.requires_grad
    assert _is_trainable(pipeline.matter)
    assert _is_trainable(pipeline.background_model)


def test_evidence_flux_curriculum_trains_only_background_correction() -> None:
    predictor = _Predictor()
    pipeline = _Pipeline(predictor)
    pipeline.config.background.evidence_flux_only = True

    stage3 = configure_stage3(predictor, pipeline)
    assert _is_trainable(pipeline.matter)
    assert _is_trainable(pipeline.background_model)
    assert any(parameter is pipeline.background_model.weight for parameter in stage3)

    stage4 = configure_stage4(predictor, pipeline)
    assert _is_trainable(pipeline.matter)
    assert _is_trainable(pipeline.background_model)
    assert any(parameter is pipeline.background_model.weight for parameter in stage4)
    assert _is_trainable(predictor.mam2_mss)
    assert _is_trainable(predictor.mam2_matter)
    assert predictor.sam2_base.lora_A.requires_grad
    assert not predictor.sam2_base.base.weight.requires_grad


def test_frozen_diffusion_parameters_stay_frozen_while_refiner_trains() -> None:
    predictor = _Predictor()
    pipeline = _Pipeline(predictor)
    diffusion = torch.nn.Conv2d(3, 3, 1)
    pipeline.background_model = MaskedTemporalBackground(
        BackgroundConfig(
            completion_backbone="dilated",
            completion_width=8,
            evidence_flux_only=True,
            completion_variant="diffusion",
        ),
        diffusion_completion=diffusion,
    )
    configure_stage3(predictor, pipeline)
    assert _is_trainable(pipeline.background_model.completion)
    assert not _is_trainable(diffusion)


def test_paper_mam2_losses_are_finite_and_differentiable() -> None:
    mask = torch.randn(1, 1, 1, 16, 16, requires_grad=True)
    trimap = torch.randn(1, 1, 3, 16, 16, requires_grad=True)
    alpha = torch.sigmoid(torch.randn(1, 1, 1, 16, 16, requires_grad=True))
    alpha.retain_grad()
    target_alpha = torch.rand_like(alpha)
    target_trimap = torch.where(
        target_alpha <= 0.05,
        torch.zeros_like(target_alpha, dtype=torch.long),
        torch.where(
            target_alpha >= 0.95,
            torch.full_like(target_alpha, 2, dtype=torch.long),
            torch.ones_like(target_alpha, dtype=torch.long),
        ),
    )
    losses = selective_semantic_loss(
        mask,
        trimap,
        alpha,
        SemanticTargets(
            trimap=target_trimap,
            alpha=target_alpha,
            alpha_validity=torch.ones_like(target_alpha),
        ),
        "image_matting",
    )

    expected = {
        "trimap",
        "mam2_alpha_unknown_l1",
        "mam2_alpha_known_l1",
        "mam2_alpha_l2",
        "mam2_alpha_laplacian",
        "mam2_alpha_gradient",
        "total",
    }
    assert expected.issubset(losses)
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert trimap.grad is not None
    assert alpha.grad is not None


def _diagnostic_target() -> RefractiveGroundTruth:
    frames = torch.zeros(1, 4, 3, 8, 8)
    masks = torch.zeros(1, 4, 1, 8, 8)
    masks[:, :, :, 2:6, 2:6] = 1
    alpha = torch.zeros_like(masks)
    alpha[:, 1, :, 2:6, 2:6] = 0.5
    alpha[:, 2, :, 2:6, 2:6] = 1.0
    alpha[:, 3, :, 2:6, 2:6] = 0.5
    return RefractiveGroundTruth(
        frames=frames,
        object_mask=masks,
        alpha=alpha,
    )


def test_alpha_gt_diagnostics_are_support_conditioned() -> None:
    target = _diagnostic_target()
    values = _alpha_gt_values(target)
    assert values is not None
    assert values.numel() == 64
    statistics = _alpha_gt_statistics(target)
    assert torch.isclose(statistics["alpha_gt_mean"], torch.tensor(0.5))
    assert torch.isclose(
        statistics["alpha_gt_near_clear_fraction"], torch.tensor(0.25)
    )
    assert torch.isclose(
        statistics["alpha_gt_translucent_fraction"], torch.tensor(0.5)
    )
    assert torch.isclose(
        statistics["alpha_gt_near_opaque_fraction"], torch.tensor(0.25)
    )
    assert torch.isclose(statistics["object_mask_coverage"], torch.tensor(0.25))


def test_gt_overlay_and_temporal_strip_preserve_panel_size() -> None:
    target = _diagnostic_target()
    overlay = _preview_mask_overlay(
        target.frames[0, 0], target.object_mask[0, 0]
    )
    temporal = _preview_temporal_gt(target.frames[0], target.object_mask[0])
    assert overlay.size == (8, 8)
    assert temporal.size == (8, 8)
    overlay_array = np.asarray(overlay)
    assert ((overlay_array[..., 0] == 255) & (overlay_array[..., 1] == 48)).any()


def test_stage1_montage_logs_input_and_prediction_gt_pairs_without_red_overlay(
    monkeypatch,
) -> None:
    target = _diagnostic_target()
    target.trimap = torch.ones_like(target.object_mask, dtype=torch.long)
    semantics = MAM2BackboneOutput(
        mask_logits=torch.zeros(1, 4, 1, 8, 8),
        trimap_logits=torch.zeros(1, 4, 3, 8, 8),
        alpha_matte=torch.zeros(1, 4, 1, 1, 1),
        non_memory_features=torch.zeros(1, 4, 2, 8, 8),
    )
    labels = []
    original_grid = train_module._labeled_grid

    def capture_grid(panels, **kwargs):
        labels.extend(label for label, _ in panels)
        return original_grid(panels, **kwargs)

    monkeypatch.setattr(train_module, "_labeled_grid", capture_grid)
    monkeypatch.setattr(
        train_module,
        "_preview_mask_overlay",
        lambda *_args: pytest.fail("Stage 1 must not draw a red GT contour"),
    )
    monkeypatch.setattr(
        train_module,
        "_preview_temporal_gt",
        lambda *_args: pytest.fail("Stage 1 must not log a red temporal strip"),
    )
    montages = _semantic_montages(
        semantics,
        target,
        ["sequence_0"],
        limit=1,
        stage="1a",
    )
    assert labels == [
        "input I",
        "mask pred",
        "mask GT",
        "trimap pred",
        "trimap GT",
    ]
    assert montages[0][1] == "sequence_0"
    assert montages[0][0].size[0] == 4 * 140
    pixels = np.asarray(montages[0][0])
    assert not ((pixels[..., 0] == 255) & (pixels[..., 1] == 48)).any()


def test_stage1b_montage_only_shows_alpha_target(monkeypatch) -> None:
    target = _diagnostic_target()
    semantics = MAM2BackboneOutput(
        mask_logits=torch.zeros(1, 4, 1, 8, 8),
        trimap_logits=torch.zeros(1, 4, 3, 8, 8),
        alpha_matte=torch.full((1, 4, 1, 8, 8), 0.4),
        non_memory_features=torch.zeros(1, 4, 2, 8, 8),
    )
    labels = []
    original_grid = train_module._labeled_grid

    def capture_grid(panels, **kwargs):
        labels.extend(label for label, _ in panels)
        return original_grid(panels, **kwargs)

    monkeypatch.setattr(train_module, "_labeled_grid", capture_grid)
    montages = _semantic_montages(
        semantics, target, ["sequence_0"], limit=1, stage="1b"
    )
    assert labels == ["input I", "alpha pred", "alpha GT"]
    assert montages[0][1] == "sequence_0"


def _physics_preview_fixture():
    target = _diagnostic_target()
    target.trimap = torch.ones_like(target.object_mask, dtype=torch.long)
    target.premultiplied_foreground = torch.full_like(target.frames, 0.2)
    target.straight_foreground = torch.full_like(target.frames, 0.4)
    target.color_transmission = torch.full_like(target.frames, 0.8)
    target.transmittance = torch.full_like(target.alpha, 0.7)
    target.refractive_flow = torch.zeros(1, 4, 2, 8, 8)
    target.residual = torch.zeros_like(target.frames)
    target.confidence = torch.ones_like(target.alpha)
    target.counterfactual_background = torch.full((1, 3, 8, 8), 0.3)
    support = torch.ones(1, 1, 8, 8, dtype=torch.bool)
    background = SimpleNamespace(
        background=torch.full((1, 3, 8, 8), 0.25),
        observed_background=torch.full((1, 3, 8, 8), 0.2),
        inverse_background=torch.full((1, 3, 8, 8), 0.15),
        direct_support=support,
        inverse_support=support,
        true_hole=support,
    )
    matter = SimpleNamespace(
        alpha=torch.full_like(target.alpha, 0.4),
        premultiplied_foreground=torch.full_like(target.frames, 0.18),
        straight_foreground=torch.full_like(target.frames, 0.45),
        color_transmission=torch.full_like(target.frames, 0.75),
        transmittance=torch.full_like(target.alpha, 0.65),
        refractive_flow=torch.zeros_like(target.refractive_flow),
        residual=torch.zeros_like(target.frames),
        confidence=torch.full_like(target.alpha, 0.9),
        surface_normal=None,
        depth=None,
    )
    semantics = MAM2BackboneOutput(
        mask_logits=torch.zeros(1, 4, 1, 8, 8),
        trimap_logits=torch.zeros(1, 4, 3, 8, 8),
        alpha_matte=torch.full_like(target.alpha, 0.45),
        non_memory_features=torch.zeros(1, 4, 2, 8, 8),
    )
    prediction = SimpleNamespace(
        reconstructed_frames=torch.full_like(target.frames, 0.1),
        backbone=semantics,
        matter=matter,
        background=background,
    )
    return prediction, target


@pytest.mark.parametrize("stage,teacher_forcing", [("2", True), ("3", True), ("4", False)])
def test_physics_montage_shows_recomposition_and_stage_targets(
    monkeypatch, stage, teacher_forcing
) -> None:
    prediction, target = _physics_preview_fixture()
    labels = []
    original_grid = train_module._labeled_grid

    def capture_grid(panels, **kwargs):
        labels.extend(label for label, _ in panels)
        return original_grid(panels, **kwargs)

    monkeypatch.setattr(train_module, "_labeled_grid", capture_grid)
    montage, caption = _supervised_prediction_montages(
        prediction, target, ["sequence_0"], limit=1,
        stage=stage, teacher_forcing=teacher_forcing,
    )[0]
    assert montage.size[0] == 4 * 160
    assert labels[:3] == ["input I t0 (T4)", "recomposed I pred", "render error x4"]
    for name in (
        "object support", "PAM alpha", "foreground F (valid, tone-mapped)", "foreground G", "color T", "transmittance tau",
        "residual R", "flow u / Phi", "confidence",
    ):
        assert f"{name} pred" in labels
        assert f"{name} GT" in labels
    assert "foreground F GT-alpha valid" in labels

    if stage == "2":
        assert "oracle B input" in labels
        assert "background B pred" not in labels
        assert "mask pred" not in labels
    else:
        assert "background B pred" in labels
        assert "background B GT" in labels
        assert "direct B pred" in labels
        assert "inverse B pred" in labels
        assert "hole B pred" in labels
        assert ("oracle B render input" in labels) == teacher_forcing
    if stage == "4":
        for name in ("mask", "trimap", "MAM2 alpha"):
            assert f"{name} pred" in labels
            assert f"{name} GT" in labels
    else:
        assert "MAM2 alpha pred" not in labels
    assert ("recomposition uses oracle background" in caption) == teacher_forcing


def test_wandb_histogram_logging_bounds_payload_to_tensor_values() -> None:
    class _FakeWandb:
        def __init__(self) -> None:
            self.logged = []

        @staticmethod
        def Histogram(value):
            return ("histogram", np.asarray(value).copy())

        def log(self, payload, *, commit):
            self.logged.append((payload, commit))

    logger = WandbLogger.__new__(WandbLogger)
    logger._run = object()
    logger._wandb = _FakeWandb()
    logger.log_histograms(
        {"train/alpha_gt_distribution": torch.tensor([0.0, 0.5, 1.0])},
        7,
        commit=False,
    )
    payload, commit = logger._wandb.logged[0]
    assert payload["global_step"] == 7
    assert payload["train/alpha_gt_distribution"][0] == "histogram"
    assert commit is False
