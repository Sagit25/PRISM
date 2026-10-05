import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from refractive_mam2.training import (
    SemanticTargets,
    configure_stage1a,
    configure_stage1b,
    configure_stage2,
    configure_stage3,
    configure_stage4,
    selective_semantic_loss,
)
from refractive_mam2.logger import WandbLogger
from refractive_mam2.losses import RefractiveGroundTruth
from refractive_mam2.train import (
    _alpha_gt_statistics,
    _alpha_gt_values,
    _loader,
    _preview_mask_overlay,
    _preview_temporal_gt,
)


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
