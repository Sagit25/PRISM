import pytest

torch = pytest.importorskip("torch")

from refractive_mam2.training import (
    configure_stage1a,
    configure_stage1b,
    configure_stage2,
    configure_stage3,
    configure_stage4,
)


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
