import pytest

torch = pytest.importorskip("torch")

from refractive_mam2 import MemorySeparableSiamese, PDDConfig


def test_mss_shapes_and_zero_initialized_mask_residual() -> None:
    config = PDDConfig(feature_channels=16, width=16, depth=1)
    module = MemorySeparableSiamese(config=config)
    memory = torch.rand(2, 16, 8, 10)
    clean = torch.rand_like(memory)
    seed = torch.randn(2, 1, 16, 20)
    output = module(memory, clean, seed)

    assert torch.allclose(output.mask_logits, seed)
    assert output.trimap_logits.shape == (2, 3, 16, 20)
    assert output.non_memory_features.data_ptr() == clean.data_ptr()


def test_mss_reuses_one_pdd_instance() -> None:
    module = MemorySeparableSiamese(config=PDDConfig(feature_channels=8, width=8, depth=1))
    assert len([name for name, _ in module.named_modules() if name.endswith("pdd")]) == 1


def test_second_pass_encodes_refined_mask() -> None:
    module = MemorySeparableSiamese(
        config=PDDConfig(feature_channels=8, width=8, depth=1)
    )
    with torch.no_grad():
        module.pdd.mask_head.bias.fill_(1.0)
    memory = torch.rand(1, 8, 4, 4)
    seed = torch.zeros(1, 1, 8, 8)
    captured = {}

    def encoder(mask_logits):
        captured["mask"] = mask_logits
        return (
            torch.zeros(1, 1, 8),
            torch.zeros(1, 8, 4, 4),
        )

    output = module(memory, memory.clone(), seed, trimap_prompt_encoder=encoder)
    assert torch.allclose(captured["mask"], output.mask_logits)
    assert torch.allclose(output.mask_logits, torch.ones_like(seed))
