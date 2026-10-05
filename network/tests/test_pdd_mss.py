import pytest

torch = pytest.importorskip("torch")

from refractive_mam2 import MemorySeparableSiamese, PDDConfig
from refractive_mam2 import PromptableDualModeDecoder
from sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2.modeling.sam.transformer import TwoWayTransformer


def test_paper_pdd_mss_shapes() -> None:
    config = PDDConfig(feature_channels=16, width=16, depth=1)
    module = MemorySeparableSiamese(config=config)
    memory = torch.rand(2, 16, 8, 10)
    clean = torch.rand_like(memory)
    seed = torch.randn(2, 1, 16, 20)
    output = module(memory, clean, seed)

    assert output.mask_logits.shape == seed.shape
    assert output.trimap_logits.shape == (2, 3, 16, 20)
    assert output.non_memory_features.data_ptr() == clean.data_ptr()
    assert module.pdd.mask_output_token.weight.shape == (1, 16)
    assert module.pdd.trimap_output_tokens.weight.shape == (3, 16)


def test_mss_reuses_one_pdd_instance() -> None:
    module = MemorySeparableSiamese(
        config=PDDConfig(feature_channels=8, width=8, depth=1, prompt_heads=4)
    )
    assert len([name for name, _ in module.named_modules() if name.endswith("pdd")]) == 1


def test_second_pass_encodes_refined_mask() -> None:
    module = MemorySeparableSiamese(
        config=PDDConfig(feature_channels=8, width=8, depth=1, prompt_heads=4)
    )
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
    assert not torch.allclose(output.mask_logits, seed)


def test_mask_guidance_reaches_pdd_trimap_branch() -> None:
    module = MemorySeparableSiamese(
        config=PDDConfig(feature_channels=16, width=16, depth=1)
    )
    memory = torch.rand(1, 16, 4, 4, requires_grad=True)
    seed = torch.zeros(1, 1, 8, 8)

    output = module(memory, memory.clone(), seed)
    output.trimap_logits.mean().backward()

    assert module.pdd.mask_augmentation[0].weight.grad is not None
    assert module.pdd.trimap_fusion[0].weight.grad is not None
    assert module.pdd.trimap_output_tokens.weight.grad is not None


def test_pdd_initializes_from_official_sam2_decoder() -> None:
    width = 16
    source = MaskDecoder(
        transformer_dim=width,
        transformer=TwoWayTransformer(
            depth=1,
            embedding_dim=width,
            mlp_dim=64,
            num_heads=4,
        ),
        use_high_res_features=True,
        pred_obj_scores=True,
    )
    pdd = PromptableDualModeDecoder(
        PDDConfig(
            feature_channels=width,
            width=width,
            depth=1,
            prompt_heads=4,
            transformer_mlp_dim=64,
        )
    )
    pdd.initialize_from_sam2(source)

    features = torch.randn(2, width, 4, 5, requires_grad=True)
    output = pdd(
        features,
        torch.randn(2, 1, 16, 20),
        image_pe=torch.randn(1, width, 4, 5),
        high_res_features=(
            torch.randn(2, width, 16, 20),
            torch.randn(2, width, 8, 10),
        ),
    )
    (output.mask_logits.mean() + output.trimap_logits.mean()).backward()

    assert pdd.sam2_initialized
    assert output.mask_logits.shape == (2, 1, 16, 20)
    assert output.trimap_logits.shape == (2, 3, 16, 20)
    assert features.grad is not None
