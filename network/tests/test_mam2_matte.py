import pytest

torch = pytest.importorskip("torch")

from refractive_mam2 import (
    ExternalMEMatteMatter,
    MAM2MatteConfig,
    MAM2TrimapMatter,
)


def test_mam2_matter_enforces_known_trimap_regions_at_inference() -> None:
    matter = MAM2TrimapMatter(
        MAM2MatteConfig(width=8, depth=1, hard_trimap_at_inference=True)
    ).eval()
    frames = torch.rand(1, 2, 3, 12, 16)
    trimap = torch.full((1, 2, 3, 6, 8), -20.0)
    trimap[:, :, 0, :, :3] = 20.0
    trimap[:, :, 1, :, 3:5] = 20.0
    trimap[:, :, 2, :, 5:] = 20.0

    alpha = matter(frames, trimap)

    assert alpha.shape == (1, 2, 1, 12, 16)
    assert torch.all(alpha[..., :6] == 0)
    assert torch.all(alpha[..., 10:] == 1)
    assert torch.all((alpha[..., 6:10] >= 0) & (alpha[..., 6:10] <= 1))


def test_mam2_matter_is_differentiable_in_unknown_region() -> None:
    matter = MAM2TrimapMatter(MAM2MatteConfig(width=8, depth=1)).train()
    frames = torch.rand(1, 1, 3, 16, 16)
    trimap = torch.zeros(1, 1, 3, 8, 8)
    trimap[:, :, 1] = 4.0
    alpha = matter(frames, trimap)
    alpha.mean().backward()
    assert matter.alpha_head.weight.grad is not None


class _FakeMEMatte(torch.nn.Module):
    def forward(self, inputs, patch_decoder=True):
        assert patch_decoder
        alpha = torch.full_like(inputs["trimap"], 0.3)
        return {"phas": alpha}, [], []


class _TrainableFakeMEMatte(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Conv2d(3, 1, 1)
        self.decoder = torch.nn.Conv2d(1, 1, 1)

    def forward(self, inputs, patch_decoder=True):
        encoded = self.backbone(inputs["image"])
        alpha = torch.sigmoid(self.decoder(encoded) + inputs["trimap"])
        return {"phas": alpha}, [], []


def test_external_mematte_adapter_uses_official_io_contract() -> None:
    matter = ExternalMEMatteMatter(_FakeMEMatte()).eval()
    frames = torch.rand(1, 1, 3, 8, 12)
    trimap = torch.full((1, 1, 3, 4, 6), -20.0)
    trimap[:, :, 0, :, :2] = 20.0
    trimap[:, :, 1, :, 2:4] = 20.0
    trimap[:, :, 2, :, 4:] = 20.0
    alpha = matter(frames, trimap)
    assert torch.all(alpha[..., :4] == 0)
    assert torch.allclose(alpha[..., 4:8], torch.full_like(alpha[..., 4:8], 0.3))
    assert torch.all(alpha[..., 8:] == 1)


def test_external_mematte_decoder_and_soft_trimap_are_differentiable() -> None:
    external = _TrainableFakeMEMatte()
    matter = ExternalMEMatteMatter(external).train()
    matter.configure_trainable(True)
    frames = torch.rand(1, 1, 3, 8, 8)
    trimap = torch.zeros(1, 1, 3, 4, 4, requires_grad=True)

    matter(frames, trimap).mean().backward()

    assert trimap.grad is not None
    assert trimap.grad.abs().sum() > 0
    assert external.decoder.weight.grad is not None
    assert external.backbone.weight.grad is None
    assert not external.training
