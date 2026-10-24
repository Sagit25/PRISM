import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")
F = pytest.importorskip("torch.nn.functional")

from refractive_mam2.color import linear_to_srgb
from refractive_mam2.fba_prior import FBAPretrainedForegroundPrior
from refractive_mam2.types import MAM2BackboneOutput


class _StubFBA(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Identity()
        self.decoder = nn.Module()
        self.decoder.conv_up4 = nn.Sequential(
            nn.Conv2d(3, 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(16, 16, 1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(16, 7, 1),
        )
        self.seen = None

    def forward(self, image, trimap, normalized, distances):
        self.seen = (image.shape, trimap.shape, normalized.shape, distances.shape)
        self.seen_image = image.detach().clone()
        return self.decoder.conv_up4(image)


def test_fba_prior_preprocessing_and_decoder_gradient() -> None:
    # Use a small stand-in for upstream FBA to test our integration contract
    # without requiring its separately licensed checkpoint in CI.
    prior = FBAPretrainedForegroundPrior.__new__(FBAPretrainedForegroundPrior)
    nn.Module.__init__(prior)
    prior.input_side = 64
    prior.train_decoder = True
    prior.model = _StubFBA()
    prior.configure_trainability()
    frames = torch.rand(1, 2, 3, 16, 16)
    mask_logits = torch.ones(1, 2, 1, 16, 16)
    backbone = MAM2BackboneOutput(
        mask_logits=mask_logits,
        trimap_logits=torch.zeros(1, 2, 3, 16, 16),
        alpha_matte=mask_logits.sigmoid(),
        non_memory_features=torch.zeros(1, 2, 8, 8, 8),
    )
    output = prior(frames, backbone)
    assert output.shape == (1, 2, 16, 16, 16)
    assert prior.model.seen == (
        (2, 3, 64, 64),
        (2, 2, 64, 64),
        (2, 3, 64, 64),
        (2, 6, 64, 64),
    )
    expected_input = linear_to_srgb(
        F.interpolate(frames.flatten(0, 1), size=(64, 64), mode="bilinear", align_corners=False)
    )
    assert torch.allclose(prior.model.seen_image, expected_input)
    expected_features = prior.model.decoder.conv_up4[:-1](prior.model.seen_image)
    expected_features = F.interpolate(
        expected_features,
        size=(16, 16),
        mode="bilinear",
        align_corners=False,
    ).reshape(1, 2, 16, 16, 16)
    assert torch.allclose(output, expected_features)
    # Conventional FBA RGB/alpha fusion is downstream of this feature and
    # must not contaminate the prior supplied to PRISM's refractive operator.
    with torch.no_grad():
        prior.model.decoder.conv_up4[-1].weight.normal_(std=10.0)
        prior.model.decoder.conv_up4[-1].bias.fill_(100.0)
    assert torch.allclose(prior(frames, backbone), output)
    output.mean().backward()
    assert prior.model.decoder.conv_up4[2].weight.grad is not None
