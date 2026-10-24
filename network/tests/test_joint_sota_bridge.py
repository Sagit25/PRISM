"""Exercise both released-model adapters inside PRISM's actual joint graph."""

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from refractive_mam2 import (
    BackgroundConfig, MAM2BackboneOutput, MaskedTemporalBackground,
    MatterConfig, PipelineConfig, RefractiveMAM2,
)
from refractive_mam2.objectclear_completion import (
    FlowGuidedObjectClearCompleter, ObjectClearSettings,
)
from refractive_mam2.official_mam2 import OfficialMAM2Matter


class _Matte(nn.Module):
    size_divisibility = 8

    def __init__(self):
        super().__init__()
        self.register_buffer("pixel_mean", torch.zeros(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.ones(1, 3, 1, 1))
        self.backbone = nn.Conv2d(4, 8, 3, padding=1)

        class Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(8, 1, 1)

            def forward(self, feature, image):
                del image
                return {"phas": self.conv(feature).sigmoid()}

        self.decoder = Decoder()


class _VideoBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.mam2_matter = OfficialMAM2Matter(_Matte())

    def forward(self, frames, prompts=None):
        del prompts
        b, t, _, h, w = frames.shape
        mask = frames.new_full((b, t, 1, h, w), -8.0)
        mask[..., 4:12, 4:12] = 8.0
        trimap = frames.new_full((b, t, 3, h, w), -8.0)
        trimap[:, :, 0] = 8.0
        trimap[:, :, 0, 4:12, 4:12] = -8.0
        trimap[:, :, 1, 4:12, 4:12] = 8.0
        alpha = self.mam2_matter(frames, trimap)
        feature = frames.new_zeros((b, t, 8, h // 2, w // 2))
        return MAM2BackboneOutput(mask, trimap, alpha, feature)


class _VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(scaling_factor=1.0)
        self.register_buffer("zero", torch.zeros(()))

    @property
    def dtype(self):
        return self.zero.dtype

    def decode(self, latent, *, return_dict):
        assert not return_dict
        return (F.interpolate(latent[:, :3], scale_factor=8, mode="nearest"),)


class _ObjectClear:
    def __init__(self):
        self.vae = _VAE()
        self.calls = 0

    def __call__(self, *, image, output_type, **kwargs):
        del kwargs
        assert output_type == "latent"
        self.calls += 1
        return SimpleNamespace(images=torch.zeros(1, 4, image.size[1] // 8, image.size[0] // 8))


def test_flow_couples_both_pretrained_model_adapters_in_joint_graph() -> None:
    torch.manual_seed(13)
    prior = _ObjectClear()
    completer = FlowGuidedObjectClearCompleter(
        prior, settings=ObjectClearSettings(min_working_side=16, use_fused_output=False)
    )
    config = PipelineConfig(
        background=BackgroundConfig(
            evidence_flux_only=True, completion_variant="diffusion",
            inverse_min_reliable_weight=0.0,
        ),
        matter=MatterConfig(feature_channels=8, width=16),
        joint_refinement_steps=1,
    )
    backbone = _VideoBackbone()
    model = RefractiveMAM2(
        backbone, config,
        background_model=MaskedTemporalBackground(config.background, diffusion_completion=completer),
    )
    frames = torch.rand(1, 2, 3, 16, 16)
    output = model(frames)
    assert prior.calls == 1
    assert output.background.true_hole.any()
    # One physical render objective must reach both trainable branches.
    objective = (output.reconstructed_frames - frames).square().mean()
    objective.backward()
    matte_grad = backbone.mam2_matter.flow_feature_modulator[-1].weight.grad
    background_grad = completer.flow_latent_adapter.body[-1].weight.grad
    flow_grad = model.matter.head.weight.grad
    assert matte_grad is not None and torch.count_nonzero(matte_grad) > 0
    assert background_grad is not None and torch.count_nonzero(background_grad) > 0
    assert flow_grad is not None and torch.count_nonzero(flow_grad) > 0
