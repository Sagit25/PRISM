from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from refractive_mam2.official_mam2 import OfficialMAM2ClipBackbone


class _FakeMain(nn.Module):
    image_size = 16

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.mask_prompts = 0
        self.closed = 0

    def init_state(self, directory, **kwargs):
        del directory, kwargs
        owner = self

        class _Frames:
            def close(self):
                owner.closed += 1

        return {"images": _Frames()}

    def add_new_mask(self, state, frame, object_id, mask):
        del state, frame, object_id
        assert mask.shape == (8, 8)
        self.mask_prompts += 1

    def propagate_in_video(self, state, **kwargs):
        del state, kwargs
        for index in range(2):
            mask = torch.ones(1, 1, 8, 8) * self.weight.detach()
            trimap = torch.zeros(1, 3, 8, 8)
            trimap[:, 1] = 1
            yield index, [1], mask, trimap

    def _get_image_feature(self, state, frame, batch):
        del state, frame, batch
        return None, {"backbone_fpn": [torch.ones(1, 256, 2, 2)]}, None

    def reset_state(self, state):
        del state


class _FakeMatter(nn.Module):
    size_divisibility = 8

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("pixel_mean", torch.zeros(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.ones(1, 3, 1, 1))
        self.backbone = nn.Conv2d(4, 8, 3, padding=1)

        class _Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(8, 1, 1)

            def forward(self, features, inputs):
                del inputs
                return {"phas": self.conv(features).sigmoid()}

        self.decoder = _Decoder()


def test_released_mam2_clip_contract_and_trainable_flow_path() -> None:
    main = _FakeMain()
    model = OfficialMAM2ClipBackbone(main, _FakeMatter())
    normalized = torch.zeros(1, 2, 3, 16, 16)
    prompt = torch.ones(1, 1, 8, 8)
    result = model.forward_mam2_clip(
        normalized, first_frame_mask_inputs=prompt, original_size=(8, 8)
    )
    result.validate()
    assert result.mask_logits.shape == (1, 2, 1, 8, 8)
    assert result.trimap_logits.shape == (1, 2, 3, 8, 8)
    assert result.non_memory_features.shape == (1, 2, 256, 2, 2)
    assert main.mask_prompts == 1 and main.closed == 1

    trainable = model.configure_joint_trainable()
    assert trainable
    assert not any(parameter.requires_grad for parameter in main.parameters())
    assert all(parameter.requires_grad for parameter in model.mam2_matter.external_model.decoder.parameters())
    assert all(parameter.requires_grad for parameter in model.mam2_matter.flow_feature_modulator.parameters())

    frames = torch.full((1, 2, 3, 8, 8), 0.4)
    background = torch.full((1, 3, 8, 8), 0.2)
    flow = torch.full((1, 2, 2, 8, 8), 1.0, requires_grad=True)
    render_error = torch.full_like(frames, 0.1)
    support = torch.ones(1, 2, 1, 8, 8)
    optimizer = torch.optim.SGD(trainable, lr=0.1)
    for _ in range(2):
        optimizer.zero_grad()
        alpha = model.mam2_matter.forward_with_flow(
            frames, result.trimap_logits, background, flow, render_error, support
        )
        alpha.mean().backward(retain_graph=True)
        optimizer.step()
    assert flow.grad is not None and bool(flow.grad.abs().sum() > 0)


def test_released_mam2_checkpoint_contains_only_tunable_matter() -> None:
    model = OfficialMAM2ClipBackbone(_FakeMain(), _FakeMatter())
    state = model.mam2_extension_state_dict()
    assert state
    assert all(name.startswith("mam2_matter.") for name in state)
    assert not any("backbone" in name for name in state)
    model.load_mam2_extension_state_dict(state)
