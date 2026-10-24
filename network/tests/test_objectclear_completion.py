from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from refractive_mam2.objectclear_completion import (
    FlowGuidedObjectClearCompleter,
    ObjectClearSettings,
)


class _FakeVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(scaling_factor=1.0)
        self.register_buffer("reference", torch.zeros(()))

    @property
    def dtype(self) -> torch.dtype:
        return self.reference.dtype

    def decode(self, latent, *, return_dict):
        assert not return_dict
        image = F.interpolate(latent[:, :3], scale_factor=8, mode="nearest")
        return (image,)

    def encode(self, pixels):
        latent = F.interpolate(pixels[:, :3], scale_factor=1 / 8, mode="area")
        latent = torch.cat((latent, torch.zeros_like(latent[:, :1])), dim=1)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latent))


class _FakeObjectClear:
    def __init__(self) -> None:
        self.vae = _FakeVAE()
        self.calls = 0
        self.output_types: list[str] = []
        self.input_images = []
        self.image_processor = SimpleNamespace(
            preprocess=lambda value: torch.zeros(1, 3, value.height, value.width)
        )

    def __call__(self, *, image, mask_image, output_type, generator, **kwargs):
        del mask_image, kwargs
        self.calls += 1
        self.output_types.append(output_type)
        self.input_images.append(image.copy())
        assert generator.initial_seed() == 42
        height, width = image.size[1], image.size[0]
        if output_type == "pil":
            from PIL import Image

            return SimpleNamespace(images=[Image.new("RGB", (width, height), (128, 128, 128))])
        return SimpleNamespace(images=torch.zeros(1, 4, height // 8, width // 8))


class _FakeDifferentiableObjectClear(_FakeObjectClear):
    def __init__(self) -> None:
        super().__init__()
        self.unet = nn.Conv2d(4, 4, 1, bias=False)
        with torch.no_grad():
            self.unet.weight.copy_(torch.eye(4).view(4, 4, 1, 1))
        self.components = {"unet": self.unet, "vae": self.vae}
        self.num_timesteps = 0

    @torch.no_grad()
    def __call__(
        self, *, image, num_inference_steps, callback_on_step_end,
        callback_on_step_end_tensor_inputs, output_type, **kwargs,
    ):
        del kwargs
        assert callback_on_step_end_tensor_inputs == ["latents"]
        assert output_type == "latent"
        self.calls += 1
        self.num_timesteps = num_inference_steps
        latent = torch.zeros(1, 4, image.height // 8, image.width // 8)
        for index in range(num_inference_steps):
            latent = self.unet(latent)
            latent = callback_on_step_end(
                self, index, index, {"latents": latent}
            )["latents"]
        return SimpleNamespace(images=latent)


def _inputs():
    canvas = torch.full((1, 3, 16, 16), 0.4)
    hole = torch.zeros((1, 1, 16, 16), dtype=torch.bool)
    hole[..., 4:12, 4:12] = True
    inverse = torch.full_like(canvas, 0.2)
    weight = torch.ones_like(hole, dtype=torch.float32)
    flow = torch.full((1, 2, 16, 16), 2.0, requires_grad=True)
    object_reference = torch.full_like(canvas, 0.8)
    return canvas, hole, inverse, weight, flow, object_reference


def test_objectclear_prior_is_cached_but_flow_adapter_is_trainable() -> None:
    prior = _FakeObjectClear()
    model = FlowGuidedObjectClearCompleter(
        prior,
        settings=ObjectClearSettings(min_working_side=16, cache_entries=2, use_fused_output=False),
    )
    canvas, hole, inverse, weight, flow, object_reference = _inputs()
    optimizer = torch.optim.SGD(model.flow_latent_adapter.parameters(), lr=0.1)
    for _ in range(2):
        optimizer.zero_grad()
        output = model(
            canvas, (~hole).float(), hole,
            flow_evidence=(inverse, weight, flow), object_reference=object_reference,
        )
        assert output.shape == canvas.shape
        output.mean().backward(retain_graph=True)
        optimizer.step()
    assert prior.calls == 1
    assert prior.output_types == ["latent"]
    assert model.image_cache_hits == 1
    assert model.image_cache_misses == 1
    assert model.flow_latent_adapter.body[-1].weight.grad is not None
    # The first update makes the zero-initialized residual responsive to u.
    assert flow.grad is not None and bool(flow.grad.abs().sum() > 0)


def test_objectclear_does_not_generate_when_no_background_hole() -> None:
    prior = _FakeObjectClear()
    model = FlowGuidedObjectClearCompleter(
        prior, settings=ObjectClearSettings(min_working_side=16, use_fused_output=False)
    )
    canvas, hole, inverse, weight, flow, object_reference = _inputs()
    hole.zero_()
    output = model(
        canvas, torch.ones_like(hole), hole,
        flow_evidence=(inverse, weight, flow), object_reference=object_reference,
    )
    torch.testing.assert_close(output, canvas)
    assert prior.calls == 0


def test_objectclear_default_uses_fused_output_before_flow_adaptation() -> None:
    prior = _FakeObjectClear()
    model = FlowGuidedObjectClearCompleter(
        prior, settings=ObjectClearSettings(min_working_side=16)
    )
    canvas, hole, inverse, weight, flow, object_reference = _inputs()
    output = model(
        canvas, (~hole).float(), hole,
        flow_evidence=(inverse, weight, flow), object_reference=object_reference,
    )
    assert output.shape == canvas.shape
    assert prior.output_types == ["pil"]
    image = prior.input_images[0]
    # ObjectClear's object encoder sees an actual target inside the hole,
    # while the input retains the observation canvas outside it.
    assert image.getpixel((8, 8))[0] > image.getpixel((1, 1))[0]
    output.mean().backward()
    assert model.flow_latent_adapter.body[-1].weight.grad is not None


def test_objectclear_rejects_blank_object_prompt() -> None:
    import pytest

    model = FlowGuidedObjectClearCompleter(
        _FakeObjectClear(), settings=ObjectClearSettings(min_working_side=16)
    )
    canvas, hole, inverse, weight, flow, _ = _inputs()
    with pytest.raises(ValueError, match="object-reference frame"):
        model(canvas, (~hole).float(), hole, flow_evidence=(inverse, weight, flow))


def test_objectclear_late_denoising_guides_generation_and_flow() -> None:
    prior = _FakeDifferentiableObjectClear()
    model = FlowGuidedObjectClearCompleter(
        prior,
        settings=ObjectClearSettings(
            min_working_side=16, inference_steps=3, late_denoise_steps=1,
            use_fused_output=False,
        ),
    )
    canvas, hole, inverse, weight, flow, object_reference = _inputs()
    optimizer = torch.optim.SGD(model.flow_latent_adapter.parameters(), lr=0.1)
    for _ in range(2):
        optimizer.zero_grad()
        flow.grad = None
        output = model(
            canvas, (~hole).float(), hole,
            flow_evidence=(inverse, weight, flow), object_reference=object_reference,
        )
        output.mean().backward()
        optimizer.step()
    assert prior.calls == 2  # Flow-conditioned diffusion cannot reuse a cached prior.
    assert model.image_cache_size == 0
    assert model.image_cache_misses == 2
    assert not any(parameter.requires_grad for parameter in prior.unet.parameters())
    assert model.flow_latent_adapter.body[-1].weight.grad.abs().sum() > 0
    assert flow.grad is not None and flow.grad.abs().sum() > 0


def test_objectclear_late_denoising_runs_during_inference_mode() -> None:
    prior = _FakeDifferentiableObjectClear()
    model = FlowGuidedObjectClearCompleter(
        prior,
        settings=ObjectClearSettings(
            min_working_side=16, inference_steps=3, late_denoise_steps=1,
            use_fused_output=False,
        ),
    ).eval()
    canvas, hole, inverse, weight, flow, object_reference = _inputs()
    with torch.inference_mode():
        output = model(
            canvas, (~hole).float(), hole,
            flow_evidence=(inverse, weight, flow),
            object_reference=object_reference,
        )
    assert output.shape == canvas.shape
    assert torch.isfinite(output).all()
    assert prior.calls == 1
