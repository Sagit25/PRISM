from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest
from torch import nn
from types import SimpleNamespace

from refractive_mam2 import (
    BackgroundConfig, MaskedTemporalBackground, MatterConfig, PipelineConfig,
    RefractiveMAM2,
)
from refractive_mam2.objectclear_completion import (
    FlowGuidedObjectClearCompleter, ObjectClearSettings,
)
from refractive_mam2.official_sam2matting import OfficialSAM2MattingClipBackbone
from refractive_mam2.runner import (
    load_refractive_checkpoint, save_refractive_checkpoint,
)
from refractive_mam2.train import _operator_head_gradient_metrics
from refractive_mam2.training import configure_stage4


class _FakeReleasedSAM2Matting(nn.Module):
    image_size = 16

    def __init__(self) -> None:
        super().__init__()
        self.tracker = nn.Linear(1, 1)
        for name in (
            "unknown_region_predictor", "unknown_fusion",
            "unknown_alpha_predictor", "alpha_pred1", "alpha_pred2",
        ):
            setattr(self, name, nn.Identity())
        self.alpha_pred3 = nn.Conv2d(256, 1, 1)
        self.prompts = 0

    def init_state(self, video_path, **kwargs):
        del video_path
        assert kwargs["offload_video_to_cpu"] is False
        assert kwargs["offload_state_to_cpu"] is False
        return {"images": torch.ones(2, 3, 16, 16)}

    def add_new_mask(self, state, frame, object_id, mask):
        del state, frame, object_id
        assert mask.shape in ((8, 8), (16, 16))
        self.prompts += 1

    def propagate_in_video(self, state):
        del state
        for index in range(2):
            yield index, [1], torch.ones(1, 1, 8, 8), torch.full((1, 8, 8), 0.5), None

    def _get_image_feature(self, state, frame, batch):
        del state, frame, batch
        return None, {"backbone_fpn": [
            torch.ones(1, 32, 4, 4),
            torch.ones(1, 64, 2, 2),
            torch.ones(1, 256, 1, 1),
        ]}, None

    def _forward_alpha_heads(self, *, mask_inputs, high_res_features, image):
        del mask_inputs, image
        return self.alpha_pred3(high_res_features[-1]).sigmoid(), None, None

    def reset_state(self, state):
        del state


def test_released_sam2matting_contract_and_flow_gradient() -> None:
    official = _FakeReleasedSAM2Matting()
    model = OfficialSAM2MattingClipBackbone(official)
    normalized = torch.zeros(1, 2, 3, 16, 16)
    result = model.forward_mam2_clip(
        normalized,
        first_frame_mask_inputs=torch.ones(1, 1, 8, 8),
        original_size=(8, 8),
    )
    result.validate()
    assert official.prompts == 1
    assert result.mask_logits.shape == (1, 2, 1, 8, 8)
    assert result.alpha_matte.shape == (1, 2, 1, 8, 8)
    assert result.non_memory_features.shape == (1, 2, 256, 1, 1)
    assert result.trimap_logits.shape == (1, 2, 3, 8, 8)

    trainable = model.configure_joint_trainable()
    assert trainable
    assert not any(value.requires_grad for value in official.tracker.parameters())
    assert all(value.requires_grad for value in official.alpha_pred3.parameters())
    optimizer = torch.optim.SGD(trainable, lr=0.1)
    flow = torch.ones(1, 2, 2, 8, 8, requires_grad=True)
    frames = torch.ones(1, 2, 3, 8, 8) * 0.4
    background = torch.ones(1, 3, 8, 8) * 0.3
    for _ in range(2):
        optimizer.zero_grad()
        alpha = model.mam2_matter.forward_with_flow(
            frames, result.trimap_logits, background, flow,
            torch.ones_like(frames) * 0.1,
            torch.ones(1, 2, 1, 8, 8),
        )
        alpha.mean().backward()
        optimizer.step()
    assert flow.grad is not None and bool(flow.grad.abs().sum() > 0)
    assert all(key.startswith(("main.alpha_pred3.", "mam2_matter.")) for key in model.mam2_extension_state_dict())


def test_trainable_alpha_head_accepts_inference_tracker_cache() -> None:
    """Stage 4 must not pass inference tensors into trainable alpha convolutions."""

    official = _FakeReleasedSAM2Matting()
    model = OfficialSAM2MattingClipBackbone(official)
    model.configure_joint_trainable()
    with torch.inference_mode():
        image = torch.ones(1, 3, 8, 8)
        mask = torch.ones(1, 1, 4, 4)
        features = [
            torch.ones(1, 32, 4, 4),
            torch.ones(1, 64, 2, 2),
            torch.ones(1, 256, 1, 1),
        ]
    model.mam2_matter.set_cached_inputs([(image, mask, features)])
    cached_image, cached_mask, cached_features = model.mam2_matter._cached_inputs[0]
    assert not any(value.is_inference() for value in (
        cached_image, cached_mask, *cached_features,
    ))

    with torch.no_grad():
        model.mam2_matter.flow_feature_modulator[-1].weight.fill_(1e-4)

    flow = torch.ones(1, 1, 2, 8, 8, requires_grad=True)
    frames = torch.ones(1, 1, 3, 8, 8)
    alpha = model.mam2_matter.forward_with_flow(
        frames, torch.zeros(1, 1, 3, 8, 8),
        torch.zeros(1, 3, 8, 8), flow,
        torch.ones_like(frames), torch.ones(1, 1, 1, 8, 8),
    )
    alpha.mean().backward()
    assert official.alpha_pred3.weight.grad is not None
    assert flow.grad is not None and bool(flow.grad.abs().sum() > 0)


def test_released_sam2matting_requires_source_and_checkpoint() -> None:
    import pytest

    with pytest.raises(FileNotFoundError):
        OfficialSAM2MattingClipBackbone.from_pretrained(
            "/nonexistent/released/source", "/nonexistent/checkpoint"
        )


class _FakeVAE(nn.Module):
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


class _FakeObjectClear:
    def __init__(self):
        self.vae = _FakeVAE()
        self.calls = 0

    def __call__(self, *, image, output_type, **kwargs):
        del kwargs
        assert output_type == "latent"
        self.calls += 1
        return SimpleNamespace(images=torch.zeros(1, 4, image.size[1] // 8, image.size[0] // 8))


class _FakeLateObjectClear(_FakeObjectClear):
    def __init__(self):
        super().__init__()
        self.unet = nn.Conv2d(4, 4, 1, bias=False)
        self.components = {"unet": self.unet, "vae": self.vae}
        self.num_timesteps = 0

    @torch.no_grad()
    def __call__(
        self, *, image, output_type, num_inference_steps,
        callback_on_step_end, callback_on_step_end_tensor_inputs, **kwargs,
    ):
        del kwargs
        assert output_type == "latent"
        assert callback_on_step_end_tensor_inputs == ["latents"]
        self.calls += 1
        self.num_timesteps = num_inference_steps
        latent = torch.zeros(1, 4, image.size[1] // 8, image.size[0] // 8)
        for index in range(num_inference_steps):
            latent = self.unet(latent)
            latent = callback_on_step_end(self, index, index, {"latents": latent})["latents"]
        return SimpleNamespace(images=latent)


@pytest.mark.parametrize("late_steps", [0, 1])
def test_sam2matting_and_objectclear_share_prism_flow_gradient(tmp_path, late_steps) -> None:
    torch.manual_seed(19)
    predictor = OfficialSAM2MattingClipBackbone(_FakeReleasedSAM2Matting())
    normalized = torch.zeros(1, 2, 3, 16, 16)
    semantic = predictor.forward_mam2_clip(
        normalized, first_frame_mask_inputs=torch.ones(1, 1, 16, 16),
        original_size=(16, 16),
    )
    prior = _FakeLateObjectClear() if late_steps else _FakeObjectClear()
    completion = FlowGuidedObjectClearCompleter(
        prior, settings=ObjectClearSettings(
            min_working_side=16, use_fused_output=False,
            inference_steps=3, late_denoise_steps=late_steps,
        ),
    )
    config = PipelineConfig(
        background=BackgroundConfig(
            evidence_flux_only=True, completion_variant="diffusion",
            inverse_min_reliable_weight=0.0,
        ),
        matter=MatterConfig(feature_channels=256, width=16),
        joint_refinement_steps=1,
    )
    model = RefractiveMAM2(
        predictor, config,
        background_model=MaskedTemporalBackground(
            config.background, diffusion_completion=completion,
        ),
    )
    trainable = configure_stage4(predictor, model)
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    frames = torch.rand(1, 2, 3, 16, 16)
    output = model.forward_from_backbone(frames, semantic)
    assert prior.calls == 1
    assert bool(output.background.true_hole.any())
    # Verify the *same* physical reconstruction objective reaches both
    # trainable released-model branches, not separate auxiliary losses.
    loss = (output.reconstructed_frames - frames).square().mean()
    loss.backward()
    assert torch.count_nonzero(predictor.mam2_matter.flow_feature_modulator[-1].weight.grad) > 0
    assert torch.count_nonzero(completion.flow_latent_adapter.body[-1].weight.grad) > 0
    assert torch.count_nonzero(model.matter.head.weight.grad) > 0
    assert torch.count_nonzero(model.matter.head.weight.grad[4:6]) > 0
    gradient_metrics = _operator_head_gradient_metrics(model)
    assert gradient_metrics["train/matte_flow_adapter_grad"] > 0
    assert gradient_metrics["train/background_flow_adapter_grad"] > 0
    to_update = {
        "alpha": predictor.main.alpha_pred3.weight,
        "matte_flow": predictor.mam2_matter.flow_feature_modulator[-1].weight,
        "background_flow": completion.flow_latent_adapter.body[-1].weight,
        "prism_flow": model.matter.head.weight,
    }
    trainable_ids = {id(parameter) for parameter in trainable}
    assert all(id(parameter) in trainable_ids for parameter in to_update.values())
    before = {name: parameter.detach().clone() for name, parameter in to_update.items()}
    optimizer.step()
    for name, parameter in to_update.items():
        updated = parameter.detach()
        original = before[name]
        if name == "prism_flow":
            updated, original = updated[4:6], original[4:6]
        assert (updated - original).abs().sum() > 0, name
    checkpoint_path = tmp_path / "joint.pt"
    save_refractive_checkpoint(checkpoint_path, predictor, model)
    payload = torch.load(checkpoint_path, weights_only=True)
    assert any(key.startswith("main.alpha_pred3.") for key in payload["predictor_mam2"])
    assert any(
        key.startswith("background_model.diffusion_completion.flow_latent_adapter.")
        for key in payload["physics_pipeline"]
    )
    expected = predictor.mam2_matter.flow_feature_modulator[-1].weight.detach().clone()
    with torch.no_grad():
        predictor.mam2_matter.flow_feature_modulator[-1].weight.zero_()
    load_refractive_checkpoint(checkpoint_path, predictor, model)
    torch.testing.assert_close(predictor.mam2_matter.flow_feature_modulator[-1].weight, expected)
