import inspect

import pytest

torch = pytest.importorskip("torch")
from refractive_mam2.vendor import activate_vendored_sam2

activate_vendored_sam2()
sam2_predictor = pytest.importorskip("sam2.sam2_video_predictor")

from refractive_mam2 import MAM2VideoPredictor, SAM2IntegrationConfig


def test_supported_official_track_step_contract() -> None:
    signature = inspect.signature(sam2_predictor.SAM2VideoPredictor._track_step)
    required = {
        "current_vision_feats",
        "feat_sizes",
        "point_inputs",
        "mask_inputs",
        "output_dict",
        "prev_sam_mask_logits",
    }
    assert required.issubset(signature.parameters)


class _RecordingMatter(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, frames, trimap_logits):
        self.calls += 1
        return frames.new_zeros((*frames.shape[:2], 1, *frames.shape[-2:]))


class _TinyCheckpointedPredictor(MAM2VideoPredictor):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)
        self.image_size = 4
        self.num_feature_levels = 1
        self.encoder = torch.nn.Conv2d(3, 4, 1)
        self.mam2_matter = _RecordingMatter()
        self.mam2_integration_config = SAM2IntegrationConfig(
            temporal_activation_checkpointing=True,
            temporal_checkpoint_chunk_size=1,
            temporal_detach_interval=1,
        )

    def forward_image(self, frames):
        features = self.encoder(frames)
        return {
            "backbone_fpn": [features],
            "vision_pos_enc": [torch.zeros_like(features)],
        }

    def track_step(self, **kwargs):
        feature = kwargs["current_vision_feats"][0]
        height, width = kwargs["feat_sizes"][0]
        image = feature.permute(1, 2, 0).reshape(feature.shape[1], 4, height, width)
        mask = image[:, :1]
        return {
            "mam2_mask_logits": mask,
            "mam2_trimap_logits": torch.cat((-mask, mask * 0, mask), dim=1),
            "mam2_non_memory_features": image,
            "maskmem_features": image,
            "obj_ptr": image.mean(dim=(2, 3)),
        }

    @staticmethod
    def denormalize_sam2_frames(frames):
        return frames


def test_stage1a_clip_skips_alpha_and_checkpoints_each_time_step() -> None:
    predictor = _TinyCheckpointedPredictor().train()
    calls = 0
    original = predictor.forward_image

    def counted(frames):
        nonlocal calls
        calls += 1
        return original(frames)

    predictor.forward_image = counted
    frames = torch.rand(1, 4, 3, 4, 4)
    first_mask = torch.ones(1, 1, 4, 4)

    output = predictor.forward_mam2_clip(
        frames,
        first_frame_mask_inputs=first_mask,
        compute_alpha=False,
    )
    output.mask_logits.mean().backward()

    assert output.alpha_matte.shape == (1, 4, 1, 1, 1)
    assert predictor.mam2_matter.calls == 0
    assert calls >= 4
    assert predictor.encoder.weight.grad is not None
