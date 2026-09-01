import inspect

import pytest

pytest.importorskip("torch")
from refractive_mam2.vendor import activate_vendored_sam2

activate_vendored_sam2()
sam2_predictor = pytest.importorskip("sam2.sam2_video_predictor")


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
