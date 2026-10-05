import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "tools" / "validate_prism_contract.py"
SPEC = importlib.util.spec_from_file_location("validate_prism_contract", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _sequence(tmp_path: Path, *, move_mask: bool) -> dict:
    prefix = tmp_path / "sample_seq0000"
    frames = [
        tmp_path / f"sample_seq0000_frame{index:04d}" for index in range(2)
    ]
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[10:22, 10:18] = 255
    moved_mask = np.zeros_like(mask)
    moved_mask[10:22, 16:24] = 255
    poses = [np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)]
    poses[1][0, 3] = 0.1
    for index, (frame, pose) in enumerate(zip(frames, poses)):
        np.save(str(frame) + "_object_pose.npy", pose)
        current_mask = moved_mask if move_mask and index == 1 else mask
        assert cv2.imwrite(str(frame) + "_object_mask.png", current_mask)
    return {"prefix": prefix, "frames": frames, "metadata": {}}


def test_temporal_validator_rejects_moving_pose_with_static_masks(tmp_path) -> None:
    with pytest.raises(AssertionError, match="acceleration structure is stale"):
        MODULE.validate_temporal_pose_alignment(
            _sequence(tmp_path, move_mask=False)
        )


def test_temporal_validator_accepts_masks_that_follow_motion(tmp_path) -> None:
    MODULE.validate_temporal_pose_alignment(_sequence(tmp_path, move_mask=True))
