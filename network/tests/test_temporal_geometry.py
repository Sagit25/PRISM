import numpy as np
import pytest

from refractive_mam2.temporal_geometry import object_motion_correspondence


def _plane_depth(height: int = 8, width: int = 8) -> np.ndarray:
    yy, xx = np.mgrid[:height, :width]
    focal = width / 2  # horizontal FOV = 90 degrees
    ray = np.stack(
        ((xx - width / 2) / focal, (yy - height / 2) / focal, np.ones_like(xx)),
        axis=-1,
    )
    return np.linalg.norm(ray, axis=-1).astype(np.float32)


def _metadata() -> dict:
    return {
        "camera_pose": {
            "origin": [0, 0, 0],
            "target": [0, 0, 1],
            "up": [0, -1, 0],
        },
        "camera_x_fov_deg": 90,
    }


def test_identity_pose_gives_zero_flow_on_visible_surface() -> None:
    depth = np.stack([_plane_depth()] * 2)
    poses = np.stack([np.eye(4)] * 2)
    flow, valid = object_motion_correspondence(
        depth, np.ones_like(depth), poses, _metadata()
    )
    assert valid[0, 3, 3] == 1
    np.testing.assert_allclose(flow[0, 3, 3], (0, 0), atol=1e-6)
    assert valid[0, 7, 7] == 0  # a complete bilinear target footprint is required


def test_pose_translation_projects_to_next_frame_and_rejects_occlusion() -> None:
    depth = np.stack([_plane_depth()] * 2)
    poses = np.stack([np.eye(4)] * 2)
    poses[1, 0, 3] = 0.1
    flow, valid = object_motion_correspondence(
        depth, np.ones_like(depth), poses, _metadata()
    )
    assert valid[0, 3, 3] == 1
    np.testing.assert_allclose(flow[0, 3, 3], (0.4, 0), atol=1e-6)

    depth[1, 3:5, 3:5] = 0.5  # another surface lies in front of this point
    _, occluded = object_motion_correspondence(
        depth, np.ones_like(depth), poses, _metadata()
    )
    assert occluded[0, 3, 3] == 0


def test_missing_target_depth_and_degenerate_camera_are_rejected() -> None:
    depth = np.stack([_plane_depth()] * 2)
    validity = np.ones_like(depth)
    validity[1, 3, 3] = 0
    poses = np.stack([np.eye(4)] * 2)
    _, support = object_motion_correspondence(depth, validity, poses, _metadata())
    assert support[0, 3, 3] == 0

    metadata = _metadata()
    metadata["camera_pose"]["up"] = [0, 0, 1]
    with pytest.raises(ValueError, match="parallel"):
        object_motion_correspondence(depth, np.ones_like(depth), poses, metadata)
