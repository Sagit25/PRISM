"""Occlusion-aware primary-surface correspondences for synthetic PRISM clips.

These are *object-motion* correspondences, not refractive-flow labels.  They
must not be used to force transmittance or refractive displacement to stay
constant as the viewing angle changes.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _camera_frame(metadata: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    pose = metadata.get("camera_pose")
    if not isinstance(pose, dict):
        raise ValueError("temporal correspondences require camera_pose metadata")
    origin = np.asarray(pose.get("origin"), dtype=np.float64)
    target = np.asarray(pose.get("target"), dtype=np.float64)
    up = np.asarray(pose.get("up", (0, 0, 1)), dtype=np.float64)
    if any(value.shape != (3,) or not np.isfinite(value).all() for value in (origin, target, up)):
        raise ValueError("camera origin, target and up must be finite 3-vectors")
    forward = target - origin
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-8:
        raise ValueError("camera origin and target must differ")
    forward /= forward_norm
    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-8:
        raise ValueError("camera up must not be parallel to view direction")
    right /= right_norm
    down = np.cross(forward, right)
    world_to_camera = np.stack((right, down, forward))
    fov = float(metadata.get("camera_x_fov_deg", 0))
    if not np.isfinite(fov) or not 0 < fov < 180:
        raise ValueError("temporal correspondences require a valid camera_x_fov_deg")
    return origin, world_to_camera, fov


def object_motion_correspondence(
    depth: np.ndarray,
    depth_validity: np.ndarray,
    object_poses: np.ndarray,
    metadata: dict[str, Any],
    *,
    relative_depth_tolerance: float = 0.02,
    absolute_depth_tolerance: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Return t->t+1 pixel flow and valid surface visibility [T-1,H,W].

    The camera/ray convention matches the dataset generator's OpenCV-style
    ``gen_rays``: principal point (W/2,H/2), pixel locations (x,y), and
    horizontal focal length derived from x-FOV.  A target is valid only if all
    bilinear depth/validity contributors belong to the object and their depth
    agrees with the reprojected surface point.  This rejects occlusions and
    silhouette mixtures rather than hallucinating temporal supervision.
    """

    depth = np.asarray(depth, dtype=np.float64)
    valid_depth = np.asarray(depth_validity) >= 0.5
    poses = np.asarray(object_poses, dtype=np.float64)
    if depth.ndim != 3 or valid_depth.shape != depth.shape:
        raise ValueError("depth and depth_validity must be [T,H,W]")
    count, height, width = depth.shape
    if height < 2 or width < 2:
        raise ValueError("temporal correspondence requires at least 2x2 pixels")
    if poses.shape != (count, 4, 4):
        raise ValueError("object_poses must be [T,4,4]")
    if not np.isfinite(poses).all() or not np.isfinite(depth).all():
        raise ValueError("object depth and poses must be finite")
    if relative_depth_tolerance < 0 or absolute_depth_tolerance < 0:
        raise ValueError("depth tolerances must be non-negative")
    flow = np.zeros((max(count - 1, 0), height, width, 2), np.float32)
    support = np.zeros((max(count - 1, 0), height, width), np.float32)
    if count < 2:
        return flow, support

    origin, world_to_camera, fov = _camera_frame(metadata)
    focal = width / (2 * np.tan(np.deg2rad(fov) / 2))
    yy, xx = np.mgrid[:height, :width].astype(np.float64)
    camera_ray = np.stack(((xx - width / 2) / focal, (yy - height / 2) / focal,
                           np.ones_like(xx)), axis=-1)
    camera_ray /= np.linalg.norm(camera_ray, axis=-1, keepdims=True)
    world_ray = camera_ray @ world_to_camera
    xy = np.stack((xx, yy), axis=-1)
    for index in range(count - 1):
        source_valid = valid_depth[index] & (depth[index] > 0)
        if not source_valid.any():
            continue
        world_point = origin + world_ray * depth[index, ..., None]
        relative_pose = poses[index + 1] @ np.linalg.inv(poses[index])
        moving_point = world_point @ relative_pose[:3, :3].T + relative_pose[:3, 3]
        next_camera = (moving_point - origin) @ world_to_camera.T
        z = next_camera[..., 2]
        safe_z = np.where(z > 1e-6, z, 1.0)
        projected_x = focal * next_camera[..., 0] / safe_z + width / 2
        projected_y = focal * next_camera[..., 1] / safe_z + height / 2
        x0 = np.floor(projected_x).astype(np.int64)
        y0 = np.floor(projected_y).astype(np.int64)
        interior = (
            source_valid & (z > 1e-6) &
            (x0 >= 0) & (y0 >= 0) & (x0 + 1 < width) & (y0 + 1 < height)
        )
        if not interior.any():
            continue
        x0 = x0.clip(0, width - 2)
        y0 = y0.clip(0, height - 2)
        x1, y1 = x0 + 1, y0 + 1
        next_valid = valid_depth[index + 1]
        four_valid = (
            next_valid[y0, x0] & next_valid[y0, x1] &
            next_valid[y1, x0] & next_valid[y1, x1]
        )
        wx = np.clip(projected_x - x0, 0, 1)
        wy = np.clip(projected_y - y0, 0, 1)
        next_depth = depth[index + 1]
        sampled_depth = (
            (1 - wx) * (1 - wy) * next_depth[y0, x0]
            + wx * (1 - wy) * next_depth[y0, x1]
            + (1 - wx) * wy * next_depth[y1, x0]
            + wx * wy * next_depth[y1, x1]
        )
        expected_depth = np.linalg.norm(moving_point - origin, axis=-1)
        tolerance = np.maximum(absolute_depth_tolerance,
                               relative_depth_tolerance * expected_depth)
        visible = (
            interior & four_valid & np.isfinite(sampled_depth) &
            (np.abs(sampled_depth - expected_depth) <= tolerance)
        )
        support[index] = visible.astype(np.float32)
        flow[index] = np.where(
            visible[..., None],
            np.stack((projected_x, projected_y), axis=-1) - xy,
            0.0,
        ).astype(np.float32)
    return flow, support
