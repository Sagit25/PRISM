"""Camera-related utility functions for intrinsic/extrinsic matrix calculations."""

import numpy as np


def normalize(v):
    """Normalize a vector to unit length."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def fov_to_intrinsic_mat(fov, fov_axis, w, h):
    """
    Convert field of view to intrinsic camera matrix.

    Args:
        fov: Field of view in degrees
        fov_axis: 'x' for horizontal FOV, 'y' for vertical FOV
        w: Image width in pixels
        h: Image height in pixels

    Returns:
        3x3 intrinsic matrix
    """
    if fov_axis == 'x':
        df = w / 2 / np.tan(fov / 2 * np.pi / 180.0)
    else:
        df = h / 2 / np.tan(fov / 2 * np.pi / 180.0)
    intrinsic_matrix = np.array([[df, 0, w/2], [0, df, h/2], [0, 0, 1]])
    return intrinsic_matrix


def get_extrinsic_matrix(target, origin, up, coord_type='opencv'):
    """
    Calculate world-to-camera extrinsic matrix from camera pose.

    Reference: https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function

    Limitation: When new_z=[0,0,1] (camera looking straight down) and up=[0,0,1],
    new_z and up are identical, causing cross product failure.

    Args:
        target: Look-at target point (3D array)
        origin: Camera position (3D array)
        up: Up vector (3D array)
        coord_type: Coordinate system ('opencv', 'mitsuba', or 'blender')

    Returns:
        3x4 extrinsic matrix [R | t]

    Note: This extrinsic matrix is calculated separately and not used in Mitsuba rendering.
    """
    target = np.array(target)
    origin = np.array(origin)
    up = normalize(np.array(up))
    new_z = normalize(target - origin)

    if (new_z == up).all() or (new_z == -up).all():
        raise ValueError('new_z and up can not be the same')

    if coord_type == 'opencv':
        new_x = normalize(np.cross(new_z, up))
    elif coord_type == 'mitsuba':
        new_x = normalize(np.cross(up, new_z))
    elif coord_type == 'blender':
        new_z = -new_z
        new_x = normalize(np.cross(up, new_z))
    else:
        raise ValueError('coord_type must be "opencv", "mitsuba" or "blender"')

    new_y = normalize(np.cross(new_z, new_x))

    extrinsic = np.zeros([3, 4])
    R = np.zeros([3, 3])
    R[0, 0:3] = new_x
    R[1, 0:3] = new_y
    R[2, 0:3] = new_z
    t = -np.dot(R, origin)
    extrinsic[:, 0:3] = R
    extrinsic[:, 3] = t

    return extrinsic


def gen_rays(camera_K, camera_R, camera_t, W, H):
    """
    Generate rays at world space from one camera.

    Args:
        camera_K: 3x3 intrinsic matrix
        camera_R: 3x3 rotation matrix (world-to-camera)
        camera_t: 3D translation vector
        W: Image width
        H: Image height

    Returns:
        rays_o: (H, W, 3) ray origins
        rays_v: (H, W, 3) ray directions
    """
    u = np.linspace(0, W - 1, W)
    v = np.linspace(0, H - 1, H)
    pixels_u, pixels_v = np.meshgrid(u, v, indexing='xy')
    p = np.stack([pixels_u, pixels_v, np.ones_like(pixels_v)], axis=-1)  # H, W, 3
    intrinsic_inv = np.linalg.inv(camera_K)
    p = np.matmul(intrinsic_inv[None, None, :, :], p[:, :, :, None]).squeeze()
    rays_v = p / np.linalg.norm(p, ord=2, axis=-1).reshape(H, W, 1)

    c2w_R = np.linalg.inv(camera_R)
    rays_v = np.matmul(c2w_R[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()
    rays_origin = -np.matmul(c2w_R, camera_t)  # 3
    rays_o = np.tile(rays_origin, [H, W, 1])
    return rays_o, rays_v
