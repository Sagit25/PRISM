"""Logging and camera parameter recording for multi-view rendering."""

import logging
import os
import os.path as osp
import glob
import json

import numpy as np
import cv2
import imageio

log = logging.getLogger(__name__)

from utils.camera_utils import fov_to_intrinsic_mat, get_extrinsic_matrix, normalize
from utils.video_utils import gen_video


def get_scale_matrix(scale):
    """Create a 4x4 scale matrix."""
    return np.diag([scale, scale, scale, 1])


class RenderLogger:
    """
    Logger for recording camera parameters and generating preview outputs.

    Saves camera intrinsics/extrinsics in multiple formats:
    - cameras_sphere.npz: NumPy format (IDR/NeuS compatible)
    - transforms.json: NeRF format
    - preview outputs: gif/video/image grids
    """

    def __init__(self, output_folder):
        self.log_file = open(os.path.join(output_folder, 'log.txt'), 'w')
        self.intri_mat = None
        self.fov_x = None
        self.res_w = None
        self.res_h = None
        self.scale_mat = np.diag([1, 1, 1, 1])
        self.extri_mat_list = []
        self.origin_list = []
        self.output_folder = output_folder
        self.image_filenames = []
        self.rotation_list = []

    def close(self):
        """Close the log file handle."""
        if self.log_file and not self.log_file.closed:
            self.log_file.close()

    def write(self, strs):
        """Write log message to file."""
        self.log_file.write('%s\n' % strs)
        self.log_file.flush()

    def set_scale_mat(self, scale):
        """Set scene scale matrix."""
        self.scale_mat = get_scale_matrix(scale)

    def set_intri_mat(self, fov, w, h):
        """Set camera intrinsic matrix from FOV and resolution."""
        self.fov_x = float(fov)
        self.res_w = int(w)
        self.res_h = int(h)
        self.intri_mat = fov_to_intrinsic_mat(fov, 'x', w, h)

    def add_extri_mat(self, target, origin, up, coord_type='opencv'):
        """Add camera extrinsic matrix for one frame."""
        target = np.array(target)
        origin = np.array(origin)
        up = normalize(np.array(up))
        extri_mat = get_extrinsic_matrix(target, origin, up, coord_type=coord_type)
        self.origin_list.append(origin)
        self.extri_mat_list.append(extri_mat)

    def add_image_filename(self, filename_basename: str):
        """Add image filename for frame alignment with extrinsics."""
        self.image_filenames.append(filename_basename)

    def add_rotation(self, rotation_radians: float):
        """Add rotation angle in radians for current frame."""
        self.rotation_list.append(rotation_radians)

    def save_mat(self):
        """Save camera parameters in multiple formats."""
        # Save cameras_sphere.npz (IDR/NeuS format)
        camera_dict = {}
        self.mat_path = os.path.join(self.output_folder, 'cameras_sphere.npz')
        camera_dict['img_num'] = len(self.extri_mat_list)
        for idx, extri in enumerate(self.extri_mat_list):
            camera_dict['rot_mat_{}'.format(idx)] = extri[:, 0:3]
            camera_dict['origin_{}'.format(idx)] = self.origin_list[idx]
            camera_dict['world_mat_{}'.format(idx)] = np.dot(self.intri_mat, extri)
            camera_dict['scale_mat_{}'.format(idx)] = self.scale_mat
        camera_dict['intrinsic_mat'] = self.intri_mat
        np.savez(self.mat_path, **camera_dict)

        # Save transforms.json (NeRF format)
        self._save_mipnerf_json()


    def _to_c2w(self, extri):
        """Convert 3x4 world-to-camera matrix to 4x4 camera-to-world matrix."""
        R = extri[:, :3]
        t = extri[:, 3:4]
        c2w = np.eye(4)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = (-R.T @ t).ravel()
        return c2w

    def _save_mipnerf_json(self):
        """Save camera parameters in MipNeRF/NeRF JSON format."""
        if self.intri_mat is None or len(self.extri_mat_list) == 0:
            return

        camera_angle_x = float(self.fov_x)

        frames = []
        names = [osp.join('image', n) for n in self.image_filenames]
        for i, extri in enumerate(self.extri_mat_list):
            c2w = self._to_c2w(extri)
            frame_data = {
                'file_path': names[i]
            }
            if i < len(self.rotation_list):
                frame_data['rotation'] = float(self.rotation_list[i])
            frame_data['transform_matrix'] = c2w.tolist()
            frames.append(frame_data)

        data = {
            'camera_angle_x': camera_angle_x,
            'frames': frames,
        }
        out_path = osp.join(self.output_folder, 'transforms.json')
        with open(out_path, 'w') as f:
            json.dump(data, f, indent=2)

    def gen_preview(self, read_path, scale=0.25, output='gif'):
        """
        Generate preview from rendered images.

        Args:
            read_path: Directory containing rendered images
            scale: Downscale factor for preview (default: 0.25)
            output: Output type - 'image' (grid), 'video' (MP4), or 'gif' (default)
        """
        img_path_list = sorted(
            glob.glob(os.path.join(read_path, '*.png')) +
            glob.glob(os.path.join(read_path, '*.jpg'))
        )
        imgs = []

        target_size = None
        resize_interp_down = cv2.INTER_AREA
        resize_interp_up = cv2.INTER_LINEAR

        for path in img_path_list:
            try:
                img = imageio.imread(path)
            except Exception:
                log.warning(f"The image {path} is not valid")
                continue

            img = np.asarray(img)
            if img.ndim == 2:
                img = np.repeat(img[:, :, None], 3, axis=2)

            if target_size is None:
                base_w, base_h = img.shape[1], img.shape[0]
                if scale is not None:
                    new_w = max(1, int(round(base_w * scale)))
                    new_h = max(1, int(round(base_h * scale)))
                else:
                    new_w, new_h = base_w, base_h
                target_size = (new_w, new_h)

            interp = resize_interp_down if img.shape[1] >= target_size[0] else resize_interp_up
            resized = cv2.resize(img, target_size, interpolation=interp)
            imgs.append(np.ascontiguousarray(resized))

        if output == 'image':
            if len(imgs) == 0:
                return

            def _make_image_grid(images, nrow=3):
                """Stack images into a grid for quick previews."""
                img_h, img_w, img_c = images[0].shape
                cols = max(1, int(nrow))
                rows = int(np.ceil(len(images) / cols))
                grid = np.zeros((img_h * rows, img_w * cols, img_c), dtype=images[0].dtype)
                for idx, image in enumerate(images):
                    row = idx // cols
                    col = idx % cols
                    y0, y1 = row * img_h, (row + 1) * img_h
                    x0, x1 = col * img_w, (col + 1) * img_w
                    grid[y0:y1, x0:x1] = image
                return grid

            preview_img = _make_image_grid(imgs, nrow=3)
            imageio.imwrite(osp.join(self.output_folder, 'preview.png'), preview_img)

        elif output == 'video':
            gen_video(imgs, osp.join(self.output_folder, 'preview.mp4'))

        elif output == 'gif':
            imageio.mimsave(osp.join(self.output_folder, 'preview.gif'), imgs, 'GIF', duration=0.3)


def export_nerf_transforms(logger, output_path, train_num=None, mixed_lighting=False,
                           n_skip=0, has_active_light=False):
    """
    Export camera parameters to NeRF-compatible JSON format.

    Args:
        logger: RenderLogger instance with intrinsic/extrinsic data
        output_path: Output directory for JSON files
        train_num: Number of training views (rest goes to validation).
                   If None, all frames go to training set.
        mixed_lighting: Whether to include has_active_light flag for mixed lighting
        n_skip: Every n_skip-th frame has active light turned off (for mixed_lighting)
        has_active_light: Whether scene has active light (shape_light)

    Output files:
        transforms_train.json (always created)
        transforms_val.json (if train_num specified and < total frames)
    """
    if logger.intri_mat is None or len(logger.extri_mat_list) == 0:
        return

    intrinsic_mat = logger.intri_mat[:3]
    W = int(2 * intrinsic_mat[0][2])
    focal = intrinsic_mat[0][0]
    camera_angle_x = float(np.arctan(W / (2 * focal)) * 2)

    total_frames = len(logger.extri_mat_list)
    if train_num is None:
        train_num = total_frames

    train_json = {
        'camera_angle_x': camera_angle_x,
        'frames': []
    }
    if has_active_light:
        train_json['mixed_lighting'] = mixed_lighting

    val_json = {
        'camera_angle_x': camera_angle_x,
        'frames': []
    }

    # Mitsuba to NeRF coordinate system transformation
    mitsuba2nerf = np.array([
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])

    for i, extri in enumerate(logger.extri_mat_list):
        rot_mat = extri[:, 0:3]
        origin = logger.origin_list[i]

        # Convert to C2W matrix
        c2w = np.eye(4)
        c2w[:3, :3] = rot_mat.T
        c2w[:3, 3] = origin
        c2w = c2w @ mitsuba2nerf

        is_train = i < train_num
        target_json = train_json if is_train else val_json
        idx = i if is_train else i - train_num
        folder = 'train' if is_train else 'val'
        path = osp.join(folder, str(idx).zfill(len(str(total_frames))))

        frame_content = {
            'file_path': path,
            'rotation': 0.0,
            'transform_matrix': c2w.tolist()
        }

        if is_train and mixed_lighting and has_active_light and n_skip > 0:
            frame_content['has_active_light'] = not ((i + 1) % n_skip == 0)

        target_json['frames'].append(frame_content)

    def default_encoder(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        return json.JSONEncoder().default(x)

    with open(osp.join(output_path, 'transforms_train.json'), 'w') as f:
        json.dump(train_json, f, indent=4, default=default_encoder)

    if train_num < total_frames:
        with open(osp.join(output_path, 'transforms_val.json'), 'w') as f:
            json.dump(val_json, f, indent=4, default=default_encoder)
