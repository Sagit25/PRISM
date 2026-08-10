"""Multi-view and video rendering projects."""

import logging
import os
import os.path as osp

import mitsuba as mi
import numpy as np
import imageio.v2 as imageio
from tqdm import tqdm

logger = logging.getLogger(__name__)

from projects.base import BaseProject
from utils.logger import RenderLogger
from scene_builder.mitsuba_utils import set_camera_pose, wrap_with_aov, swap_base_color, restore_base_color
from utils.video_utils import gen_video_from_folder
from utils.registry import PROJECT_REGISTRY
from utils.tool_utils import convert_to_dict


@PROJECT_REGISTRY.register()
class MultiView(BaseProject):
    """
    Multi-view rendering project for static scenes.

    Renders RGB images from multiple camera poses, with optional normal/depth/mask outputs.
    Records camera parameters in multiple formats (npz, json) for downstream reconstruction tasks.

    Config options:
        gen_normal (bool): Generate normal maps (default: False)
        gen_depth (bool): Generate depth maps (default: False)
        record_camera (bool): Record camera parameters (default: auto - True if <200 images)
        log_extri_mat_type (str): Coordinate system for extrinsics ('opencv', 'mitsuba', 'blender')
    """

    def __init__(self, conf):
        # Read configuration options before parent init
        self.gen_normal = conf.get('gen_normal', False)
        self.gen_depth = conf.get('gen_depth', False)
        self.gen_albedo = conf.get('gen_albedo', False)
        self.gen_roughness = conf.get('gen_roughness', False)
        self.gen_metallic = conf.get('gen_metallic', False)
        self.gen_mask = self.gen_normal or self.gen_depth
        self._needs_material_pass = self.gen_roughness or self.gen_metallic
        self._needs_albedo_aov = self.gen_albedo or self._needs_material_pass
        self.has_aov = self.gen_normal or self.gen_depth or self._needs_albedo_aov
        self.export_video = conf.get('export_video', False)
        self.video_fps = conf.get('video_fps', 60)
        if 'record_camera' in conf:
            self.record_camera = conf['record_camera']
        else:
            self.record_camera = not self.export_video
        self.log_extri_mat_type = conf.get('log_extri_mat_type', 'opencv')

        # Modify integrator configuration to support AOV if needed
        if self.has_aov:
            self._configure_aov_integrator(conf)

        # Initialize parent class
        super().__init__(conf)

        # Read scale from Scene conf (moved from Scene class to MultiViewProject)
        self.scale = conf['Scene'].get('scale', 1.0)

        # Initialize logger
        self.logger = RenderLogger(self.output_folder)

        # Record camera information if enabled
        if self.record_camera:
            self.record_info()

    def _configure_aov_integrator(self, conf):
        """Modify integrator configuration to add AOV support for normal/depth."""
        init_conf = convert_to_dict(conf['Scene']['integrator'])
        conf['Scene']['integrator'] = wrap_with_aov(
            init_conf,
            normal=self.gen_normal,
            depth=self.gen_depth,
            albedo=self._needs_albedo_aov,
        )

    def record_info(self):
        """Record camera intrinsic and extrinsic parameters."""
        # Set scale matrix
        self.logger.set_scale_mat(self.scale)

        # Get camera intrinsics
        params = mi.traverse(self.camera)
        # Extract x_fov - handle both array and scalar types
        x_fov_param = params.get('x_fov', params['x_fov'])
        try:
            x_fov = float(x_fov_param[0])
        except (TypeError, IndexError):
            x_fov = float(x_fov_param)
        w, h = params['film.size']
        self.logger.set_intri_mat(x_fov, w, h)

        # Write scene information
        with open(osp.join(self.output_folder, 'scene.txt'), 'w') as file:
            print(self.scene.scene, file=file)

        # Record all camera extrinsics
        for idx, pose in enumerate(self.pose_generator.pose_list):
            filename = os.path.basename(self.get_path(idx))
            self.logger.add_image_filename(filename)
            self.logger.add_extri_mat(**pose, coord_type=self.log_extri_mat_type)
            self.logger.write('{} origin: {}  lookat: {}'.format(
                filename, pose['origin'], pose['target']))

        # Save camera parameters
        self.logger.save_mat()

    def setup_output_paths(self):
        """Configure output paths for images and optional outputs."""
        super().setup_output_paths()
        self.img_num = self.pose_generator.img_num

        # Create output folders
        folders = ['image']
        if self.gen_normal:
            folders.append('normal')
        if self.gen_depth:
            folders.append('depth')
        if self.gen_albedo:
            folders.append('albedo')
        if self.gen_roughness:
            folders.append('roughness')
        if self.gen_metallic:
            folders.append('metallic')
        if self.gen_mask:
            folders.append('mask')
        self.make_folders(folders)

    def process_extra_output(self, image, idx):
        """
        Process and save AOV outputs (normal, depth, mask).

        Args:
            image: Rendered image with AOV channels
            idx: Frame index

        Returns:
            Tuple of (normal_img, mask, depth_img, albedo_img)
        """
        normal_img, depth_img, albedo_img, mask = None, None, None, None
        channel_idx = 3

        # Extract and process normal map (channels 3:6)
        if self.gen_normal:
            normal = np.array(image[:, :, channel_idx:channel_idx + 3])
            channel_idx += 3

            # Transform normal from world space to camera space
            extrin_mat = self.logger.extri_mat_list[idx]
            rot_mat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]) @ extrin_mat[:, 0:3]
            H, W = normal.shape[:2]
            normal = (rot_mat[None, None, :, :] @ normal[:, :, :, None]).reshape(H, W, 3)

            # Normalize to [0,1] for saving
            normal_img = (normal + 1) / 2
            mask = np.linalg.norm(normal, axis=2) > 0.1
            normal_img[~mask, :] = 1

            # Save normal map (linear data, skip sRGB gamma)
            imageio.imwrite(self.get_path(idx, 'normal'),
                            (np.clip(normal_img, 0, 1) * 255).astype(np.uint8))

        # Extract and process depth map (channel 6)
        if self.gen_depth:
            depth = np.array(image[:, :, channel_idx])
            channel_idx += 1
            depth_max = np.max(depth)
            if depth_max > 0:
                depth_img = depth / depth_max  # Normalize to [0,1]
            else:
                depth_img = depth
            # Save depth map (linear data, skip sRGB gamma)
            imageio.imwrite(self.get_path(idx, 'depth'),
                            (np.clip(depth_img, 0, 1) * 255).astype(np.uint8))

        if self.gen_albedo:
            albedo_img = np.array(image[:, :, channel_idx:channel_idx + 3])
            mi.util.write_bitmap(self.get_path(idx, 'albedo'), albedo_img)

        # Save mask
        if self.gen_mask and mask is not None:
            mi.util.write_bitmap(self.get_path(idx, 'mask'), mask.astype(np.uint8) * 255)

        return normal_img, mask, depth_img, albedo_img

    def run(self):
        """Main rendering loop for multi-view project."""
        for idx, pose in enumerate(tqdm(self.pose_generator.pose_list, desc="Rendering frames")):
            # Set camera pose
            set_camera_pose(self.camera, **pose)

            # Render image
            image = mi.render(self.scene.scene, sensor=self.camera)

            # Process and save outputs
            if self.has_aov:
                # Save RGB (first 3 channels)
                rgb = image[:, :, :3]
                mi.util.write_bitmap(self.get_path(idx), rgb)
                # Process and save AOV outputs
                self.process_extra_output(image, idx)
            else:
                # Save RGB directly
                mi.util.write_bitmap(self.get_path(idx), image)

        # Render material property maps via base_color substitution
        if self._needs_material_pass:
            self._render_material_maps()

        # Generate preview if camera recording is enabled
        if self.record_camera:
            image_folder = os.path.join(self.output_folder, 'image')
            self.logger.gen_preview(image_folder, scale=0.25, output='gif')

        if self.export_video:
            image_folder = os.path.join(self.output_folder, 'image')
            video_path = os.path.join(self.output_folder, 'video.mp4')
            gen_video_from_folder(image_folder, video_path, fps=self.video_fps)
            logger.info(f"Video saved to: {video_path}")

    def _extract_albedo_channels(self, image):
        """Extract albedo channels from AOV image."""
        ch = 3  # skip RGB
        if self.gen_normal:
            ch += 3
        if self.gen_depth:
            ch += 1
        return np.array(image[:, :, ch:ch + 3])

    def _render_material_maps(self):
        """Render material property maps via base_color substitution."""
        props = []
        if self.gen_roughness:
            props.append(('roughness', 'roughness'))
        if self.gen_metallic:
            props.append(('metallic', 'metallic'))

        for prop_name, folder in props:
            originals = swap_base_color(self.scene.scene, prop_name)
            if not originals:
                logger.warning(f"No {prop_name} textures found, skipping")
                continue

            for idx, pose in enumerate(tqdm(self.pose_generator.pose_list, desc=f"Rendering {prop_name}")):
                set_camera_pose(self.camera, **pose)
                # Use low spp for texture readback (no noise in albedo AOV)
                image = mi.render(self.scene.scene, sensor=self.camera, spp=4)
                prop_img = self._extract_albedo_channels(image)
                # Save material map (linear data, skip sRGB gamma)
                imageio.imwrite(self.get_path(idx, folder),
                                (np.clip(prop_img, 0, 1) * 255).astype(np.uint8))

            restore_base_color(self.scene.scene, originals)
