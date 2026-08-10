"""Multi-view transparent object data with refraction correspondence tracing."""
import mitsuba as mi
from mitsuba import ScalarTransform4f as T

from projects.TransRecon import TransRecon
from scene_builder.mitsuba_utils import set_camera_pose

from utils.camera_utils import fov_to_intrinsic_mat, get_extrinsic_matrix, gen_rays
from utils.tracer_factory import build_tracer
from utils.tool_utils import linear_to_srgb

import os
import os.path as osp
import imageio.v2 as imageio
from tqdm import tqdm
import numpy as np
import cv2
from omegaconf import ListConfig

from utils.registry import PROJECT_REGISTRY

try:
    import flow_vis
    HAS_FLOW_VIS = True
except ImportError:
    HAS_FLOW_VIS = False


@PROJECT_REGISTRY.register()
class TransCorresRecon(TransRecon):
    """
    Extends TransRecon with ground-truth correspondence tracing.

    For each camera view, traces refraction rays through the transparent object
    to compute output directions and UV correspondences on the background.
    Saves per-view correspondence maps and aggregated numpy arrays.
    """

    @staticmethod
    def _get_scene_element(scene_conf, elem_type):
        elements = scene_conf['element']
        if not isinstance(elements, (list, ListConfig)):
            raise TypeError('Scene.element must be a list of element configs.')
        return next(elem for elem in elements if elem['type'] == elem_type)

    def __init__(self, conf):
        super().__init__(conf)

        # Load mesh and create tracer
        source_path = conf['Scene']['source_path']
        mesh_elem = self._get_scene_element(conf['Scene'], 'transparent_mesh')
        env_elem = self._get_scene_element(conf['Scene'], 'envmap_light')
        shape_filename = mesh_elem['mesh_filename']
        ior = mesh_elem.get('IoR', 1.5)

        import trimesh
        mesh = trimesh.load(osp.join(source_path, 'shape', shape_filename))
        self.tracer = build_tracer(mesh, conf, obj_ior=ior)

        # Create a separate scene for rendering background (env light only)
        envmap_filename = env_elem['envmap_filename']
        self.env_scene = mi.load_dict({
            'type': 'scene',
            'env_light': {
                'type': 'envmap',
                'filename': osp.join(source_path, 'env_map', envmap_filename),
                'to_world': T.rotate(axis=[1, 0, 0], angle=90)
            },
            'integrator': {
                'type': 'path',
                'max_depth': 1
            },
        })

    def setup_output_paths(self):
        super().setup_output_paths()
        # Correspondence output folders
        self.outdir_folder = osp.join(self.output_folder, 'out_dir')
        os.makedirs(self.outdir_folder, exist_ok=True)
        self.corres_map_folder = osp.join(self.output_folder, 'corres_map')
        os.makedirs(self.corres_map_folder, exist_ok=True)

    def _get_cam_params(self):
        """Extract camera intrinsic and film size from the sensor."""
        params = mi.traverse(self.camera)
        x_fov_param = params.get('x_fov', params['x_fov'])
        try:
            x_fov = float(x_fov_param[0])
        except (TypeError, IndexError):
            x_fov = float(x_fov_param)
        w, h = params['film.size']
        intri_mat = fov_to_intrinsic_mat(x_fov, 'x', w, h)
        return intri_mat, (w, h)

    def tracing_refraction(self, cam_intri_mat, cam_extri_mat, img_size):
        """Trace refraction correspondence for current camera pose."""
        R = cam_extri_mat[:, 0:3]
        t = cam_extri_mat[:, 3]
        w, h = img_size
        rays_o, rays_d = gen_rays(cam_intri_mat, R, t, w, h)
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        output = self.tracer.trace_out_dir(rays_o, rays_d)
        return output

    def direction_to_uvcoord(self, light_dirs, cam_intri_mat, cam_extri_mat, img_size):
        """Project output light directions to background UV coordinates."""
        R = cam_extri_mat[:, 0:3]
        camera_dirs = (R @ light_dirs.T).T

        pixel_coords = (cam_intri_mat @ camera_dirs.T).T
        pixel_coords /= (pixel_coords[:, 2:3] + 1e-12)

        w, h = img_size
        u = pixel_coords[:, 0] / w
        v = pixel_coords[:, 1] / h
        uv_coords = np.stack([u, v], axis=1)
        return uv_coords

    def fetch_correspondence(self, correspondence, img):
        """Fetch colors from image using UV correspondence map."""
        h, w, _ = img.shape
        u = correspondence[:, :, 0]
        v = correspondence[:, :, 1]
        u = (u * (w - 1)).astype(np.float32)
        v = (v * (h - 1)).astype(np.float32)
        color = cv2.remap(img, u, v, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return color

    def run(self):
        # First run parent (renders RGB + mask + normal, copies assets)
        super().run()

        # Now trace correspondence for each view
        cam_intri_mat, img_size = self._get_cam_params()
        w, h = img_size
        img_num = self.pose_generator.img_num

        # Pre-allocate arrays instead of accumulating in lists
        corres_all = np.zeros((img_num, h, w, 2), dtype=np.float32)
        outdir_all = np.zeros((img_num, h, w, 3), dtype=np.float32)
        valid_mask_all = np.zeros((img_num, h, w), dtype=bool)

        for idx, camera_pose in tqdm(enumerate(self.pose_generator.pose_list),
                                     total=img_num,
                                     desc="Tracing correspondence"):
            set_camera_pose(self.camera, **camera_pose)
            cam_extri_mat = get_extrinsic_matrix(**camera_pose)

            tracing_output = self.tracing_refraction(cam_intri_mat, cam_extri_mat, img_size)
            valid_mask = tracing_output['twice_mask']
            valid_mask = valid_mask.reshape(-1)

            corres = self.direction_to_uvcoord(
                tracing_output['out_dir'], cam_intri_mat, cam_extri_mat, img_size)
            corres[valid_mask == 0, :] = 0
            corres = corres.clip(0, 1)
            corres = corres.reshape(h, w, 2)
            corres_all[idx] = corres
            valid_mask = valid_mask.reshape(h, w)

            outdir_all[idx] = tracing_output['out_dir'].reshape(h, w, 3)
            valid_mask_all[idx] = valid_mask

            # Render background and create visualization
            background = mi.render(self.env_scene, sensor=self.camera)
            background_np = np.array(background[:, :, :3])
            background_srgb = np.clip(linear_to_srgb(background_np), 0, 1)

            refract_img = self.fetch_correspondence(corres, background_srgb)
            imageio.imwrite(osp.join(self.corres_map_folder, f'refract_{idx}.png'),
                            (255 * refract_img).clip(0, 255).astype(np.uint8))
            imageio.imwrite(osp.join(self.corres_map_folder, f'background_{idx}.png'),
                            (255 * background_srgb).clip(0, 255).astype(np.uint8))

            if HAS_FLOW_VIS:
                corresmap = flow_vis.flow_to_color(corres * 2 - 1)
                corresmap[valid_mask == 0, :] = 0
                imageio.imwrite(self.get_path(idx, 'corres_map'), corresmap)

        # Save aggregated numpy arrays
        np.save(osp.join(self.outdir_folder, 'out_dir.npy'), outdir_all)
        np.save(osp.join(self.outdir_folder, 'correspondence.npy'), corres_all)
        np.save(osp.join(self.outdir_folder, 'valid_mask.npy'), valid_mask_all)
