"""Batch dataset generation for transparent object reconstruction training."""
import mitsuba as mi
from mitsuba import ScalarTransform4f as T
import drjit as dr

from projects.base import BaseProject
from scene_builder.build_utils import build_scene, build_pose_generator
from scene_builder.mitsuba_utils import set_camera_pose
from utils.camera_utils import fov_to_intrinsic_mat, get_extrinsic_matrix, gen_rays
from utils.tracer_factory import build_tracer
from utils.tool_utils import convert_to_dict, linear_to_srgb

import gc
import os
import os.path as osp
import imageio.v2 as imageio
from tqdm import tqdm
import numpy as np
import cv2
import trimesh

from utils.registry import PROJECT_REGISTRY

try:
    import flow_vis
    HAS_FLOW_VIS = True
except ImportError:
    HAS_FLOW_VIS = False


@PROJECT_REGISTRY.register()
class RefractiveCorresDataset(BaseProject):
    """
    Batch dataset generation for transparent object reconstruction.

    Iterates through shapes x environments x cameras, rendering images
    with refraction correspondence tracing. Supports checkpoint/resume.

    Config options:
        ior_range (list): Index of refraction sampling range [min, max]
        envnum_per_shape (int): Number of environment maps per shape
        camnum_per_env (int): Number of camera views per environment
    """

    def __init__(self, conf):
        self.raw_output_folder = None
        self.conf = conf
        self.preview_mode = conf.get('preview', False)
        self.resource_folder = conf['Scene']['source_path']

        self.load_resource_info()
        self.bootstrap_shape_path, self.bootstrap_envmap_path = self._select_bootstrap_assets()

        # Build scene and camera
        scene_conf = convert_to_dict(conf['Scene'])
        scene_conf = self._patch_scene_bootstrap(scene_conf)
        # Add AOV integrator for normal output
        init_integrator = scene_conf.get('integrator', {'type': 'path', 'max_depth': 20})
        scene_conf['integrator'] = {
            'type': 'aov',
            'aovs': 'nn:sh_normal',
            'my_image': init_integrator,
        }
        conf_copy = dict(conf)
        conf_copy['Scene'] = scene_conf
        self.scene = build_scene(scene_conf)

        camera_conf = convert_to_dict(conf['Camera'])
        if self.preview_mode:
            camera_conf['film']['height'] = max(1, camera_conf['film']['height'] // 4)
            camera_conf['film']['width'] = max(1, camera_conf['film']['width'] // 4)
            camera_conf['sampler']['sample_count'] = 128
        self.camera = mi.load_dict(camera_conf)

        # Build camera pose generator (RandomCamPose for on-the-fly sampling)
        campose_conf = convert_to_dict(conf['CamPose'])
        self.pose_generator = build_pose_generator(campose_conf)

        self.setup_output_paths()

        self.envnum_per_shape = conf['envnum_per_shape']
        self.camnum_per_env = conf['camnum_per_env']
        self.ior_range = list(conf.get('ior_range', [1.3, 1.6]))

        # Create a separate scene for rendering background
        self.env_scene = mi.load_dict({
            'type': 'scene',
            'env_light': {
                'type': 'envmap',
                'filename': osp.join(self.resource_folder, 'env_map', self.bootstrap_envmap_path),
                'to_world': T.rotate(axis=[1, 0, 0], angle=90)
            },
            'integrator': {
                'type': 'path',
                'max_depth': 1
            },
        })

        # Cache traverse params to avoid repeated allocation
        self._scene_params = mi.traverse(self.scene.scene)
        self._env_params = mi.traverse(self.env_scene)
        self._cam_params = mi.traverse(self.camera)

    def setup_output_paths(self, split='train'):
        output_root = self.conf.get('output_folder', './result')
        project_name = self.conf.get('project_name', 'dataset')
        base_folder = osp.join(output_root, project_name)
        os.makedirs(base_folder, exist_ok=True)

        if self.raw_output_folder is None:
            self.raw_output_folder = base_folder

        self.info_file = osp.join(self.raw_output_folder, '{}_file.txt'.format(split))
        self.output_folder = osp.join(self.raw_output_folder, split)
        os.makedirs(self.output_folder, exist_ok=True)

    def load_resource_info(self):
        """Load lists of shape and environment map filenames."""
        with open(osp.join(self.resource_folder, 'shape', 'train_shape.txt'), 'r') as f:
            self.train_shape_list = [line for line in f.read().rstrip().split('\n') if line]
        with open(osp.join(self.resource_folder, 'shape', 'test_shape.txt'), 'r') as f:
            self.test_shape_list = [line for line in f.read().rstrip().split('\n') if line]
        with open(osp.join(self.resource_folder, 'env_map', 'train_env.txt'), 'r') as f:
            self.train_envmap_list = [line for line in f.read().rstrip().split('\n') if line]
        with open(osp.join(self.resource_folder, 'env_map', 'test_env.txt'), 'r') as f:
            self.test_envmap_list = [line for line in f.read().rstrip().split('\n') if line]

    def _first_available(self, primary_list, fallback_list, kind):
        if primary_list:
            return primary_list[0]
        if fallback_list:
            return fallback_list[0]
        raise FileNotFoundError(f'No {kind} entries found in resource index files under {self.resource_folder}')

    def _require_resource_file(self, subdir, rel_path, kind):
        abs_path = osp.join(self.resource_folder, subdir, rel_path)
        if not osp.exists(abs_path):
            raise FileNotFoundError(f'Bootstrap {kind} file not found: {abs_path}')
        return rel_path

    def _select_bootstrap_assets(self):
        shape_path = self._first_available(self.train_shape_list, self.test_shape_list, 'shape')
        envmap_path = self._first_available(self.train_envmap_list, self.test_envmap_list, 'envmap')
        shape_path = self._require_resource_file('shape', shape_path, 'shape')
        envmap_path = self._require_resource_file('env_map', envmap_path, 'envmap')
        return shape_path, envmap_path

    def _patch_scene_bootstrap(self, scene_conf):
        elements = scene_conf['element']
        for elem in elements:
            if elem['type'] == 'transparent_mesh':
                elem['mesh_filename'] = self.bootstrap_shape_path
            elif elem['type'] == 'envmap_light':
                elem['envmap_filename'] = self.bootstrap_envmap_path
        return scene_conf

    def update_mesh(self, shape_path, ior):
        """Update the scene mesh and IOR via mi.traverse()."""
        if hasattr(self, 'tracer'):
            del self.tracer

        mesh = trimesh.load(osp.join(self.resource_folder, 'shape', shape_path))
        self.tracer = build_tracer(mesh, self.conf, obj_ior=ior)

        self._scene_params['transparent_mesh.bsdf.eta'] = ior
        self._scene_params['transparent_mesh.vertex_positions'] = dr.ravel(mi.Point3f(np.array(mesh.vertices, dtype=np.float32).T))
        self._scene_params['transparent_mesh.vertex_normals'] = dr.ravel(mi.Point3f(np.array(mesh.vertex_normals, dtype=np.float32).T))
        self._scene_params['transparent_mesh.faces'] = dr.ravel(mi.Point3u(np.array(mesh.faces, dtype=np.uint32).T))
        self._scene_params.update()

    def update_envmap(self, envmap_path):
        """Update the environment map via mi.traverse()."""
        env = cv2.imread(
            osp.join(self.resource_folder, 'env_map', envmap_path),
            cv2.IMREAD_UNCHANGED)[:, :, :3]
        env_rgb = env[:, :, ::-1]
        self._scene_params['env_light.data'] = env_rgb
        self._scene_params.update()

        self._env_params['env_light.data'] = env_rgb
        self._env_params.update()

    def _random_cam_pose(self):
        """Sample a random camera pose and compute camera matrices."""
        self.cam_pose = self.pose_generator.random_sample()
        set_camera_pose(self.camera, **self.cam_pose)
        x_fov_param = self._cam_params.get('x_fov', self._cam_params['x_fov'])
        try:
            self.cam_Xfov = float(x_fov_param[0])
        except (TypeError, IndexError):
            self.cam_Xfov = float(x_fov_param)
        size = self._cam_params['film.size']
        w = size[0]
        h = size[1]
        self.img_size = (w, h)
        self.cam_intri_mat = fov_to_intrinsic_mat(self.cam_Xfov, 'x', w, h)
        self.cam_extri_mat = get_extrinsic_matrix(**self.cam_pose)

    def tracing_refraction(self):
        """Trace refraction correspondence for current camera."""
        R = self.cam_extri_mat[:, 0:3]
        t = self.cam_extri_mat[:, 3]
        w = self.img_size[0]
        h = self.img_size[1]
        rays_o, rays_d = gen_rays(self.cam_intri_mat, R, t, w, h)
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        output = self.tracer.trace_out_dir(rays_o, rays_d)
        return output

    def direction_to_uvcoord(self, light_dirs):
        """Project output light directions to background UV coordinates."""
        R = self.cam_extri_mat[:, 0:3]
        camera_dirs = (R @ light_dirs.T).T
        pixel_coords = (self.cam_intri_mat @ camera_dirs.T).T
        pixel_coords /= (pixel_coords[:, 2:3] + 1e-12)

        w, h = self.img_size
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

    def single_render(self):
        """Render a single view with correspondence tracing."""
        image = mi.render(self.scene.scene, sensor=self.camera)
        image_np = np.array(image)
        del image
        rgb_np = image_np[:, :, :3]
        rgb_srgb = np.clip(linear_to_srgb(rgb_np), 0, 1)
        normal = image_np[:, :, 3:6]
        obj_mask = ~(normal.sum(-1) == 0)

        tracing_output = self.tracing_refraction()

        background = mi.render(self.env_scene, sensor=self.camera)
        background_np = np.array(background[:, :, :3])
        del background
        background_srgb = np.clip(linear_to_srgb(background_np), 0, 1)
        valid_mask = tracing_output['twice_mask']

        corres = self.direction_to_uvcoord(tracing_output['out_dir'])

        # Filter out rays outside the imaging plane
        inside_mask = (corres[:, 0] > 0) & (corres[:, 0] < 1) & (corres[:, 1] > 0) & (corres[:, 1] < 1)
        valid_mask = valid_mask & inside_mask

        corres[~valid_mask, :] = 0
        corres = corres.clip(0, 1)
        corres = corres.reshape(self.img_size[1], self.img_size[0], 2)
        valid_mask = valid_mask.reshape(self.img_size[1], self.img_size[0])

        refract_img = self.fetch_correspondence(corres, background_srgb)
        refract_img[~valid_mask] = 0

        residual = rgb_srgb - refract_img
        residual = residual.clip(0, 1)
        residual[~obj_mask] = 0

        output = {
            'input_image': rgb_np,
            'obj_mask': obj_mask,
            'background': background_np,
            'correspondence': corres,
            'valid_mask': valid_mask.astype(np.uint8),
            'refract_img': refract_img,
            'residual': residual,
            'obj_normal': normal,
        }
        return output

    def write_flo_file(self, flow, filename):
        """Write optical flow in .flo format."""
        H, W, _ = flow.shape
        with open(filename, 'wb') as f:
            np.array([202021.25], dtype=np.float32).tofile(f)
            np.array([W, H], dtype=np.int32).tofile(f)
            flow.astype(np.float32).tofile(f)

    def single_save(self, render_result, basename):
        """Save all outputs from a single render."""
        image_path = osp.join(self.output_folder, basename + '_image.png')
        obj_mask_path = osp.join(self.output_folder, basename + '_obj_mask.png')
        valid_mask_path = osp.join(self.output_folder, basename + '_valid_mask.png')
        residual_path = osp.join(self.output_folder, basename + '_residual.png')
        background_path = osp.join(self.output_folder, basename + '_background.png')
        corres_path = osp.join(self.output_folder, basename + '_correspondence.flo')
        refract_path = osp.join(self.output_folder, basename + '_refract.png')
        normal_map_path = osp.join(self.output_folder, basename + '_normal.png')
        flow_map_path = osp.join(self.output_folder, basename + '_corres_map.png')

        mi.util.write_bitmap(image_path, render_result['input_image'])
        mi.util.write_bitmap(background_path, render_result['background'])
        imageio.imwrite(obj_mask_path, (255 * render_result['obj_mask']).astype(np.uint8))
        imageio.imwrite(valid_mask_path, (255 * render_result['valid_mask']).astype(np.uint8))
        imageio.imwrite(residual_path, (255 * render_result['residual']).clip(0, 255).astype(np.uint8))
        imageio.imwrite(refract_path, (render_result['refract_img'] * 255).clip(0, 255).astype(np.uint8))
        self.write_flo_file(render_result['correspondence'], corres_path)
        imageio.imwrite(normal_map_path,
                        (255 * (render_result['obj_normal'] + 1) / 2).clip(0, 255).astype(np.uint8))
        if HAS_FLOW_VIS:
            flow_map = flow_vis.flow_to_color(render_result['correspondence'] * 2 - 1)
            flow_map[render_result['valid_mask'] == 0, :] = 0
            imageio.imwrite(flow_map_path, flow_map)

    def _run_split(self, split, shape_list, envmap_list):
        """Run dataset generation for one split (train or test)."""
        self.setup_output_paths(split)

        # Load already-rendered entries for skip-based resume
        if osp.exists(self.info_file):
            with open(self.info_file, 'r') as f:
                existing = set(line.strip() for line in f if line.strip())
        else:
            existing = set()
        total_samples = len(shape_list) * self.envnum_per_shape * self.camnum_per_env
        remaining_samples = max(0, total_samples - len(existing))
        print('Split: {} | {} already rendered / {} planned samples | {} remaining'.format(
            split, len(existing), total_samples, remaining_samples))

        f = open(self.info_file, 'a')
        pbar = tqdm(total=total_samples, initial=min(len(existing), total_samples), desc=f'Rendering {split}')
        rendered_now = 0
        skipped_now = 0
        for shape_iter in range(len(shape_list)):
            shape_path = shape_list[shape_iter]
            shape_name = shape_path.split('/')[-1].split('.')[0]
            mesh_loaded = False
            for env_iter in range(self.envnum_per_shape):
                env_path = np.random.choice(envmap_list)
                envmap_name = env_path.split('/')[-1].split('.')[0]
                env_loaded = False
                for cam_iter in range(self.camnum_per_env):
                    basename = '{}_{}_cam{}'.format(shape_name, envmap_name, cam_iter)
                    if basename in existing:
                        skipped_now += 1
                        pbar.update(1)
                        continue

                    # Lazy load: only load mesh/envmap when we actually need to render
                    if not mesh_loaded:
                        ior = np.random.uniform(self.ior_range[0], self.ior_range[1])
                        self.update_mesh(shape_path, ior)
                        mesh_loaded = True
                    if not env_loaded:
                        self.update_envmap(env_path)
                        env_loaded = True

                    self._random_cam_pose()
                    render_result = self.single_render()
                    self.single_save(render_result, basename)
                    del render_result
                    f.write('{}\n'.format(basename))
                    f.flush()
                    existing.add(basename)
                    rendered_now += 1
                    pbar.update(1)

            # Free memory after each shape
            gc.collect()
            dr.flush_malloc_cache()
            dr.flush_kernel_cache()
        pbar.close()
        f.close()
        print('Split: {} finished | rendered {} new samples | skipped {} existing samples'.format(
            split, rendered_now, skipped_now))

    def run(self):
        self._run_split('train', self.train_shape_list, self.train_envmap_list)
        self._run_split('test', self.test_shape_list, self.test_envmap_list)
