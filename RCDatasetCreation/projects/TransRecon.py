"""Multi-view transparent object data generation with mask and normal."""
import mitsuba as mi

from projects.MultiView import MultiView
from scene_builder.mitsuba_utils import set_camera_pose

import os
import os.path as osp
import imageio.v2 as imageio
from tqdm import tqdm
import numpy as np
import shutil
from omegaconf import ListConfig

from utils.registry import PROJECT_REGISTRY


@PROJECT_REGISTRY.register()
class TransRecon(MultiView):
    """
    Render multi-view images of a transparent object, then generate
    ground-truth mask and normal maps by re-rendering with AOV integrator
    (environment light removed).

    Config options (in addition to MultiView):
        gen_maskandnormal (bool): Whether to generate mask and normal (default: True)
    """

    def __init__(self, conf):
        # Force gen_normal and gen_mask for the main rendering pass
        conf['gen_normal'] = conf.get('gen_normal', True)
        conf['gen_mask'] = conf.get('gen_mask', True)
        super().__init__(conf)

    def run(self):
        # Render multi-view RGB images (and normal/mask via AOV if configured)
        super().run()

        # Copy mesh and envmap to output folder for reference
        self._copy_assets()

    def _copy_assets(self):
        """Copy the source mesh and environment map to output folder."""
        source_path = self.conf['Scene']['source_path']
        elements = self.conf['Scene']['element']
        if not isinstance(elements, (list, ListConfig)):
            raise TypeError('Scene.element must be a list of element configs.')
        mesh_elem = next(elem for elem in elements if elem['type'] == 'transparent_mesh')
        env_elem = next(elem for elem in elements if elem['type'] == 'envmap_light')

        # Copy mesh
        mesh_filename = mesh_elem['mesh_filename']
        suffix = mesh_filename.split('.')[-1]
        mesh_path = osp.join(source_path, 'shape', mesh_filename)
        if osp.exists(mesh_path):
            shutil.copy(mesh_path, osp.join(self.output_folder, 'gt.{}'.format(suffix)))

        # Copy envmap
        envmap_filename = env_elem['envmap_filename']
        envmap_path = osp.join(source_path, 'env_map', envmap_filename)
        suffix = envmap_filename.split('.')[-1]
        if osp.exists(envmap_path):
            shutil.copy(envmap_path, osp.join(self.output_folder, 'envmap.{}'.format(suffix)))
