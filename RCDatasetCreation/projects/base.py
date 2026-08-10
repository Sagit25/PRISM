"""Base class for all projects."""
import os
import os.path as osp
from abc import ABC, abstractmethod

import mitsuba as mi

from scene_builder.build_utils import build_pose_generator, build_scene
from utils.tool_utils import convert_to_dict


class BaseProject(ABC):
    """Shared plumbing for loading scenes, cameras, and output folders."""

    def __init__(self, conf):
        self.conf = conf
        self.preview = conf.get('preview', False)
        self.scene = build_scene(conf['Scene'])
        self.pose_generator = build_pose_generator(conf['CamPose'])
        camera_conf = convert_to_dict(conf['Camera'])
        self.camera = mi.load_dict(camera_conf)
        if self.preview:
            self._apply_preview()
        self.setup_output_paths()

    def _apply_preview(self):
        params = mi.traverse(self.camera)
        width, height = params['film.size']
        params['film.size'] = [max(1, width // 2), max(1, height // 2)]
        if 'sampler.sample_count' in params:
            params['sampler.sample_count'] = max(1, params['sampler.sample_count'] // 4)
        params.update()

    def setup_output_paths(self):
        output_root = self.conf.get('output_folder', './result')
        project_name = self.conf.get('project_name', self.conf.get('Project', 'project'))
        self.output_folder = osp.join(output_root, project_name)
        os.makedirs(self.output_folder, exist_ok=True)

    def get_path(self, idx, folder='image'):
        """Generate zero-padded file path for multi-image outputs."""
        digit = max(2, len(str(self.img_num - 1))) if hasattr(self, 'img_num') else 4
        filename = str(idx).zfill(digit) + ".png"
        return os.path.join(self.output_folder, folder, filename)

    def make_folders(self, names):
        """Create subdirectories under output_folder."""
        for name in names:
            os.makedirs(os.path.join(self.output_folder, name), exist_ok=True)

    @abstractmethod
    def run(self):
        raise NotImplementedError
