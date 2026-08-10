"""Single-pose camera generators."""
import numpy as np

from utils.tool_utils import spherical2cartesian
from utils.registry import CAMPOSE_REGISTRY


@CAMPOSE_REGISTRY.register()
class SinglePose:
    """Single camera pose defined either explicitly or via spherical angles."""

    def __init__(self, conf):
        self.conf = conf
        self._pose = self._build_pose()
        self.pose_list = [self._pose]
        self.img_num = 1

    def _build_pose(self):
        if 'theta' in self.conf:
            theta = self.conf.get('theta',45.0)
            phi = self.conf.get('phi',0.0)
            radius = self.conf.get('r',1.0)
            shift = np.array(self.conf.get('shift',[0.0, 0.0, 0.0]), dtype=np.float32)
            origin = spherical2cartesian(theta, phi, radius) + shift
        else:
            origin = np.array(self.conf.get('origin',[0.0, 0.0, 0.0]), dtype=np.float32)

        target = self.conf.get('target',[0.0, 0.0, 0.0])
        up = self.conf.get('up',[0.0, 0.0, 1.0])
        return {
            'origin': origin,
            'target': target,
            'up': up,
        }
