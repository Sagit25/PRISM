"""Random camera pose sampling (on-the-fly, not pre-generated)."""
import numpy as np

from utils.tool_utils import spherical2cartesian
from utils.registry import CAMPOSE_REGISTRY


@CAMPOSE_REGISTRY.register()
class RandomCamPose:
    """
    Random camera pose sampler for batch dataset generation.

    Unlike other CamPose classes, this does NOT pre-generate a pose list.
    Instead, call random_sample() to get a single random pose on the fly.
    """

    def __init__(self, conf):
        self.conf = conf
        self.theta_range = conf.get('theta_range', [20, 70])
        self.r_range = conf.get('r_range', [1.0, 1.2])
        self.shift = np.array(conf.get('shift', [0, 0, 0]))
        self.target = conf.get('target', [0, 0, 0])
        # No pre-generated pose list; img_num is not fixed
        self.pose_list = []
        self.img_num = 0

    def random_sample(self):
        """Sample a single random camera pose."""
        r = np.random.uniform(self.r_range[0], self.r_range[1])
        # Uniform sampling on the spherical cap
        theta_range_cos = [np.cos(np.deg2rad(self.theta_range[1])),
                           np.cos(np.deg2rad(self.theta_range[0]))]
        v = np.random.uniform(*theta_range_cos)
        theta = np.rad2deg(np.arccos(v))
        phi = np.random.uniform(0, 360)
        x, y, z = spherical2cartesian(theta, phi, r)
        position = np.array([x, y, z]) + self.shift
        return {
            'origin': position,
            'target': self.target,
            'up': [0, 0, 1],
        }

    def random_multi_sample(self, sample_num):
        """Sample multiple random camera poses."""
        return [self.random_sample() for _ in range(sample_num)]
