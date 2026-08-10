"""Sphere-based multi-view camera sampling."""
import numpy as np

from utils.tool_utils import spherical2cartesian
from utils.registry import CAMPOSE_REGISTRY


@CAMPOSE_REGISTRY.register()
class SphericalGridPose:
    """Spherical camera trajectory with theta/phi grid sampling."""

    def __init__(self, conf):
        self.conf = conf
        self.theta_range = conf.get('theta_range', [0, 90])
        self.theta_num = conf.get('theta_num', 3)
        self.phi_range = conf.get('phi_range', [0, 360])
        self.phi_num = conf.get('phi_num', 8)
        self.r = conf.get('r', 3.0)
        self.target = conf.get('target', [0, 0, 0])
        self.up = conf.get('up', [0, 0, 1])
        self.shift = np.array(conf.get('shift', [0, 0, 0]), dtype=np.float32)

        # Generate pose list
        self.pose_list = self._sphere_sample()
        self.img_num = len(self.pose_list)

    def _sphere_sample(self):
        """Generate camera poses on a spherical grid."""
        # Generate theta and phi samples
        theta_samples = np.linspace(self.theta_range[0], self.theta_range[1], self.theta_num)
        phi_samples = np.linspace(self.phi_range[0], self.phi_range[1], self.phi_num)

        # If phi covers full circle [0, 360], remove last sample to avoid duplication
        if self.phi_range[0] == 0 and self.phi_range[1] == 360 and self.phi_num > 1:
            phi_samples = phi_samples[:-1]

        pose_list = []
        for theta in theta_samples:
            for phi in phi_samples:
                # Convert spherical to Cartesian coordinates
                x, y, z = spherical2cartesian(theta, phi, self.r)
                origin = np.array([x, y, z], dtype=np.float32) + self.shift

                pose = {
                    'origin': origin,
                    'target': self.target,
                    'up': self.up,
                }
                pose_list.append(pose)

        return pose_list


@CAMPOSE_REGISTRY.register()
class SphericalRandomPose:
    def __init__(self, conf):
        self.conf = conf
        self.theta_range = conf.get('theta_range', [20, 70])
        self.sample_num = conf.get('sample_num', 6)
        self.r = conf.get('r', 1.0)
        self.shift = np.array(self.conf.get('shift', [0, 0, 0]))
        self.target = conf.get('target', [0, 0, 0])

        self.pose_list = self.sphere_uniform_random_sample()
        self.img_num = len(self.pose_list)

    def sphere_uniform_random_sample(self):
        theta = np.random.uniform(*self.theta_range, self.sample_num)
        phi = np.random.uniform(0, 360, self.sample_num)
        x,y,z = spherical2cartesian(theta, phi, self.r)
        positions = np.stack([x,y,z],axis=1)
        positions += self.shift[np.newaxis]

        return [{
            'origin': p,
            'target': self.target,
            'up': [0,0,1]
        } for p in positions]


@CAMPOSE_REGISTRY.register()
class SphericalFibonacciPose:
    def __init__(self, conf):
        self.conf = conf
        self.theta_range = conf.get('theta_range', [20, 70])
        self.sample_num = conf.get('sample_num', 6)
        self.r = conf.get('r', 1.0)
        self.shift = np.array(self.conf.get('shift', [0, 0, 0]))
        self.target = conf.get('target', [0, 0, 0])

        self.pose_list = self.sphere_uniform_sample()
        self.img_num = len(self.pose_list)

    def sphere_uniform_sample(self):

        # fibonacci lattice uniform sampling
        golden_angle = np.pi * (3. - np.sqrt(5.))

        #* fast version
        cos_theta = np.linspace(np.cos(np.deg2rad(self.theta_range[1])), np.cos(np.deg2rad(self.theta_range[0])), self.sample_num)
        sin_theta = np.sqrt(1 - cos_theta**2)
        phi = np.arange(self.sample_num) * golden_angle
        x = np.cos(phi) * sin_theta
        y = np.sin(phi) * sin_theta
        z = cos_theta
        positions = np.stack([x,y,z],axis=1) * self.r

        pose_list = []
        for p in positions:
            dic = {}
            dic['origin'] = p
            dic['target'] = self.target
            dic['up'] = [0,0,1]
            pose_list.append(dic)
        return pose_list
