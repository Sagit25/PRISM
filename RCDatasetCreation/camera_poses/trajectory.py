"""Sequential path/trajectory camera samplers."""
import numpy as np

from utils.tool_utils import spherical2cartesian
from utils.registry import CAMPOSE_REGISTRY


@CAMPOSE_REGISTRY.register()
class SpiralPose:
    def __init__(self, conf):
        self.conf = conf
        self.total_num = conf.get('total_num', 60)
        self.phi_range = conf.get('phi_range', [0, 360])
        self.theta_range = conf.get('theta_range', [20, 70])
        self.r = conf['r']
        self.shift = np.array(self.conf.get('shift', [0, 0, 0]))
        self.target = conf.get('target', [0, 0, 0])

        self.pose_list = self.spiral_sample()
        self.img_num = len(self.pose_list)

    def spiral_sample(self):
        theta = np.linspace(self.theta_range[0], self.theta_range[1], self.total_num)
        phi = np.linspace(self.phi_range[0], self.phi_range[1], self.total_num)

        position_list = []
        for t, p in zip(theta, phi):
            position = spherical2cartesian(t, p, self.r)
            position += self.shift
            position_list.append(position)

        pose_list = []
        for position in position_list:
            dic = {}
            dic['origin'] = position
            dic['target'] = self.target
            dic['up'] = [0,0,1]
            pose_list.append(dic)
        return pose_list


@CAMPOSE_REGISTRY.register()
class UniformSpiralPose:
    def __init__(self, conf):
        """Uniform arc-length spiral camera path on a hemisphere."""
        self.conf = conf
        self.total_num = conf.get('total_num',60)
        self.turns = conf.get('turns',2)
        self.r = conf.get('r',1.0)
        self.shift = np.array(conf.get('shift',[0, 0, 0]))
        self.target = np.array(conf.get('target',[0, 0, 0]))
        self.theta_range = np.array(conf.get('theta_range',[0, np.rad2deg(np.pi/2)]))

        self.pose_list = self.spiral_sample()
        self.img_num = len(self.pose_list)

    def spiral_sample(self):
        """Generate uniformly spaced camera poses along a hemisphere spiral."""
        # Densely sample the spiral curve
        oversample = 10000
        t_dense = np.linspace(0, 1, oversample)

        # Spherical coordinates along the spiral
        phi = (np.pi * 2 * self.turns) * t_dense
        theta = np.linspace(np.deg2rad(self.theta_range[0]), np.deg2rad(self.theta_range[1]), oversample)

        # Convert to Cartesian
        x = self.r * np.sin(theta) * np.cos(phi)
        y = self.r * np.sin(theta) * np.sin(phi)
        z = self.r * np.cos(theta)

        # Cumulative arc length
        dx = np.diff(x)
        dy = np.diff(y)
        dz = np.diff(z)
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        s = np.concatenate(([0.0], np.cumsum(dist)))

        # Resample at equal arc-length intervals
        s_samples = np.linspace(0, s[-1], self.total_num)
        x_samples = np.interp(s_samples, s, x)
        y_samples = np.interp(s_samples, s, y)
        z_samples = np.interp(s_samples, s, z)

        # Build pose list
        pose_list = []
        for i in range(self.total_num):
            position = np.array([x_samples[i], y_samples[i], z_samples[i]]) + self.shift
            pose_dict = {
                'origin': position,
                'target': self.target,
                'up': [0, 0, 1]
            }
            pose_list.append(pose_dict)

        return pose_list


@CAMPOSE_REGISTRY.register()
class OrbitVideoPose:
    def __init__(self,conf):
        self.conf = conf
        self.phi_range = conf.get('phi_range', [0, 360])
        circle_time = conf.get('circle_time', 10)
        self.phi_num = int(circle_time * 60)
        self.theta_range = conf.get('theta_range', [20, 70])
        self.theta_num = conf.get('theta_num', 6)
        self.r = conf['r']
        self.shift = np.array(self.conf.get('shift', [0, 0, 0]))
        self.target = conf.get('target', [0, 0, 0])

        self.pose_list = self.video_sample()
        self.img_num = len(self.pose_list)

    def video_sample(self):
        sample_num = int(self.theta_num*self.phi_num)
        theta = np.linspace(self.theta_range[0], self.theta_range[1], sample_num) * np.pi / 180.0

        if self.phi_range[1] == 360 and self.phi_range[0] == 0:
            phi = np.linspace(self.phi_range[0], self.phi_range[1], self.phi_num + 1) * np.pi / 180.0
            phi = phi[:-1]
        else:
            phi = np.linspace(self.phi_range[0], self.phi_range[1], sample_num) * np.pi / 180.0

        position_list = []
        for idx in range(sample_num):
            t = theta[idx]
            p = phi[idx%self.phi_num]
            position = self.r * np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])
            position += self.shift
            position_list.append(position)

        pose_list = []
        for position in position_list:
            dic = {}
            dic['origin'] = position
            dic['target'] = self.target
            dic['up'] = [0,0,1]
            pose_list.append(dic)
        return pose_list


@CAMPOSE_REGISTRY.register()
class ZoomInPose:
    def __init__(self, conf) -> None:
        self.conf = conf
        self.init_position = conf.get('init_position', [1, 0, 0])
        self.end_position = conf.get('end_position', [0, 0, 0])
        self.sample_num = conf.get('sample_num', 6)
        self.target = conf.get('target', [0, 0, 0])
        self.pose_list = self.interp_location()
        self.img_num = len(self.pose_list)

    def interp_location(self):
        pose_list = []
        init_position = np.array(self.init_position)
        end_position = np.array(self.end_position)
        for i in range(self.sample_num):
            position = init_position + (end_position - init_position) * i / (self.sample_num - 1)
            dic = {}
            dic['origin'] = list(position)
            dic['target'] = self.target
            dic['up'] = [0,0,1]
            pose_list.append(dic)
        return pose_list
