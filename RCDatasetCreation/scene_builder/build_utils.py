"""Factories for instantiating scenes and camera pose generators."""
from utils.registry import SCENE_REGISTRY, CAMPOSE_REGISTRY

# Import modules to trigger registration
import scene_builder.scene  # noqa: F401
import camera_poses  # noqa: F401


def build_scene(conf):
    scene_cls = SCENE_REGISTRY.get(conf['type'])
    return scene_cls(conf)


def build_pose_generator(conf):
    cam_cls = CAMPOSE_REGISTRY.get(conf['type'])
    return cam_cls(conf)
