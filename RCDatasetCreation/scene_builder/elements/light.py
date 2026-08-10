"""Light-related scene elements (emitters, environment maps, etc.)."""
from __future__ import annotations

from utils.registry import ELEMENT_REGISTRY
import scene_builder.mitsuba_utils as lib

__all__ = [
    'envmap_light', 'constant_environment_light', 'point_light', 'directional_light',
    'spot_light', 'mesh_area_light', 'rectangle_area_light',
]


@ELEMENT_REGISTRY.register()
def envmap_light(scene, conf):
    """Attach an environment map emitter from `env_map/<filename>`."""
    env_path = scene.resolve_source_path('env_map', conf['envmap_filename'])
    intensity = conf.get('intensity', 1.0)
    z_up = conf.get('z_up', False)
    node_name = conf.get('name') or 'env_light'
    return {node_name: lib.make_environment_map_light(env_path, intensity, z_up)}


@ELEMENT_REGISTRY.register()
def constant_environment_light(scene, conf):
    """Create a constant (uniform) environment light."""
    intensity = conf.get('intensity', 1.0)
    node_name = conf.get('name') or 'env_light'
    return {node_name: lib.make_constant_environment_light(intensity)}


@ELEMENT_REGISTRY.register()
def point_light(scene, conf):
    """Create a point light source at specified position."""
    position = conf['position']
    intensity = conf.get('intensity', 1.0)
    node_name = conf.get('name') or 'point_light'
    return {node_name: lib.point_light(position, intensity)}


@ELEMENT_REGISTRY.register()
def directional_light(scene, conf):
    """Create a directional (parallel) light source."""
    direction = conf.get('direction', [0, 0, -1])
    intensity = conf.get('intensity', 1.0)
    node_name = conf.get('name') or 'directional_light'
    return {node_name: lib.directional_light(direction, intensity)}


@ELEMENT_REGISTRY.register()
def spot_light(scene, conf):
    """Create a spot light with specified origin pointing at target."""
    origin = conf['origin']
    target = conf.get('target', [0, 0, 0])
    up = conf.get('up', [0, 0, 1])
    angle = conf.get('angle', 20.0)
    intensity = conf.get('intensity', 1.0)
    shape_scale = conf.get('shape_scale', 1.0)
    node_name = conf.get('name') or 'spot_light'
    to_world = lib.look_at(origin, target, up) @ lib.transform_scale(shape_scale)
    return {node_name: lib.make_spot_light(intensity, angle, to_world)}


@ELEMENT_REGISTRY.register()
def mesh_area_light(scene, conf):
    """Create an area light from a mesh shape."""
    object_path = scene.resolve_source_path('shape', conf['mesh_filename'])
    node_name = conf.get('name') or 'mesh_area_light'
    intensity = conf.get('intensity', 1.0)

    if 'pattern_filename' in conf:
        pattern_path = scene.resolve_source_path('texture', conf['texture_filename'])
        emitter = lib.make_textured_area_emitter(pattern_path, intensity)
    else:
        emitter = lib.make_constant_area_emitter(intensity)

    return {node_name: lib.make_mesh_shape(object_path, other_property={'emitter': emitter})}


@ELEMENT_REGISTRY.register()
def rectangle_area_light(scene, conf):
    """Create a rectangle area light at specified position."""
    intensity = conf.get('intensity', 1.0)
    scale = conf.get('scale', 1.0)
    translate = conf['translate']
    node_name = conf.get('name') or 'rectangle_area_light'

    to_world = (lib.transform_translate(translate)
                @ lib.transform_rotate([1, 0, 0], 180)
                @ lib.transform_scale(scale))
    emitter = lib.make_constant_area_emitter(intensity)
    return {node_name: lib.make_rectangle_light_shape(emitter, to_world)}
