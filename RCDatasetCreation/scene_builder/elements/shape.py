"""Shape-related scene elements (meshes, planes, etc.)."""
from __future__ import annotations

from utils.registry import ELEMENT_REGISTRY
import scene_builder.mitsuba_utils as lib

__all__ = [
    'white_mesh', 'transparent_mesh', 'pbr_mesh', 'texture_mesh', 'vertex_color_mesh',
    'svbrdf_rectangle', 'textured_rectangle', 'diffuse_rectangle',
]


@ELEMENT_REGISTRY.register()
def white_mesh(scene, conf):
    """Load a diffuse mesh with constant albedo."""
    object_path = scene.resolve_source_path('shape', conf['mesh_filename'])
    reflectance = conf.get('reflectance', 0.5)
    node_name = conf.get('name') or 'white_mesh'
    return {node_name: lib.make_mesh_shape(object_path, other_property={'bsdf': lib.diffuse_color(reflectance)})}


@ELEMENT_REGISTRY.register()
def transparent_mesh(scene, conf):
    """Load a glass/dielectric mesh for transparent object rendering."""
    object_path = scene.resolve_source_path('shape', conf['mesh_filename'])
    ior = conf.get('IoR', 1.5)
    reflection_flag = conf.get('reflection_flag', True)
    face_normal = conf.get('face_normal', False)
    node_name = conf.get('name') or 'transparent_mesh'
    bsdf = lib.dielectric_bsdf(ior=ior, reflection=reflection_flag)
    return {node_name: lib.make_mesh_shape(object_path, face_normal=face_normal, other_property={'bsdf': bsdf})}


@ELEMENT_REGISTRY.register()
def pbr_mesh(scene, conf):
    """Load a mesh with PBR (Disney Principled) material from texture maps."""
    shape_path = scene.resolve_source_path('shape', conf['mesh_filename'])
    a_path = scene.resolve_source_path('texture', conf['albedo_filename'])
    node_name = conf.get('name') or 'pbr_mesh'

    r_path = scene.resolve_source_path('texture', conf['roughness_filename']) if 'roughness_filename' in conf else None
    m_path = scene.resolve_source_path('texture', conf['metalness_filename']) if 'metalness_filename' in conf else None
    n_path = scene.resolve_source_path('texture', conf['normal_filename']) if 'normal_filename' in conf else None

    bsdf = lib.principled_bsdf(a_path, roughness_path=r_path, metalness_path=m_path, normal_path=n_path)
    face_normal = conf.get('face_normal', False)
    return {node_name: lib.make_mesh_shape(shape_path, face_normal=face_normal, other_property={'bsdf': bsdf})}


@ELEMENT_REGISTRY.register()
def texture_mesh(scene, conf):
    """Load a mesh with diffuse texture. Use 'checkboard' for procedural pattern."""
    object_path = scene.resolve_source_path('shape', conf['mesh_filename'])
    texture_map = conf['texture_filename']
    node_name = conf.get('name') or 'texture_mesh'

    if texture_map == 'checkboard':
        material = lib.diffuse_checkboard()
    else:
        texture_path = scene.resolve_source_path('texture', texture_map)
        material = lib.diffuse_texture(texture_path, raw=True)

    return {node_name: lib.make_mesh_shape(object_path, other_property={'bsdf': material})}


@ELEMENT_REGISTRY.register()
def vertex_color_mesh(scene, conf):
    """Load a mesh whose colors are stored directly on its vertices."""
    object_path = scene.resolve_source_path('shape', conf['mesh_filename'])
    node_name = conf.get('name') or 'vertex_color_mesh'
    face_normal = conf.get('face_normal', False)
    vertex_color_attribute = conf.get('vertex_color_attribute', 'vertex_color')
    return {
        node_name: lib.vertex_color_mesh(
            object_path,
            vertex_color_attribute=vertex_color_attribute,
            face_normal=face_normal,
        )
    }


@ELEMENT_REGISTRY.register()
def textured_rectangle(scene, conf):
    """Create a textured rectangle plane."""
    texture_path = scene.resolve_source_path('texture', conf['texture_filename'])
    scale = conf.get('scale', 1.0)
    node_name = conf.get('name') or 'textured_rectangle'
    to_world = lib.transform_scale(scale)
    bsdf = lib.diffuse_texture(texture_path)
    return {node_name: lib.make_rectangle_shape(material=bsdf, to_world=to_world)}


@ELEMENT_REGISTRY.register()
def diffuse_rectangle(scene, conf):
    """Create a diffuse rectangle plane."""
    reflectance = conf.get('reflectance', 0.5)
    scale = conf.get('scale', 1.0)
    node_name = conf.get('name') or 'diffuse_rectangle'
    to_world = lib.transform_scale(scale)
    if 'translate' in conf:
        to_world = lib.transform_translate(conf['translate']) @ to_world
    bsdf = lib.diffuse_color(reflectance)
    return {node_name: lib.make_rectangle_shape(material=bsdf, to_world=to_world)}


@ELEMENT_REGISTRY.register()
def svbrdf_rectangle(scene, conf):
    """Create a rectangle with SVBRDF material (Cook-Torrance model)."""
    svbrdf_path = scene.resolve_source_path('texture', conf['svbrdf_filename'])
    scale = conf.get('scale', 1.0)
    node_name = conf.get('name') or 'svbrdf_rectangle'
    n, d, r, s = lib.read_svbrdf(svbrdf_path)
    bsdf = lib.svbrdf_bsdf(n, d, r, s, svbrdf_path)
    to_world = lib.transform_scale([scale, scale, 1])
    return {node_name: lib.make_rectangle_shape(material=bsdf, to_world=to_world)}
