"""Misc Mitsuba helpers used across projects."""
import numpy as np

import mitsuba as mi

DEFAULT_TEXTURE_PATH = './resources/default.png'


def _T():
    """Get ScalarTransform4f type after variant is set."""
    return mi.ScalarTransform4f


def _P3f():
    """Get ScalarPoint3f type after variant is set."""
    return mi.ScalarPoint3f


# ============================================================================
# Transform functions
# ============================================================================

def look_at(origin, target, up=(0, 0, 1)):
    """Create a look-at transform matrix."""
    T, P3f = _T(), _P3f()
    if isinstance(origin, (list, tuple)):
        origin = P3f(float(origin[0]), float(origin[1]), float(origin[2]))
    if isinstance(target, (list, tuple)):
        target = P3f(float(target[0]), float(target[1]), float(target[2]))
    if isinstance(up, (list, tuple)):
        up = P3f(float(up[0]), float(up[1]), float(up[2]))
    return T().look_at(origin, target, up)


def transform_scale(scale):
    """Create a scale transform. Accepts scalar or [sx, sy, sz]."""
    T, P3f = _T(), _P3f()
    if isinstance(scale, (int, float)):
        s = float(scale)
        return T().scale(P3f(s, s, s))
    if isinstance(scale, (list, tuple)) and len(scale) == 3:
        return T().scale(P3f(float(scale[0]), float(scale[1]), float(scale[2])))
    return T().scale(scale)


def transform_translate(trans):
    """Create a translation transform from [tx, ty, tz]."""
    T, P3f = _T(), _P3f()
    if isinstance(trans, (list, tuple)) and len(trans) == 3:
        return T().translate(P3f(float(trans[0]), float(trans[1]), float(trans[2])))
    return T().translate(trans)


def transform_rotate(axis, angle):
    """Create a rotation transform. Axis is [x, y, z], angle in degrees."""
    T, P3f = _T(), _P3f()
    if isinstance(axis, (list, tuple)) and len(axis) == 3:
        return T().rotate(P3f(float(axis[0]), float(axis[1]), float(axis[2])), float(angle))
    return T().rotate(axis, float(angle))


def transform():
    """Return identity transform."""
    return _T()()


# ============================================================================
# Texture and color functions
# ============================================================================

def make_rgb_value(value):
    """Create RGB color. Value can be scalar or [r, g, b]."""
    return mi.load_dict({'type': 'rgb', 'value': value})


def make_bitmap_texture(path=DEFAULT_TEXTURE_PATH, raw=False):
    """
    Load a bitmap texture.

    Args:
        path: Texture file path
        raw: If True, load exact pixel values without gamma correction.
             If False, apply inverse gamma for sRGB textures.
             Use raw=True for normal maps, roughness maps, etc.
             Use raw=False for color/albedo textures.
    """
    if path is None:
        return mi.load_dict({'type': 'rgb', 'value': 0.5})
    return mi.load_dict({
        'type': 'bitmap',
        'filename': path,
        'raw': raw,
    })


# ============================================================================
# Material functions
# ============================================================================

def diffuse_color(scale=0.5):
    """Create a diffuse BSDF with constant color."""
    return mi.load_dict({
        'type': 'diffuse',
        'reflectance': make_rgb_value(scale),
    })


def diffuse_texture(path=DEFAULT_TEXTURE_PATH, raw=False):
    """Create a diffuse BSDF with texture. Use raw=False for sRGB textures."""
    return mi.load_dict({
        'type': 'diffuse',
        'reflectance': make_bitmap_texture(path, raw=raw),
    })


def diffuse_checkboard():
    """Create a diffuse BSDF with checkerboard pattern."""
    return mi.load_dict({
        'type': 'diffuse',
        'reflectance': {
            'type': 'checkerboard',
            'color0': make_rgb_value(0.1),
            'color1': make_rgb_value(0.5),
        },
    })


def dielectric_bsdf(ior=1.5, ext_ior=1.0, reflection=True):
    """Create a dielectric (glass) BSDF."""
    bsdf_dict = {
        'type': 'dielectric',
        'int_ior': ior,
        'ext_ior': ext_ior,
    }
    if not reflection:
        bsdf_dict['specular_reflectance'] = {'type': 'uniform', 'value': 0}
    return mi.load_dict(bsdf_dict)


def principled_bsdf(albedo_path, roughness_path=None, metalness_path=None, normal_path=None):
    """Create a Disney Principled BSDF with optional PBR texture maps."""
    bsdf_dict = {
        'type': 'principled',
        'base_color': make_bitmap_texture(albedo_path),
    }
    if roughness_path is not None:
        bsdf_dict['roughness'] = make_bitmap_texture(roughness_path, raw=True)
    if metalness_path is not None:
        bsdf_dict['metallic'] = make_bitmap_texture(metalness_path, raw=True)

    bsdf = mi.load_dict(bsdf_dict)

    if normal_path is not None:
        bsdf = mi.load_dict({
            'type': 'normalmap',
            'normalmap': make_bitmap_texture(normal_path, raw=True),
            'bsdf': bsdf,
        })
    return bsdf


# ============================================================================
# Light functions
# ============================================================================

def point_light(position, intensity=1.0):
    """Create a point light source."""
    return mi.load_dict({
        'type': 'point',
        'position': position,
        'intensity': {'type': 'rgb', 'value': intensity},
    })


def directional_light(direction, intensity=1.0):
    """Create a directional (infinite distance) light source."""
    return mi.load_dict({
        'type': 'directional',
        'direction': direction,
        'irradiance': {'type': 'rgb', 'value': intensity},
    })


def make_spot_light(intensity=1.0, cutoff_angle=20, to_world=None):
    """Create a spot light source."""
    if to_world is None:
        to_world = transform_scale(1.0)
    return mi.load_dict({
        'type': 'spot',
        'intensity': intensity,
        'cutoff_angle': cutoff_angle,
        'to_world': to_world,
    })


def make_constant_environment_light(intensity=1.0):
    """Create a constant (uniform) environment light."""
    return mi.load_dict({
        'type': 'constant',
        'radiance': intensity,
    })


def make_environment_map_light(path, intensity=1.0, z_up=False):
    """Create an environment map emitter from texture file."""
    emitter = {
        'type': 'envmap',
        'filename': path,
        'scale': intensity,
    }
    if z_up:
        emitter['to_world'] = _T().rotate(axis=[1, 0, 0], angle=90)
    return mi.load_dict(emitter)


def make_constant_area_emitter(intensity):
    """Create a constant area emitter."""
    return mi.load_dict({
        'type': 'area',
        'radiance': intensity,
    })


def make_textured_area_emitter(path, intensity=1.0):
    """Create a textured area emitter."""
    emitter = mi.load_dict({
        'type': 'area',
        'radiance': make_bitmap_texture(path, raw=True),
    })
    if intensity != 1.0:
        params = mi.traverse(emitter)
        params['radiance.data'] = params['radiance.data'] * intensity
        params.update()
    return emitter


# ============================================================================
# Geometry functions
# ============================================================================

def make_mesh_shape(path, face_normal=False, other_property=None):
    """
    Load a mesh from file.

    Args:
        path: Mesh file path
        face_normal: If True, use face normals instead of interpolated vertex normals
        other_property: Additional properties dict (e.g., {'bsdf': ...})
    """
    if other_property is None:
        other_property = {}
    filetype = path.split('.')[-1]
    shape_dict = {
        'type': filetype,
        'filename': path,
        'face_normals': face_normal,
    }
    shape_dict.update(other_property)
    return mi.load_dict(shape_dict)


def vertex_color_mesh(path, vertex_color_attribute='vertex_color', face_normal=False):
    """Load a mesh and shade it using an embedded per-vertex color attribute."""
    bsdf = mi.load_dict({
        'type': 'diffuse',
        'reflectance': {
            'type': 'mesh_attribute',
            'name': vertex_color_attribute,
        }
    })
    return make_mesh_shape(path, face_normal=face_normal, other_property={'bsdf': bsdf})


def make_rectangle_shape(material=None, to_world=None):
    """Create a rectangle shape with optional material and transform."""
    if material is None:
        material = diffuse_color(0.5)
    if to_world is None:
        to_world = transform()
    return mi.load_dict({
        'type': 'rectangle',
        'bsdf': material,
        'to_world': to_world,
    })


def make_rectangle_light_shape(emitter=None, to_world=None):
    """Create a rectangle with area emitter."""
    if to_world is None:
        to_world = transform()
    return mi.load_dict({
        'type': 'rectangle',
        'to_world': to_world,
        'emitter': emitter,
    })


# ============================================================================
# Camera and scene manipulation functions
# ============================================================================

def set_camera_pose(sensor, origin, target, up=(0, 0, 1)):
    """Update the Mitsuba sensor with a new look-at transform."""
    params = mi.traverse(sensor)
    params['to_world'] = mi.Transform4f.look_at(origin=origin, target=target, up=up)
    params.update()


def set_camera_size(camera, size):
    """Update the camera film resolution."""
    params = mi.traverse(camera)
    params['film.size'] = size
    params.update()


def set_to_world_transform(scene, key_word, transform):
    """Update the transform of an object in the scene."""
    params = mi.traverse(scene)
    params[key_word + '.to_world'] = transform
    params.update()


# ============================================================================
# SVBRDF functions
# ============================================================================

def read_svbrdf(path, power_roughness=True):
    """Read packed SVBRDF image and split into normal, diffuse, roughness, specular."""
    import numpy as np
    import imageio

    svbrdf = np.asarray(imageio.imread(path)).astype(np.float32)
    if np.issubdtype(svbrdf.dtype, np.integer) or svbrdf.max() > 1.0:
        svbrdf = svbrdf / 255.0

    size = svbrdf.shape[0]
    svbrdf = svbrdf[:, -size*4:]
    n = svbrdf[:, :size]
    d = svbrdf[:, size:size*2]
    r = svbrdf[:, size*2:size*3]
    s = svbrdf[:, size*3:]

    if power_roughness:
        r = np.power(r, 2)
    return n, d, r, s


def svbrdf_bsdf(normal, diffuse, roughness, specular, svbrdf_path):
    """Build a normalmap+blend+roughconductor BSDF from SVBRDF channel arrays."""
    import numpy as np
    import cv2

    # Build BSDF with placeholder textures
    normal_bmp = mi.load_dict({'type': 'bitmap', 'filename': svbrdf_path, 'raw': True})
    diffuse_bmp = mi.load_dict({'type': 'bitmap', 'filename': svbrdf_path, 'raw': True})
    alpha_bmp = mi.load_dict({'type': 'bitmap', 'filename': svbrdf_path, 'raw': True})
    spec_bmp = mi.load_dict({'type': 'bitmap', 'filename': svbrdf_path, 'raw': True})

    bsdf_diffuse = mi.load_dict({'type': 'diffuse', 'reflectance': diffuse_bmp})
    rc = mi.load_dict({
        'type': 'roughconductor',
        'alpha': alpha_bmp,
        'distribution': 'ggx',
        'specular_reflectance': spec_bmp,
    })
    blend = mi.load_dict({'type': 'blendbsdf', 'weight': 0.5, 'bsdf_0': bsdf_diffuse, 'bsdf_1': rc})
    bsdf = mi.load_dict({'type': 'normalmap', 'normalmap': normal_bmp, 'bsdf': blend})

    # Update with actual SVBRDF data
    params = mi.traverse(bsdf)
    keys = [
        'normalmap.data',
        'nested_bsdf.bsdf_0.reflectance.data',
        'nested_bsdf.bsdf_1.alpha.data',
        'nested_bsdf.bsdf_1.specular_reflectance.data',
    ]
    for key, data in zip(keys, [normal, diffuse, roughness, specular]):
        data_arr = np.array(data)
        cur = params[key]
        target_shape = tuple(cur.shape)
        if len(target_shape) >= 2:
            tgt_h, tgt_w = int(target_shape[0]), int(target_shape[1])
            if data_arr.shape[0] != tgt_h or data_arr.shape[1] != tgt_w:
                data_arr = cv2.resize(data_arr, (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR)
        if len(target_shape) == 3:
            tgt_c = int(target_shape[2])
            if data_arr.ndim == 2:
                data_arr = np.repeat(data_arr[:, :, None], tgt_c, axis=2)
            elif data_arr.shape[2] != tgt_c:
                if data_arr.shape[2] < tgt_c:
                    reps = (tgt_c + data_arr.shape[2] - 1) // data_arr.shape[2]
                    data_arr = np.tile(data_arr, (1, 1, reps))
                data_arr = data_arr[:, :, :tgt_c]
        else:
            if data_arr.ndim == 3:
                data_arr = data_arr.mean(axis=2)
        params[key] = mi.TensorXf(data_arr.astype(np.float32))
    params.update()
    return bsdf


# ============================================================================
# Integrator functions
# ============================================================================

def wrap_with_aov(integrator_conf, normal=False, depth=False, albedo=False):
    """
    Wrap an integrator config with AOV for normal/depth output.

    Args:
        integrator_conf: Base integrator configuration dict
        normal: If True, include shading normal in AOV output
        depth: If True, include depth in AOV output
        albedo: If True, include albedo in AOV output

    Returns:
        Wrapped integrator config if AOV is needed, otherwise original config
    """
    aov_parts = []
    if normal:
        aov_parts.append('nn:sh_normal')
    if depth:
        aov_parts.append('dd.y:depth')
    if albedo:
        aov_parts.append('ab:albedo')

    if not aov_parts:
        return integrator_conf

    return {
        'type': 'aov',
        'aovs': ','.join(aov_parts),
        'my_image': integrator_conf
    }


def swap_base_color(scene, property_name):
    """Swap base_color texture with another BSDF property (e.g., roughness, metallic).
    Returns dict of original base_color data for restoration."""
    params = mi.traverse(scene)
    originals = {}

    for key in list(params.keys()):
        if 'base_color.data' not in key:
            continue
        prop_key = key.replace('base_color.data', f'{property_name}.data')
        if prop_key not in params:
            continue
        # Save original
        originals[key] = mi.TensorXf(params[key])
        # Handle channel mismatch: roughness is 1-ch, base_color is 3-ch
        src = np.array(params[prop_key])
        tgt = np.array(params[key])
        if src.ndim == 3 and tgt.ndim == 3 and src.shape[2] != tgt.shape[2]:
            if src.shape[2] == 1:
                src = np.repeat(src, tgt.shape[2], axis=2)
            else:
                src = src[:, :, :tgt.shape[2]]
        params[key] = mi.TensorXf(src.astype(np.float32))

    params.update()
    return originals


def restore_base_color(scene, originals):
    """Restore base_color textures from saved originals."""
    params = mi.traverse(scene)
    for key, data in originals.items():
        params[key] = data
    params.update()
