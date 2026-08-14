"""Sequence dataset generation for moving transparent objects.

This version extends the original RCTrans dataset generator with:

* a fixed camera and either a textured plane or a true 3D background scene,
* deterministic rigid object motion,
* full, reflection-off, object-only, background, and white-transmission passes,
* a directly rendered black-reference premultiplied foreground,
* seed-aligned decomposition passes with identical camera sampling,
* MAM2-compatible linear-RGB decomposition targets with reflection absorbed
  into the foreground,
* 3D refracted-background hits and pixel-space displacement ``Phi``,
* explicit source-coordinate correspondence ``Phi_src``,
* full-scene and object-only shading-normal/depth ground truth, and
* sequence-aware checkpoint/resume and metadata.

The continuous reflection control assumes the companion changes described for
``scene_builder/mitsuba_utils.py`` and ``scene_builder/elements/shape.py``.
For compatibility with an unmodified tracer/camera sampler, this file falls
back to rebuilding the tracer per frame and temporarily seeding NumPy.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
import os.path as osp
import warnings

import cv2
import drjit as dr
import imageio.v2 as imageio
import mitsuba as mi
import numpy as np
import trimesh
from mitsuba import ScalarTransform4f as T
from tqdm import tqdm

from projects.base import BaseProject
from scene_builder.build_utils import build_pose_generator, build_scene
from scene_builder.mitsuba_utils import set_camera_pose
from utils.camera_utils import fov_to_intrinsic_mat, gen_rays, get_extrinsic_matrix
from utils.registry import PROJECT_REGISTRY
from utils.tool_utils import convert_to_dict
from utils.tracer_factory import build_tracer

try:
    import flow_vis

    HAS_FLOW_VIS = True
except ImportError:
    HAS_FLOW_VIS = False


@PROJECT_REGISTRY.register()
class RefractiveCorresDataset(BaseProject):
    """Generate fixed-camera sequences of a moving transparent object."""

    GENERATOR_VERSION = "v14_full_fresnel_reflection"

    def __init__(self, conf):
        self.raw_output_folder = None
        self.conf = conf
        self.preview_mode = bool(conf.get("preview", False))
        self.resource_folder = conf["Scene"]["source_path"]

        background_conf = convert_to_dict(conf.get("Background", {}))
        self.background_mode = str(
            background_conf.get("mode", "plane")
        ).lower()
        if self.background_mode not in ("plane", "scene"):
            raise ValueError("Background.mode must be 'plane' or 'scene'")
        self.background_subdir = (
            "background" if self.background_mode == "plane" else "background_3d"
        )

        self.load_resource_info()
        (
            self.bootstrap_shape_path,
            self.bootstrap_background_path,
        ) = self._select_bootstrap_assets()

        self.backgrounds_per_shape = int(
            conf.get("backgrounds_per_shape", conf.get("envnum_per_shape", 1))
        )
        self.sequence_num_per_background = int(
            conf.get(
                "sequence_num_per_background",
                conf.get("sequence_num_per_env", 1),
            )
        )
        self.frames_per_sequence = int(conf.get("frames_per_sequence", 8))
        self.dataset_seed = int(conf.get("dataset_seed", 20260810))
        self.ior_range = list(conf.get("ior_range", [1.3, 1.6]))

        if self.backgrounds_per_shape < 1:
            raise ValueError("backgrounds_per_shape must be at least 1")
        if self.sequence_num_per_background < 1:
            raise ValueError("sequence_num_per_background must be at least 1")
        if self.frames_per_sequence < 1:
            raise ValueError("frames_per_sequence must be at least 1")
        if len(self.ior_range) != 2 or self.ior_range[0] > self.ior_range[1]:
            raise ValueError("ior_range must be [min, max]")

        decomp_conf = conf.get("Decomposition", {})
        self.decomp_eps = float(decomp_conf.get("eps", 1e-4))
        self.validation_tolerance = float(
            decomp_conf.get("validation_tolerance", 1e-5)
        )
        self.save_debug_passes = bool(
            decomp_conf.get("save_debug_passes", True)
        )

        flow_png_conf = conf.get("FlowPNG", {})
        self.save_flow_png = bool(flow_png_conf.get("enabled", True))

        flow_arrow_conf = conf.get("FlowArrows", {})
        self.save_flow_arrows = bool(flow_arrow_conf.get("enabled", True))
        self.flow_arrow_stride = int(flow_arrow_conf.get("stride", 4))
        self.flow_arrow_canvas_scale = int(
            flow_arrow_conf.get("canvas_scale", 4)
        )
        self.flow_arrow_min_magnitude = float(
            flow_arrow_conf.get("min_magnitude_px", 0.25)
        )
        self.flow_arrow_max_count = int(
            flow_arrow_conf.get("max_arrows", 1200)
        )
        if self.flow_arrow_stride < 1:
            raise ValueError("FlowArrows.stride must be at least 1")
        if self.flow_arrow_canvas_scale < 1:
            raise ValueError("FlowArrows.canvas_scale must be at least 1")
        if self.flow_arrow_min_magnitude < 0.0:
            raise ValueError(
                "FlowArrows.min_magnitude_px must be non-negative"
            )
        if self.flow_arrow_max_count < 1:
            raise ValueError("FlowArrows.max_arrows must be at least 1")

        self.background_z = float(background_conf.get("z", -1.25))
        self.background_z_range = self._parse_range(
            background_conf.get("z_range", [self.background_z, self.background_z]),
            "Background.z_range",
        )
        self.background_tilt_x_range = self._parse_range(
            background_conf.get("tilt_x_deg_range", [0.0, 0.0]),
            "Background.tilt_x_deg_range",
        )
        self.background_tilt_y_range = self._parse_range(
            background_conf.get("tilt_y_deg_range", [0.0, 0.0]),
            "Background.tilt_y_deg_range",
        )
        self.background_tilt_x_deg = 0.0
        self.background_tilt_y_deg = 0.0
        plane_half_size = background_conf.get("half_size", [4.0, 4.0])
        if np.isscalar(plane_half_size):
            plane_half_size = [plane_half_size, plane_half_size]
        self.background_half_size = np.asarray(
            plane_half_size, dtype=np.float32
        )
        if (
            self.background_half_size.shape != (2,)
            or np.any(self.background_half_size <= 0.0)
        ):
            raise ValueError("Background.half_size must contain two positives")
        self.background_intensity = float(background_conf.get("intensity", 1.0))
        if self.background_intensity < 0.0:
            raise ValueError("Background.intensity must be non-negative")
        self.background_intensity_range = self._parse_range(
            background_conf.get(
                "intensity_range",
                [self.background_intensity, self.background_intensity],
            ),
            "Background.intensity_range",
            nonnegative=True,
        )
        self.current_background_intensity = self.background_intensity
        texture_conf = convert_to_dict(
            background_conf.get("TextureAugmentation", {})
        )
        self.texture_augmentation_enabled = bool(
            texture_conf.get("enabled", False)
        )
        self.texture_zoom_range = self._parse_range(
            texture_conf.get("zoom_range", [1.0, 1.0]),
            "Background.TextureAugmentation.zoom_range",
            positive=True,
        )
        self.texture_offset_x_range = self._parse_range(
            texture_conf.get("offset_x_range", [0.0, 0.0]),
            "Background.TextureAugmentation.offset_x_range",
        )
        self.texture_offset_y_range = self._parse_range(
            texture_conf.get("offset_y_range", [0.0, 0.0]),
            "Background.TextureAugmentation.offset_y_range",
        )
        self.texture_rotation_range = self._parse_range(
            texture_conf.get("rotation_deg_range", [0.0, 0.0]),
            "Background.TextureAugmentation.rotation_deg_range",
        )
        self.texture_flip_probability = float(
            texture_conf.get("horizontal_flip_probability", 0.0)
        )
        if not 0.0 <= self.texture_flip_probability <= 1.0:
            raise ValueError(
                "TextureAugmentation.horizontal_flip_probability must lie in [0, 1]"
            )
        self.ambient_intensity = float(
            background_conf.get("ambient_intensity", 1.0)
        )
        self.clean_visibility_tolerance = float(
            background_conf.get("clean_visibility_tolerance", 0.04)
        )
        if self.clean_visibility_tolerance < 0.0:
            raise ValueError(
                "Background.clean_visibility_tolerance must be non-negative"
            )
        self.background_texture_shape = None
        self.current_background_linear = None
        self.current_background_augmentation = None
        if self.background_mode == "plane":
            self.background_texture_shape = self._load_background_linear(
                self.bootstrap_background_path
            ).shape[:2]
        self.current_background_path = self.bootstrap_background_path

        base_scene_conf = convert_to_dict(conf["Scene"])
        base_scene_conf = self._patch_scene_bootstrap(base_scene_conf)

        # Full scene. Explicit reflection_scale=1.0 ensures a traversable
        # uniform reflectance parameter when the companion BSDF patch is used.
        self._full_scene_conf = copy.deepcopy(base_scene_conf)
        self._set_transparent_element_reflection(
            self._full_scene_conf, enabled=True, scale=1.0
        )
        self._wrap_geometry_aovs(self._full_scene_conf)

        # Counterfactual scene with the same camera/integrator but no reflection.
        self._no_ref_scene_conf = copy.deepcopy(base_scene_conf)
        self._set_transparent_element_reflection(
            self._no_ref_scene_conf, enabled=False, scale=0.0
        )
        self._wrap_geometry_aovs(self._no_ref_scene_conf)

        # Black-reference scenes keep the exact same dielectric, camera, and
        # illumination as their full-scene counterparts.  Their background
        # geometry is replaced by a non-contributing black version below, so
        # their RGB output is the standard premultiplied foreground alpha*F,
        # instead of a residual obtained by subtracting a single-flow warp.
        self._object_only_scene_conf = copy.deepcopy(base_scene_conf)
        self._set_transparent_element_reflection(
            self._object_only_scene_conf, enabled=True, scale=1.0
        )
        self._wrap_geometry_aovs(self._object_only_scene_conf)

        self._object_only_no_ref_scene_conf = copy.deepcopy(base_scene_conf)
        self._set_transparent_element_reflection(
            self._object_only_no_ref_scene_conf, enabled=False, scale=0.0
        )
        self._wrap_geometry_aovs(self._object_only_no_ref_scene_conf)

        self._build_background_dependent_scenes(
            self.bootstrap_background_path
        )

        # White illumination isolates per-channel transmission A=(1-alpha)T.
        self.transmission_scene = self._build_transmission_scene(
            self.bootstrap_shape_path
        )

        camera_conf = convert_to_dict(conf["Camera"])
        camera_fov = float(camera_conf.get("fov", 55.0))
        self.camera_fov_range = self._parse_range(
            camera_conf.pop("fov_range", [camera_fov, camera_fov]),
            "Camera.fov_range",
            positive=True,
        )
        self.current_camera_fov = camera_fov
        if self.preview_mode:
            camera_conf["film"]["height"] = max(
                1, camera_conf["film"]["height"] // 4
            )
            camera_conf["film"]["width"] = max(
                1, camera_conf["film"]["width"] // 4
            )
            camera_conf["sampler"]["sample_count"] = 128
        self.camera = mi.load_dict(camera_conf)

        campose_conf = convert_to_dict(conf["CamPose"])
        self.pose_generator = build_pose_generator(campose_conf)

        self._transmission_params = mi.traverse(self.transmission_scene)
        self._cam_params = mi.traverse(self.camera)
        self.current_reflection_scale = 1.0
        self.current_ior = None
        self.current_object_pose = np.eye(4, dtype=np.float32)

        self.setup_output_paths()

    @staticmethod
    def _parse_range(value, name, positive=False, nonnegative=False):
        values = np.asarray(value, dtype=np.float64)
        if values.shape != (2,) or not np.isfinite(values).all():
            raise ValueError(f"{name} must contain two finite numbers")
        low, high = float(values[0]), float(values[1])
        if low > high:
            raise ValueError(f"{name} minimum exceeds maximum")
        if positive and low <= 0.0:
            raise ValueError(f"{name} values must be positive")
        if nonnegative and low < 0.0:
            raise ValueError(f"{name} values must be non-negative")
        return (low, high)

    @staticmethod
    def _wrap_geometry_aovs(scene_conf):
        base_integrator = scene_conf.get(
            "integrator", {"type": "path", "max_depth": 20}
        )
        scene_conf["integrator"] = {
            "type": "aov",
            # Keep normal first so the established channel layout remains
            # RGB=[0:3], N=[3:6]. Depth is the final scalar channel.
            "aovs": "nn:sh_normal,dd.y:depth",
            "my_image": base_integrator,
        }

    @staticmethod
    def _set_transparent_element_reflection(scene_conf, enabled, scale):
        found = False
        for element in scene_conf["element"]:
            if element["type"] == "transparent_mesh":
                element["reflection_flag"] = bool(enabled)
                element["reflection_scale"] = float(scale)
                found = True
        if not found:
            raise KeyError("Scene.element must contain a transparent_mesh")

    def _background_plane_dict(
        self, background_path=None, white=False, black=False
    ):
        """Create the finite textured plane that defines the clean background."""
        if white and black:
            raise ValueError("A background plane cannot be both white and black")
        half_width, half_height = self.background_half_size.tolist()
        if black:
            radiance = {"type": "rgb", "value": [0.0, 0.0, 0.0]}
        elif white:
            radiance = {"type": "rgb", "value": 1.0}
        else:
            # Mitsuba 3.7 has no generic texture plugin named ``scale``.
            # Supply linear RGB pixels directly and apply the scalar here.
            background = self.current_background_linear
            if background is None:
                background = self._load_background_linear(background_path)
            background = np.ascontiguousarray(
                background * self.current_background_intensity,
                dtype=np.float32,
            )
            radiance = {
                "type": "bitmap",
                # Mitsuba >= 3.7 no longer implicitly converts NumPy arrays
                # supplied to a bitmap texture's ``data`` property.
                "data": mi.TensorXf(background),
                "raw": True,
                "filter_type": "bilinear",
                "wrap_mode": "clamp",
            }
        plane_transform = (
            T.translate([0.0, 0.0, self.background_z])
            @ T.rotate([1.0, 0.0, 0.0], self.background_tilt_x_deg)
            @ T.rotate([0.0, 1.0, 0.0], self.background_tilt_y_deg)
            @ T.scale([half_width, half_height, 1.0])
        )
        plane = {
            "type": "rectangle",
            "to_world": plane_transform,
            "emitter": {
                "type": "area",
                "radiance": radiance,
            },
        }
        return plane

    def _background_plane_frame(self):
        """Return centre and orthonormal axes matching the Mitsuba transform."""
        tilt_x = np.deg2rad(self.background_tilt_x_deg)
        tilt_y = np.deg2rad(self.background_tilt_y_deg)
        rx = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(tilt_x), -np.sin(tilt_x)],
                [0.0, np.sin(tilt_x), np.cos(tilt_x)],
            ],
            dtype=np.float32,
        )
        ry = np.asarray(
            [
                [np.cos(tilt_y), 0.0, np.sin(tilt_y)],
                [0.0, 1.0, 0.0],
                [-np.sin(tilt_y), 0.0, np.cos(tilt_y)],
            ],
            dtype=np.float32,
        )
        rotation = rx @ ry
        return (
            np.asarray([0.0, 0.0, self.background_z], dtype=np.float32),
            rotation[:, 0],
            rotation[:, 1],
            rotation[:, 2],
        )

    def _load_background_scene_manifest(self, background_path):
        manifest_path = osp.join(
            self.resource_folder, "background_3d", background_path
        )
        with open(manifest_path, "r") as file:
            manifest = json.load(file)
        objects = manifest.get("objects", [])
        if not objects:
            raise ValueError(
                f"3D background manifest has no objects: {manifest_path}"
            )
        ids = [int(item["id"]) for item in objects]
        if any(object_id <= 0 for object_id in ids) or len(ids) != len(set(ids)):
            raise ValueError(
                "Every 3D background object id must be unique and positive"
            )
        return manifest

    @staticmethod
    def _rgb_texture(value):
        color = np.asarray(value, dtype=np.float32)
        if color.shape != (3,) or np.any(color < 0.0):
            raise ValueError("Background object color must be a nonnegative RGB triple")
        return {"type": "rgb", "value": color.tolist()}

    def _background_object_dict(self, item, black=False):
        """Convert one editable manifest entry to a Mitsuba shape dict."""
        shape_type = str(item["type"]).lower()
        result = {"type": shape_type}

        if shape_type in ("rectangle", "cube", "disk"):
            transform = T.translate(item.get("translate", [0.0, 0.0, 0.0]))
            angle = float(item.get("rotate_deg", 0.0))
            if abs(angle) > 1e-12:
                transform = transform @ T.rotate(
                    item.get("rotate_axis", [0.0, 0.0, 1.0]), angle
                )
            transform = transform @ T.scale(item.get("scale", [1.0, 1.0, 1.0]))
            result["to_world"] = transform
        elif shape_type == "sphere":
            result["center"] = item.get("center", [0.0, 0.0, 0.0])
            result["radius"] = float(item.get("radius", 1.0))
        elif shape_type == "cylinder":
            result["p0"] = item.get("p0", [0.0, 0.0, 0.0])
            result["p1"] = item.get("p1", [0.0, 0.0, 1.0])
            result["radius"] = float(item.get("radius", 1.0))
        else:
            raise ValueError(
                f"Unsupported 3D background primitive '{shape_type}'. "
                "Use rectangle, cube, disk, sphere, or cylinder."
            )

        result["bsdf"] = {
            "type": "diffuse",
            "reflectance": self._rgb_texture(
                [0.0, 0.0, 0.0]
                if black
                else item.get("color", [0.5, 0.5, 0.5])
            ),
        }
        return result

    @staticmethod
    def _background_key(item):
        safe_name = "".join(
            char if char.isalnum() else "_"
            for char in str(item.get("name", "object"))
        )
        return f"background_{int(item['id']):04d}_{safe_name}"

    def _attach_background_scene(self, scene_wrapper, manifest, black=False):
        for item in manifest["objects"]:
            scene_wrapper.scene_dict[self._background_key(item)] = (
                self._background_object_dict(item, black=black)
            )
        scene_wrapper.scene = mi.load_dict(scene_wrapper.scene_dict)

    def _constant_emitter_dict(self, intensity=None):
        if intensity is None:
            intensity = self.ambient_intensity
        return {
            "type": "constant",
            "radiance": {
                "type": "rgb",
                "value": [float(intensity)] * 3,
            },
        }

    def _attach_background_plane(
        self, scene_wrapper, background_path, black=False
    ):
        """Add the same textured plane to a scene-builder scene and rebuild it."""
        scene_wrapper.scene_dict["background_plane"] = (
            self._background_plane_dict(
                background_path=background_path, black=black
            )
        )
        scene_wrapper.scene = mi.load_dict(scene_wrapper.scene_dict)

    def _build_background_scene(self, background_path):
        scene_dict = {
            "type": "scene",
            "integrator": {
                "type": "aov",
                "aovs": "nn:sh_normal,dd.y:depth",
                "my_image": {"type": "path", "max_depth": 4},
            },
        }
        if self.background_mode == "plane":
            scene_dict["background_plane"] = self._background_plane_dict(
                background_path=background_path
            )
        else:
            for item in self.background_manifest["objects"]:
                scene_dict[self._background_key(item)] = (
                    self._background_object_dict(item)
                )
            scene_dict["background_ambient"] = self._constant_emitter_dict()
        return mi.load_dict(scene_dict)

    def _build_background_intersectors(self):
        self._background_object_scenes = []
        self.background_object_ids = []
        self.background_object_names = []
        self.background_object_albedos = []
        if self.background_mode != "scene":
            return
        for item in self.background_manifest["objects"]:
            shape_dict = self._background_object_dict(item)
            scene = mi.load_dict({"type": "scene", "shape": shape_dict})
            self._background_object_scenes.append(scene)
            self.background_object_ids.append(int(item["id"]))
            self.background_object_names.append(str(item.get("name", "object")))
            self.background_object_albedos.append(
                np.asarray(item.get("color", [0.5, 0.5, 0.5]), dtype=np.float32)
            )

    def _build_background_dependent_scenes(self, background_path):
        """Build full/no-ref/object-only/clean scenes for one background."""
        self.current_background_path = background_path
        self.background_manifest = None
        if self.background_mode == "scene":
            self.background_manifest = self._load_background_scene_manifest(
                background_path
            )

        self.scene = build_scene(copy.deepcopy(self._full_scene_conf))
        self.no_ref_scene = build_scene(copy.deepcopy(self._no_ref_scene_conf))
        self.object_only_scene = build_scene(
            copy.deepcopy(self._object_only_scene_conf)
        )
        self.object_only_no_ref_scene = build_scene(
            copy.deepcopy(self._object_only_no_ref_scene_conf)
        )
        if self.background_mode == "plane":
            self._attach_background_plane(self.scene, background_path)
            self._attach_background_plane(self.no_ref_scene, background_path)
            self._attach_background_plane(
                self.object_only_scene, background_path, black=True
            )
            self._attach_background_plane(
                self.object_only_no_ref_scene, background_path, black=True
            )
        else:
            self._attach_background_scene(self.scene, self.background_manifest)
            self._attach_background_scene(
                self.no_ref_scene, self.background_manifest
            )
            self._attach_background_scene(
                self.object_only_scene,
                self.background_manifest,
                black=True,
            )
            self._attach_background_scene(
                self.object_only_no_ref_scene,
                self.background_manifest,
                black=True,
            )

        self.background_scene = self._build_background_scene(background_path)
        self._build_background_intersectors()

        self._scene_params = mi.traverse(self.scene.scene)
        self._no_ref_params = mi.traverse(self.no_ref_scene.scene)
        self._object_only_params = mi.traverse(self.object_only_scene.scene)
        self._object_only_no_ref_params = mi.traverse(
            self.object_only_no_ref_scene.scene
        )
        self._background_params = mi.traverse(self.background_scene)
        self._reflection_param_groups = [
            (
                self._scene_params,
                self._find_reflection_param_keys(self._scene_params),
            ),
            (
                self._object_only_params,
                self._find_reflection_param_keys(self._object_only_params),
            ),
        ]
        if self.background_mode == "plane":
            self._background_data_keys = {
                id(params): self._find_background_data_keys(params)
                for params in (
                    self._scene_params,
                    self._no_ref_params,
                    self._background_params,
                )
            }
        else:
            self._background_data_keys = {}

    def _build_transmission_scene(self, shape_path):
        mesh_path = osp.join(self.resource_folder, "shape", shape_path)
        mesh_type = osp.splitext(mesh_path)[1].lstrip(".").lower()
        if not mesh_type:
            raise ValueError(f"Cannot infer mesh type from: {mesh_path}")

        return mi.load_dict(
            {
                "type": "scene",
                "transparent_mesh": {
                    "type": mesh_type,
                    "filename": mesh_path,
                    "face_normals": False,
                    "bsdf": {
                        "type": "dielectric",
                        "int_ior": 1.5,
                        "ext_ior": 1.0,
                        "specular_reflectance": {
                            "type": "uniform",
                            "value": 0.0,
                        },
                        "specular_transmittance": {
                            "type": "uniform",
                            "value": 1.0,
                        },
                    },
                },
                "white_environment": self._constant_emitter_dict(intensity=1.0),
                "integrator": {"type": "path", "max_depth": 20},
            }
        )

    def setup_output_paths(self, split="train"):
        output_root = self.conf.get("output_folder", "./result")
        project_name = self.conf.get("project_name", "dataset")
        base_folder = osp.join(output_root, project_name)
        os.makedirs(base_folder, exist_ok=True)

        if self.raw_output_folder is None:
            self.raw_output_folder = base_folder
        self.info_file = osp.join(
            self.raw_output_folder, "{}_file.txt".format(split)
        )
        self.output_folder = osp.join(self.raw_output_folder, split)
        os.makedirs(self.output_folder, exist_ok=True)

    def load_resource_info(self):
        """Load shape and selected background-mode resource indices."""
        with open(
            osp.join(self.resource_folder, "shape", "train_shape.txt"), "r"
        ) as file:
            self.train_shape_list = [
                line for line in file.read().rstrip().split("\n") if line
            ]
        with open(
            osp.join(self.resource_folder, "shape", "test_shape.txt"), "r"
        ) as file:
            self.test_shape_list = [
                line for line in file.read().rstrip().split("\n") if line
            ]
        train_index = (
            "train_background.txt"
            if self.background_mode == "plane"
            else "train_scene.txt"
        )
        test_index = (
            "test_background.txt"
            if self.background_mode == "plane"
            else "test_scene.txt"
        )
        with open(
            osp.join(self.resource_folder, self.background_subdir, train_index),
            "r",
        ) as file:
            self.train_background_list = [
                line for line in file.read().rstrip().split("\n") if line
            ]
        with open(
            osp.join(self.resource_folder, self.background_subdir, test_index),
            "r",
        ) as file:
            self.test_background_list = [
                line for line in file.read().rstrip().split("\n") if line
            ]

    @staticmethod
    def _first_available(primary_list, fallback_list, kind):
        if primary_list:
            return primary_list[0]
        if fallback_list:
            return fallback_list[0]
        raise FileNotFoundError(f"No {kind} entries found in resource indices")

    def _require_resource_file(self, subdir, rel_path, kind):
        abs_path = osp.join(self.resource_folder, subdir, rel_path)
        if not osp.exists(abs_path):
            raise FileNotFoundError(f"Bootstrap {kind} file not found: {abs_path}")
        return rel_path

    def _select_bootstrap_assets(self):
        shape_path = self._first_available(
            self.train_shape_list, self.test_shape_list, "shape"
        )
        background_path = self._first_available(
            self.train_background_list,
            self.test_background_list,
            "background",
        )
        shape_path = self._require_resource_file("shape", shape_path, "shape")
        background_path = self._require_resource_file(
            self.background_subdir, background_path, "background"
        )
        return shape_path, background_path

    def _patch_scene_bootstrap(self, scene_conf):
        patched_elements = []
        has_ambient = False
        for element in scene_conf["element"]:
            if element["type"] == "transparent_mesh":
                element["mesh_filename"] = self.bootstrap_shape_path
                patched_elements.append(element)
            elif element["type"] == "envmap_light":
                # A 2D clean background is not an environment map. Replace the
                # old HDR emitter with neutral illumination used only for the
                # reflection/highlight term.
                if self.ambient_intensity > 0.0 and not has_ambient:
                    patched_elements.append(
                        {
                            "type": "constant_environment_light",
                            "name": "ambient_light",
                            "intensity": self.ambient_intensity,
                        }
                    )
                    has_ambient = True
            else:
                if element["type"] == "constant_environment_light":
                    has_ambient = True
                patched_elements.append(element)
        if self.ambient_intensity > 0.0 and not has_ambient:
            patched_elements.append(
                {
                    "type": "constant_environment_light",
                    "name": "ambient_light",
                    "intensity": self.ambient_intensity,
                }
            )
        scene_conf["element"] = patched_elements
        return scene_conf

    @staticmethod
    def _find_reflection_param_keys(params):
        return [
            key
            for key in params.keys()
            if "transparent_mesh.bsdf.specular_reflectance" in key
            and key.endswith(".value")
        ]

    @staticmethod
    def _find_background_data_keys(params):
        keys = [
            key
            for key in params.keys()
            if key.startswith("background_plane.") and key.endswith(".data")
        ]
        if len(keys) != 1:
            raise RuntimeError(
                "Expected one traversable background bitmap, found "
                f"{len(keys)}: {keys}"
            )
        return keys

    @staticmethod
    def _stable_seed(*parts):
        payload = "|".join(str(part) for part in parts).encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        return int.from_bytes(digest[:8], byteorder="little", signed=False)

    @staticmethod
    def _render_seed(seed):
        # Mitsuba expects a 32-bit seed. Keep it nonzero for clarity.
        return int(seed % (2**32 - 1)) + 1

    def _push_mesh_to_params(
        self, params, mesh, update_faces=False, update_ior=False
    ):
        params["transparent_mesh.vertex_positions"] = dr.ravel(
            mi.Point3f(np.asarray(mesh.vertices, dtype=np.float32).T)
        )
        params["transparent_mesh.vertex_normals"] = dr.ravel(
            mi.Point3f(np.asarray(mesh.vertex_normals, dtype=np.float32).T)
        )
        if update_faces:
            params["transparent_mesh.faces"] = dr.ravel(
                mi.Point3u(np.asarray(mesh.faces, dtype=np.uint32).T)
            )
        if update_ior:
            params["transparent_mesh.bsdf.eta"] = float(self.current_ior)
        params.update()

    def _push_mesh_to_all_scenes(
        self, mesh, update_faces=False, update_ior=False
    ):
        for params in (
            self._scene_params,
            self._no_ref_params,
            self._object_only_params,
            self._object_only_no_ref_params,
            self._transmission_params,
        ):
            self._push_mesh_to_params(
                params,
                mesh,
                update_faces=update_faces,
                update_ior=update_ior,
            )

    def update_mesh(self, shape_path, ior):
        """Load a canonical mesh at the beginning of a sequence."""
        mesh_path = osp.join(self.resource_folder, "shape", shape_path)
        mesh = trimesh.load(mesh_path, force="mesh", process=False)

        self.base_mesh = mesh.copy()
        self.mesh_pivot = (
            np.asarray(self.base_mesh.bounds[0], dtype=np.float32)
            + np.asarray(self.base_mesh.bounds[1], dtype=np.float32)
        ) * 0.5
        self.current_ior = float(ior)

        if hasattr(self, "tracer"):
            del self.tracer
        self.tracer = build_tracer(
            self.base_mesh, self.conf, obj_ior=self.current_ior
        )

        self._push_mesh_to_all_scenes(
            self.base_mesh, update_faces=True, update_ior=True
        )

    def set_object_pose(self, object_to_world):
        """Apply one canonical-to-world pose to every render/tracing scene."""
        object_to_world = np.asarray(object_to_world, dtype=np.float32)
        if object_to_world.shape != (4, 4):
            raise ValueError("object_to_world must be a 4x4 matrix")

        rotation = object_to_world[:3, :3]
        translation = object_to_world[:3, 3]
        base_vertices = np.asarray(self.base_mesh.vertices, dtype=np.float32)
        base_normals = np.asarray(self.base_mesh.vertex_normals, dtype=np.float32)

        vertices = (rotation @ base_vertices.T).T + translation
        normals = (rotation @ base_normals.T).T
        moved_mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=np.asarray(self.base_mesh.faces),
            vertex_normals=normals,
            process=False,
        )

        self._push_mesh_to_all_scenes(
            moved_mesh, update_faces=False, update_ior=False
        )

        if hasattr(self.tracer, "update_mesh"):
            self.tracer.update_mesh(moved_mesh)
        else:
            # Compatibility fallback for the current public repository.
            self.tracer = build_tracer(
                moved_mesh, self.conf, obj_ior=self.current_ior
            )

        self.current_object_pose = object_to_world.copy()

    @staticmethod
    def _srgb_to_linear(image):
        image = np.asarray(image, dtype=np.float32)
        return np.where(
            image <= 0.04045,
            image / 12.92,
            np.power((image + 0.055) / 1.055, 2.4),
        ).astype(np.float32)

    def _load_background_linear(self, background_path):
        background_abs_path = osp.join(
            self.resource_folder, "background", background_path
        )
        image = cv2.imread(background_abs_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(
                f"Could not read background image: {background_abs_path}"
            )
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError(
                f"Expected a grayscale/RGB background: {background_abs_path}"
            )

        image = image[..., :3][..., ::-1]
        if np.issubdtype(image.dtype, np.integer):
            scale = float(np.iinfo(image.dtype).max)
            image = self._srgb_to_linear(image.astype(np.float32) / scale)
        else:
            image = np.asarray(image, dtype=np.float32)
        if not np.isfinite(image).all():
            raise ValueError(
                f"Background contains NaN/Inf values: {background_abs_path}"
            )
        return image

    def _augment_background_texture(self, background, rng):
        """Apply one deterministic sequence-level crop/scale augmentation."""
        height, width = background.shape[:2]
        if not self.texture_augmentation_enabled:
            params = {
                "enabled": False,
                "zoom": 1.0,
                "offset_x": 0.0,
                "offset_y": 0.0,
                "rotation_deg": 0.0,
                "horizontal_flip": False,
            }
            return np.ascontiguousarray(background), params

        zoom = float(rng.uniform(*self.texture_zoom_range))
        offset_x = float(rng.uniform(*self.texture_offset_x_range))
        offset_y = float(rng.uniform(*self.texture_offset_y_range))
        rotation_deg = float(rng.uniform(*self.texture_rotation_range))
        horizontal_flip = bool(rng.random() < self.texture_flip_probability)

        # BORDER_REFLECT_101 prevents an artificial black border from becoming
        # a shortcut when zoom-out, offset, or rotation exposes source edges.
        centre = ((width - 1) * 0.5, (height - 1) * 0.5)
        matrix = cv2.getRotationMatrix2D(centre, rotation_deg, zoom)
        matrix[0, 2] += offset_x * width
        matrix[1, 2] += offset_y * height
        augmented = cv2.warpAffine(
            background,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        if horizontal_flip:
            augmented = augmented[:, ::-1]
        params = {
            "enabled": True,
            "zoom": zoom,
            "offset_x": offset_x,
            "offset_y": offset_y,
            "rotation_deg": rotation_deg,
            "horizontal_flip": horizontal_flip,
        }
        return np.ascontiguousarray(augmented, dtype=np.float32), params

    def update_background(self, background_path, rng=None):
        """Sample and install one background held fixed for a sequence."""
        if self.background_mode == "scene":
            if background_path != self.current_background_path:
                self._build_background_dependent_scenes(background_path)
            return {"mode": "scene"}

        if rng is None:
            rng = np.random.default_rng(
                self._stable_seed(self.dataset_seed, background_path, "plane")
            )
        self.background_z = float(rng.uniform(*self.background_z_range))
        self.background_tilt_x_deg = float(
            rng.uniform(*self.background_tilt_x_range)
        )
        self.background_tilt_y_deg = float(
            rng.uniform(*self.background_tilt_y_range)
        )
        self.current_background_intensity = float(
            rng.uniform(*self.background_intensity_range)
        )
        original = self._load_background_linear(background_path)
        augmented, texture_params = self._augment_background_texture(
            original, rng
        )
        self.current_background_linear = augmented
        self.background_texture_shape = augmented.shape[:2]

        # Rectangle transforms are not traversable in every supported Mitsuba
        # release. Rebuilding keeps rendering and analytic GT in exact sync.
        self._build_background_dependent_scenes(background_path)
        centre, tangent_u, tangent_v, normal = self._background_plane_frame()
        self.current_background_augmentation = {
            "mode": "plane",
            "z": self.background_z,
            "tilt_x_deg": self.background_tilt_x_deg,
            "tilt_y_deg": self.background_tilt_y_deg,
            "half_size": self.background_half_size.tolist(),
            "centre": centre.tolist(),
            "tangent_u": tangent_u.tolist(),
            "tangent_v": tangent_v.tolist(),
            "normal": normal.tolist(),
            "intensity": self.current_background_intensity,
            "texture": texture_params,
        }
        return copy.deepcopy(self.current_background_augmentation)

    def set_reflection_scale(self, reflection_scale):
        """Set reflection strength while leaving transmission unchanged."""
        reflection_scale = float(reflection_scale)
        if not 0.0 <= reflection_scale <= 1.0:
            raise ValueError("reflection_scale must lie in [0, 1]")

        groups_with_keys = 0
        for params, keys in self._reflection_param_groups:
            if not keys:
                continue
            groups_with_keys += 1
            for key in keys:
                params[key] = reflection_scale
            params.update()
        if (
            reflection_scale not in (0.0, 1.0)
            and groups_with_keys != len(self._reflection_param_groups)
        ):
            raise RuntimeError(
                "Continuous reflection control is unavailable. Apply the "
                "companion dielectric_bsdf()/transparent_mesh() patches so "
                "specular_reflectance.value is traversable."
            )

        self.current_reflection_scale = reflection_scale

    def sample_reflection_scale(self, rng):
        reflection_conf = self.conf.get("Reflection", {})
        if not bool(reflection_conf.get("enabled", True)):
            return 0.0

        zero_probability = float(
            reflection_conf.get("zero_probability", 0.0)
        )
        if not 0.0 <= zero_probability <= 1.0:
            raise ValueError("Reflection.zero_probability must lie in [0, 1]")
        if rng.random() < zero_probability:
            return 0.0

        low, high = reflection_conf.get(
            "reflection_scale_range", [1.0, 1.0]
        )
        low, high = float(low), float(high)
        if not 0.0 <= low <= high <= 1.0:
            raise ValueError(
                "Reflection.reflection_scale_range must lie in [0, 1]"
            )
        return float(rng.uniform(low, high))

    def _random_cam_pose(self, rng=None):
        """Sample one reproducible camera pose for an entire sequence."""
        if not hasattr(self.pose_generator, "random_sample"):
            if not getattr(self.pose_generator, "pose_list", None):
                raise RuntimeError(
                    "Camera pose generator provides neither random_sample() "
                    "nor a nonempty pose_list"
                )
            self.cam_pose = copy.deepcopy(self.pose_generator.pose_list[0])
        elif rng is None:
            self.cam_pose = self.pose_generator.random_sample()
        else:
            try:
                self.cam_pose = self.pose_generator.random_sample(rng=rng)
            except TypeError:
                # Compatibility fallback for the unmodified RandomCamPose.
                legacy_state = np.random.get_state()
                try:
                    np.random.seed(
                        int(rng.integers(0, 2**32 - 1, dtype=np.uint32))
                    )
                    self.cam_pose = self.pose_generator.random_sample()
                finally:
                    np.random.set_state(legacy_state)

        set_camera_pose(self.camera, **self.cam_pose)
        if rng is None:
            self.current_camera_fov = float(self.camera_fov_range[0])
        else:
            self.current_camera_fov = float(
                rng.uniform(*self.camera_fov_range)
            )
        self._cam_params["x_fov"] = self.current_camera_fov
        self._cam_params.update()
        x_fov_param = self._cam_params["x_fov"]
        try:
            self.cam_Xfov = float(x_fov_param[0])
        except (TypeError, IndexError):
            self.cam_Xfov = float(x_fov_param)

        size = self._cam_params["film.size"]
        width, height = int(size[0]), int(size[1])
        self.img_size = (width, height)
        self.cam_intri_mat = fov_to_intrinsic_mat(
            self.cam_Xfov, "x", width, height
        ).astype(np.float32)
        self.cam_extri_mat = get_extrinsic_matrix(
            **self.cam_pose
        ).astype(np.float32)

    def tracing_refraction(self):
        """Trace two-interface refraction to the selected background."""
        rotation = self.cam_extri_mat[:, :3]
        translation = self.cam_extri_mat[:, 3]
        width, height = self.img_size
        rays_o, rays_d = gen_rays(
            self.cam_intri_mat,
            rotation,
            translation,
            width,
            height,
        )
        rays_o = rays_o.reshape(-1, 3).astype(np.float32)
        rays_d = rays_d.reshape(-1, 3).astype(np.float32)

        if self.background_mode == "plane":
            exit_trace = self._trace_exit_rays(rays_o, rays_d)
            hit = self._intersect_background_plane(
                exit_trace["exit_o"], exit_trace["exit_d"]
            )
            return self._scatter_background_trace(
                rays_o, exit_trace, hit, object_id=1
            )

        exit_trace = self._trace_exit_rays(rays_o, rays_d)
        hit = self._intersect_background_scene(
            exit_trace["exit_o"], exit_trace["exit_d"]
        )
        return self._scatter_background_trace(rays_o, exit_trace, hit)

    def _scatter_background_trace(
        self, rays_o, exit_trace, hit, object_id=None
    ):
        """Scatter packed two-refraction hits back to the image-ray order."""
        ray_count = rays_o.shape[0]
        final_idx = exit_trace["ray_idx"]

        def scatter(values, width=None, dtype=np.float32):
            shape = (ray_count,) if width is None else (ray_count, width)
            target = np.zeros(shape, dtype=dtype)
            target[final_idx] = values
            return target

        bg_hit_valid = scatter(hit["valid"], dtype=bool)
        path_length = np.zeros(ray_count, dtype=np.float32)
        if final_idx.size:
            camera_to_entry = np.linalg.norm(
                exit_trace["entry_o"] - rays_o[final_idx], axis=1
            )
            inside_object = np.linalg.norm(
                exit_trace["exit_o"] - exit_trace["entry_o"], axis=1
            )
            exit_to_background = np.linalg.norm(
                hit["xyz"] - exit_trace["exit_o"], axis=1
            )
            packed_length = camera_to_entry + inside_object + exit_to_background
            packed_length[~hit["valid"]] = 0.0
            path_length[final_idx] = packed_length.astype(np.float32)

        if object_id is None:
            packed_object_id = hit["object_id"]
        else:
            packed_object_id = np.where(
                hit["valid"], int(object_id), 0
            ).astype(np.int32)
        return {
            "obj_mask": exit_trace["obj_mask"],
            "twice_mask": exit_trace["twice_mask"],
            "normal": exit_trace["normal"],
            "bg_hit_xyz": scatter(hit["xyz"], width=3),
            "bg_hit_normal": scatter(hit["normal"], width=3),
            "bg_hit_uv": scatter(hit["uv"], width=2),
            "bg_object_id": scatter(packed_object_id, dtype=np.int32),
            "bg_hit_albedo": scatter(hit["albedo"], width=3),
            "bg_hit_valid": bg_hit_valid,
            "bg_path_length": path_length,
        }

    def _intersect_background_plane(self, rays_o, rays_d):
        """Intersect packed exit rays with the sampled finite tilted plane."""
        ray_count = rays_o.shape[0]
        centre, tangent_u, tangent_v, normal = self._background_plane_frame()
        denominator = rays_d @ normal
        numerator = (centre[None, :] - rays_o) @ normal
        distance = np.zeros(ray_count, dtype=np.float32)
        nonparallel = np.abs(denominator) > 1e-7
        distance[nonparallel] = numerator[nonparallel] / denominator[nonparallel]
        xyz = rays_o + distance[:, None] * rays_d
        relative = xyz - centre[None, :]
        local_u = relative @ tangent_u
        local_v = relative @ tangent_v
        valid = (
            nonparallel
            & np.isfinite(distance)
            & (distance > 1e-6)
            & (np.abs(local_u) <= self.background_half_size[0])
            & (np.abs(local_v) <= self.background_half_size[1])
        )
        xyz[~valid] = 0.0
        uv = np.zeros((ray_count, 2), dtype=np.float32)
        uv[:, 0] = 0.5 * (local_u / self.background_half_size[0] + 1.0)
        uv[:, 1] = 0.5 * (local_v / self.background_half_size[1] + 1.0)
        uv[~valid] = 0.0
        normals = np.tile(normal[None, :], (ray_count, 1)).astype(np.float32)
        normals[~valid] = 0.0
        return {
            "valid": valid,
            "xyz": xyz.astype(np.float32),
            "normal": normals,
            "uv": uv,
            "object_id": np.where(valid, 1, 0).astype(np.int32),
            "albedo": np.zeros((ray_count, 3), dtype=np.float32),
        }

    @staticmethod
    def _mi_xyz_to_numpy(value):
        return np.stack(
            [
                np.asarray(value.x, dtype=np.float32),
                np.asarray(value.y, dtype=np.float32),
                np.asarray(value.z, dtype=np.float32),
            ],
            axis=-1,
        )

    def _trace_exit_rays(self, rays_o, rays_d):
        """Use the public Mitsuba tracer bounce API to expose exit rays."""
        required = ("first_bounce", "second_bounce")
        if any(not hasattr(self.tracer, name) for name in required):
            raise RuntimeError(
                "Tilted planar and 3D backgrounds require the Mitsuba tracer "
                "with first_bounce() and second_bounce()."
            )
        mi_rays_o = mi.Point3f(rays_o[:, 0], rays_o[:, 1], rays_o[:, 2])
        mi_rays_d = mi.Vector3f(rays_d[:, 0], rays_d[:, 1], rays_d[:, 2])
        first = self.tracer.first_bounce(mi_rays_o, mi_rays_d)
        second = self.tracer.second_bounce(
            first["trans_ray"]["rays_o"], first["trans_ray"]["rays_d"]
        )

        second_relative_idx = second["ray_idx"]
        first_valid_hit_idx = first["trans_ray"]["valid_idx"]
        first_trans_to_full = dr.gather(
            mi.UInt32, first["ray_idx"], first_valid_hit_idx
        )
        final_idx = np.asarray(
            dr.gather(mi.UInt32, first_trans_to_full, second_relative_idx),
            dtype=np.int64,
        )
        entry_rays = first["trans_ray"]["rays_o"]
        entry_o = mi.Point3f(
            dr.gather(mi.Float, entry_rays.x, second_relative_idx),
            dr.gather(mi.Float, entry_rays.y, second_relative_idx),
            dr.gather(mi.Float, entry_rays.z, second_relative_idx),
        )
        exit_o = second["trans_ray"]["rays_o"]
        exit_d = second["trans_ray"]["rays_d"]

        ray_count = rays_o.shape[0]
        twice_mask = np.zeros(ray_count, dtype=bool)
        twice_mask[final_idx] = True
        normal = np.zeros((ray_count, 3), dtype=np.float32)
        first_idx = np.asarray(first["ray_idx"], dtype=np.int64)
        normal[first_idx] = self._mi_xyz_to_numpy(first["normal"])
        return {
            "exit_o": self._mi_xyz_to_numpy(exit_o),
            "exit_d": self._mi_xyz_to_numpy(exit_d),
            "entry_o": self._mi_xyz_to_numpy(entry_o),
            "ray_idx": final_idx,
            "obj_mask": np.asarray(first["mask"], dtype=bool),
            "twice_mask": twice_mask,
            "normal": normal,
        }

    def _intersect_background_scene(self, rays_o, rays_d):
        """Intersect packed exit rays and retain the nearest 3D object hit."""
        ray_count = rays_o.shape[0]
        best_t = np.full(ray_count, np.inf, dtype=np.float32)
        best_xyz = np.zeros((ray_count, 3), dtype=np.float32)
        best_normal = np.zeros((ray_count, 3), dtype=np.float32)
        best_uv = np.zeros((ray_count, 2), dtype=np.float32)
        best_id = np.zeros(ray_count, dtype=np.int32)
        best_albedo = np.zeros((ray_count, 3), dtype=np.float32)
        if ray_count == 0:
            return {
                "valid": np.zeros(0, dtype=bool),
                "xyz": best_xyz,
                "normal": best_normal,
                "uv": best_uv,
                "object_id": best_id,
                "albedo": best_albedo,
            }

        ray = mi.Ray3f(
            o=mi.Point3f(rays_o[:, 0], rays_o[:, 1], rays_o[:, 2]),
            d=mi.Vector3f(rays_d[:, 0], rays_d[:, 1], rays_d[:, 2]),
        )
        for scene, object_id, albedo in zip(
            self._background_object_scenes,
            self.background_object_ids,
            self.background_object_albedos,
        ):
            si = scene.ray_intersect(ray)
            valid = np.asarray(si.is_valid(), dtype=bool)
            distance = np.asarray(si.t, dtype=np.float32)
            take = valid & np.isfinite(distance) & (distance > 1e-6) & (
                distance < best_t
            )
            if not np.any(take):
                continue
            best_t[take] = distance[take]
            best_xyz[take] = self._mi_xyz_to_numpy(si.p)[take]
            best_normal[take] = self._mi_xyz_to_numpy(si.sh_frame.n)[take]
            uv = np.stack(
                [
                    np.asarray(si.uv.x, dtype=np.float32),
                    np.asarray(si.uv.y, dtype=np.float32),
                ],
                axis=-1,
            )
            best_uv[take] = uv[take]
            best_id[take] = int(object_id)
            best_albedo[take] = albedo

        return {
            "valid": np.isfinite(best_t),
            "xyz": best_xyz,
            "normal": best_normal,
            "uv": best_uv,
            "object_id": best_id,
            "albedo": best_albedo,
        }

    def world_points_to_pixel(self, world_points):
        """Project background-plane world points into clean-image coordinates."""
        world_points = np.asarray(world_points, dtype=np.float32)
        rotation = self.cam_extri_mat[:, :3]
        translation = self.cam_extri_mat[:, 3]
        camera_points = (
            rotation @ world_points.T
        ).T + translation[None, :]
        homogeneous = (self.cam_intri_mat @ camera_points.T).T
        depth = homogeneous[:, 2].copy()
        pixel_xy = homogeneous[:, :2] / (depth[:, None] + 1e-12)
        return pixel_xy.astype(np.float32), depth.astype(np.float32)

    @staticmethod
    def fetch_displacement(displacement, image):
        """Backward warp ``P(x) = B(x + Phi(x))`` in pixel units."""
        height, width = image.shape[:2]
        yy, xx = np.meshgrid(
            np.arange(height, dtype=np.float32),
            np.arange(width, dtype=np.float32),
            indexing="ij",
        )
        map_x = xx + displacement[..., 0]
        map_y = yy + displacement[..., 1]
        return cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    @staticmethod
    def _axis_angle_to_matrix(axis, angle_deg):
        axis = np.asarray(axis, dtype=np.float32)
        axis /= np.linalg.norm(axis) + 1e-12
        x, y, z = axis
        angle = np.deg2rad(float(angle_deg))
        cosine = np.cos(angle)
        sine = np.sin(angle)
        one_minus_cosine = 1.0 - cosine
        return np.asarray(
            [
                [
                    cosine + x * x * one_minus_cosine,
                    x * y * one_minus_cosine - z * sine,
                    x * z * one_minus_cosine + y * sine,
                ],
                [
                    y * x * one_minus_cosine + z * sine,
                    cosine + y * y * one_minus_cosine,
                    y * z * one_minus_cosine - x * sine,
                ],
                [
                    z * x * one_minus_cosine - y * sine,
                    z * y * one_minus_cosine + x * sine,
                    cosine + z * z * one_minus_cosine,
                ],
            ],
            dtype=np.float32,
        )

    def sample_object_trajectory(self, rng):
        motion_conf = self.conf.get("ObjectMotion", {})
        trans_min = np.asarray(
            motion_conf.get("translation_min", [-0.12, -0.12, -0.04]),
            dtype=np.float32,
        )
        trans_max = np.asarray(
            motion_conf.get("translation_max", [0.12, 0.12, 0.04]),
            dtype=np.float32,
        )
        if trans_min.shape != (3,) or trans_max.shape != (3,):
            raise ValueError("ObjectMotion translations must be 3-vectors")
        if np.any(trans_min > trans_max):
            raise ValueError("ObjectMotion.translation_min exceeds translation_max")

        start_translation = rng.uniform(trans_min, trans_max)
        end_translation = rng.uniform(trans_min, trans_max)
        min_distance = float(
            motion_conf.get("min_translation_distance", 0.0)
        )
        for _ in range(20):
            if np.linalg.norm(end_translation - start_translation) >= min_distance:
                break
            end_translation = rng.uniform(trans_min, trans_max)

        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis) + 1e-12
        start_angle = float(
            rng.uniform(
                *motion_conf.get("start_angle_range", [-15.0, 15.0])
            )
        )
        delta_angle = float(
            rng.uniform(
                *motion_conf.get("delta_angle_range", [-45.0, 45.0])
            )
        )

        trajectory = []
        for frame_idx in range(self.frames_per_sequence):
            tau = (
                frame_idx / (self.frames_per_sequence - 1)
                if self.frames_per_sequence > 1
                else 0.0
            )
            translation = (
                (1.0 - tau) * start_translation + tau * end_translation
            )
            rotation = self._axis_angle_to_matrix(
                axis, start_angle + tau * delta_angle
            )

            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = rotation
            # Rotate around the canonical mesh centre, then translate.
            pose[:3, 3] = (
                self.mesh_pivot
                + translation
                - rotation @ self.mesh_pivot
            )
            trajectory.append(pose)
        return trajectory

    @staticmethod
    def _render_array(scene, camera, render_seed):
        return np.asarray(
            mi.render(scene, sensor=camera, seed=int(render_seed)),
            dtype=np.float32,
        )

    def single_render(
        self, background, background_depth, background_depth_valid, render_seed
    ):
        """Render one frame and derive all linear-RGB supervision targets."""
        no_ref_pass = self._render_array(
            self.no_ref_scene.scene, self.camera, render_seed
        )
        image_no_ref = no_ref_pass[..., :3]

        object_only_no_ref_pass = self._render_array(
            self.object_only_no_ref_scene.scene, self.camera, render_seed
        )
        image_object_only_no_ref = object_only_no_ref_pass[..., :3]

        # For exactly zero reflection, reuse the counterfactual render. This
        # guarantees C_R == 0 and avoids an unnecessary render.
        if self.current_reflection_scale == 0.0:
            full_pass = no_ref_pass
            object_only_pass = object_only_no_ref_pass
        else:
            full_pass = self._render_array(
                self.scene.scene, self.camera, render_seed
            )
            object_only_pass = self._render_array(
                self.object_only_scene.scene, self.camera, render_seed
            )
        if full_pass.shape[-1] < 7:
            raise RuntimeError(
                "Expected RGB + shading-normal + depth AOV channels, got "
                f"shape {full_pass.shape}. Check _wrap_geometry_aovs()."
            )
        image = full_pass[..., :3]
        image_object_only = object_only_pass[..., :3]
        normal_full = np.asarray(full_pass[..., 3:6], dtype=np.float32)
        depth_full = np.asarray(full_pass[..., 6], dtype=np.float32)

        transmission = self._render_array(
            self.transmission_scene, self.camera, render_seed
        )[..., :3]
        transmission = np.clip(transmission, 0.0, 1.0)

        trace = self.tracing_refraction()
        height, width = image.shape[:2]
        object_mask = np.asarray(trace["obj_mask"], dtype=bool).reshape(
            height, width
        )
        twice_mask = np.asarray(trace["twice_mask"], dtype=bool)

        bg_hit_xyz = np.asarray(trace["bg_hit_xyz"], dtype=np.float32)
        bg_hit_valid_flat = np.asarray(trace["bg_hit_valid"], dtype=bool)
        target_xy, target_depth = self.world_points_to_pixel(bg_hit_xyz)
        yy, xx = np.meshgrid(
            np.arange(height, dtype=np.float32),
            np.arange(width, dtype=np.float32),
            indexing="ij",
        )
        source_xy = np.stack([xx, yy], axis=-1).reshape(-1, 2)
        # Finite-plane support is already part of bg_hit_valid. This remains
        # correct after tilting because it is evaluated in local plane axes.
        plane_inside = np.ones(bg_hit_xyz.shape[0], dtype=bool)
        inside = (
            bg_hit_valid_flat
            &
            (target_depth > 1e-6)
            & plane_inside
            & (target_xy[:, 0] >= 0.0)
            & (target_xy[:, 0] <= width - 1)
            & (target_xy[:, 1] >= 0.0)
            & (target_xy[:, 1] <= height - 1)
        )
        # A 3D hit can project behind another clean-background surface. In that
        # case there is no clean-image pixel that contains the actual hit. Keep
        # the 3D GT, but exclude it from 2D Phi/P backward sampling.
        camera_center = -self.cam_extri_mat[:, :3].T @ self.cam_extri_mat[:, 3]
        hit_direct_distance = np.linalg.norm(
            bg_hit_xyz - camera_center[None, :], axis=1
        ).astype(np.float32)
        if self.background_mode == "scene":
            map_x = target_xy[:, 0].reshape(height, width).astype(np.float32)
            map_y = target_xy[:, 1].reshape(height, width).astype(np.float32)
            clean_depth_at_hit = cv2.remap(
                np.asarray(background_depth, dtype=np.float32),
                map_x,
                map_y,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            ).reshape(-1)
            clean_valid_at_hit = cv2.remap(
                np.asarray(background_depth_valid, dtype=np.uint8),
                map_x,
                map_y,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            ).reshape(-1).astype(bool)
            tolerance = self.clean_visibility_tolerance * np.maximum(
                1.0, hit_direct_distance
            )
            clean_visible = (
                inside
                & clean_valid_at_hit
                & (np.abs(clean_depth_at_hit - hit_direct_distance) <= tolerance)
            )
        else:
            clean_visible = inside.copy()

        hit_direct_distance_output = hit_direct_distance.copy()
        hit_direct_distance_output[~bg_hit_valid_flat] = 0.0

        projected_valid_flat = twice_mask & inside
        phi_valid_flat = projected_valid_flat & clean_visible
        phi_flat = target_xy - source_xy
        phi_flat[~phi_valid_flat] = 0.0
        phi = phi_flat.reshape(height, width, 2).astype(np.float32)
        phi_valid = phi_valid_flat.reshape(height, width)
        source_coordinates_flat = target_xy.copy()
        source_coordinates_flat[~phi_valid_flat] = 0.0
        source_coordinates = source_coordinates_flat.reshape(
            height, width, 2
        ).astype(np.float32)
        projected_source_flat = target_xy.copy()
        projected_source_flat[~projected_valid_flat] = 0.0
        projected_source = projected_source_flat.reshape(
            height, width, 2
        ).astype(np.float32)

        warped_background = self.fetch_displacement(phi, background)
        # The AOV records the primary visible surface. Keep that full-scene
        # result (transparent object or finite background plane), and derive a
        # separate object-only copy with the geometry-specific tracer mask.
        support_mask = object_mask.copy()
        normal_full_valid = np.all(np.isfinite(normal_full), axis=-1) & (
            np.linalg.norm(normal_full, axis=-1) > 1e-6
        )
        normal_full[~normal_full_valid] = 0.0
        depth_full_valid = np.isfinite(depth_full) & (depth_full > 0.0)
        depth_full[~depth_full_valid] = 0.0

        normal_object = normal_full.copy()
        normal_object_valid = support_mask & normal_full_valid
        normal_object[~normal_object_valid] = 0.0
        depth_object = depth_full.copy()
        depth_object_valid = support_mask & depth_full_valid
        depth_object[~depth_object_valid] = 0.0

        # The white-environment render is already exactly one outside the
        # object. Do not overwrite those pixels with the centre-ray mask: the
        # rendered pass contains the correct subpixel silhouette coverage.
        # Rays whose centre sample hits the object but lacks a valid
        # two-interface path receive no background transmission.
        transmission[support_mask & ~phi_valid] = 0.0

        transmission_strength = transmission.max(axis=-1)
        alpha = 1.0 - transmission_strength
        transmission_rgb = np.ones_like(transmission)
        stable = transmission_strength > self.decomp_eps
        transmission_rgb[stable] = (
            transmission[stable] / transmission_strength[stable, None]
        )
        transmission_rgb[support_mask & ~stable] = 0.0

        # Direct renderer-native black-reference passes.  In these two passes
        # the background geometry is present at the exact same location but has
        # zero emission/reflectance.  Consequently reflection, highlights, TIR,
        # and other object/illumination terms are preserved, while transmitted
        # background appearance is absent. Under the standard matting model,
        # rendering against B=0 gives I_black = alpha * F_std. No single-flow
        # warp is subtracted to construct this premultiplied foreground.
        foreground_premultiplied = np.asarray(
            image_object_only, dtype=np.float32
        ).copy()
        foreground_material_premultiplied = np.asarray(
            image_object_only_no_ref, dtype=np.float32
        ).copy()

        # Remove any directly visible environment in uncovered pixels (mainly
        # relevant when a sparse 3D background manifest has gaps).  Alpha comes
        # from the multisample transmission render, so this retains antialiased
        # silhouette coverage instead of applying the single centre-ray mask.
        foreground_valid = alpha > self.decomp_eps
        foreground_premultiplied[~foreground_valid] = 0.0
        foreground_material_premultiplied[~foreground_valid] = 0.0

        # R remains an auxiliary physical decomposition inside the directly
        # rendered foreground.  When reflection_scale is zero the two passes
        # are the exact same array, hence C_R is exactly zero.
        reflection_contribution = (
            foreground_premultiplied - foreground_material_premultiplied
        )

        # Standard straight-foreground matting with a refracted background:
        #
        #   I ~= alpha * F_std + (1 - alpha) * P
        #   P(x) = B(x + Phi(x)),  C_F := alpha * F_std = I_black
        #
        # F_std is the conventional straight foreground color used by image
        # matting. It is not the observed image on a white background. The
        # reconstruction is not forced by defining C_F as an image residual;
        # its error measures the single-Phi/background model gap.
        background_weight = 1.0 - alpha[..., None]
        foreground = np.zeros_like(image)
        foreground[foreground_valid] = (
            foreground_premultiplied[foreground_valid]
            / alpha[foreground_valid, None]
        )

        foreground_material = np.zeros_like(image)
        foreground_material[foreground_valid] = (
            foreground_material_premultiplied[foreground_valid]
            / alpha[foreground_valid, None]
        )

        reconstructed = (
            alpha[..., None] * foreground
            + background_weight * warped_background
        )
        reconstruction_error = np.abs(image - reconstructed)
        factorization_error = np.abs(
            transmission
            - (1.0 - alpha[..., None]) * transmission_rgb
        )

        max_reconstruction_error = float(reconstruction_error.max())
        max_factorization_error = float(factorization_error.max())
        if max_factorization_error > self.validation_tolerance:
            warnings.warn(
                "Transmission factorization error exceeds tolerance: "
                f"{max_factorization_error:.6e}",
                RuntimeWarning,
            )

        return {
            "I": image,
            "I_no_ref": image_no_ref,
            "I_black_reference": image_object_only,
            "I_black_reference_no_ref": image_object_only_no_ref,
            "P": warped_background,
            "Phi": phi,
            "Phi_src": source_coordinates,
            "phi_valid": phi_valid.astype(np.uint8),
            "Bg_hit_src": projected_source,
            "Bg_projected_valid": projected_valid_flat.reshape(
                height, width
            ).astype(np.uint8),
            "Bg_clean_visible": clean_visible.reshape(height, width).astype(
                np.uint8
            ),
            "Bg_hit_xyz": bg_hit_xyz.reshape(height, width, 3),
            "Bg_hit_normal": np.asarray(
                trace["bg_hit_normal"], dtype=np.float32
            ).reshape(height, width, 3),
            "Bg_hit_uv": np.asarray(
                trace["bg_hit_uv"], dtype=np.float32
            ).reshape(height, width, 2),
            "Bg_object_id": np.asarray(
                trace["bg_object_id"], dtype=np.int32
            ).reshape(height, width),
            "Bg_hit_albedo": np.asarray(
                trace["bg_hit_albedo"], dtype=np.float32
            ).reshape(height, width, 3),
            "Bg_hit_valid": bg_hit_valid_flat.reshape(height, width).astype(
                np.uint8
            ),
            "Bg_hit_distance": hit_direct_distance_output.reshape(
                height, width
            ).astype(np.float32),
            "N_refract": np.asarray(
                trace["bg_hit_normal"], dtype=np.float32
            ).reshape(height, width, 3),
            "D_refract": np.asarray(
                trace["bg_path_length"], dtype=np.float32
            ).reshape(height, width),
            "object_mask": object_mask.astype(np.uint8),
            "support_mask": support_mask.astype(np.uint8),
            "A": transmission,
            "alpha": alpha.astype(np.float32),
            "T": transmission_rgb,
            "C_F": foreground_premultiplied,
            "F": foreground,
            "F_valid": foreground_valid.astype(np.uint8),
            "C_F_material": foreground_material_premultiplied,
            "F_material": foreground_material,
            "C_R": reflection_contribution,
            # Canonical N/D now cover every primary surface visible to the
            # camera, including the finite clean-background plane.
            "N": normal_full.astype(np.float32),
            "N_valid": normal_full_valid.astype(np.uint8),
            "D": depth_full.astype(np.float32),
            "D_valid": depth_full_valid.astype(np.uint8),
            # Preserve the v6 object-only geometry targets explicitly.
            "N_object": normal_object.astype(np.float32),
            "N_object_valid": normal_object_valid.astype(np.uint8),
            "D_object": depth_object.astype(np.float32),
            "D_object_valid": depth_object_valid.astype(np.uint8),
            "B": np.asarray(background, dtype=np.float32),
            "object_pose": self.current_object_pose.copy(),
            "reconstruction_error": reconstruction_error,
            "max_reconstruction_error": max_reconstruction_error,
            "max_factorization_error": max_factorization_error,
        }

    @staticmethod
    def _write_exr(path, array):
        mi.util.write_bitmap(path, np.asarray(array, dtype=np.float32))

    @staticmethod
    def _write_mask(path, mask):
        imageio.imwrite(path, (255 * np.asarray(mask, dtype=np.uint8)))

    @staticmethod
    def _write_normal_preview(path, normal, valid_mask):
        """Write an 8-bit view of signed world-space shading normals."""
        normal = np.asarray(normal, dtype=np.float32)
        valid = np.asarray(valid_mask, dtype=bool)
        encoded = np.rint(
            np.clip(0.5 * (normal + 1.0), 0.0, 1.0) * 255.0
        ).astype(np.uint8)
        encoded[~valid] = 0
        if not cv2.imwrite(path, np.ascontiguousarray(encoded[..., ::-1])):
            raise IOError(f"Could not write normal preview: {path}")

    @staticmethod
    def _write_depth_preview(path, depth, valid_mask):
        """Write a near-to-far false-color preview; EXR/NPY stay metric."""
        depth = np.asarray(depth, dtype=np.float32)
        valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(depth)
        normalized = np.zeros(depth.shape, dtype=np.uint8)
        if np.any(valid):
            valid_depth = depth[valid]
            near = float(valid_depth.min())
            far = float(valid_depth.max())
            if far > near + 1e-8:
                normalized[valid] = np.rint(
                    (depth[valid] - near) / (far - near) * 255.0
                ).astype(np.uint8)
            else:
                normalized[valid] = 128
        preview = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        preview[~valid] = 0
        if not cv2.imwrite(path, preview):
            raise IOError(f"Could not write depth preview: {path}")

    @staticmethod
    def _write_id_preview(path, object_id, valid_mask):
        """Write a deterministic color preview of integer background IDs."""
        object_id = np.asarray(object_id, dtype=np.int32)
        valid = np.asarray(valid_mask, dtype=bool)
        hsv = np.zeros((*object_id.shape, 3), dtype=np.uint8)
        hsv[..., 0] = np.mod(object_id * 37, 180).astype(np.uint8)
        hsv[..., 1] = np.where(valid, 220, 0).astype(np.uint8)
        hsv[..., 2] = np.where(valid, 255, 0).astype(np.uint8)
        preview = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        if not cv2.imwrite(path, preview):
            raise IOError(f"Could not write object-ID preview: {path}")

    @classmethod
    def _write_rgb_preview(cls, path, linear_rgb, valid_mask=None):
        """Write a tone-mapped diagnostic PNG; EXR remains canonical."""
        preview = cls._preview_rgb(linear_rgb)
        if valid_mask is not None:
            valid = np.asarray(valid_mask, dtype=bool)
            if valid.shape != preview.shape[:2]:
                raise ValueError("valid_mask must match the RGB image size")
            preview[~valid] = 0
        if not cv2.imwrite(path, np.ascontiguousarray(preview[..., ::-1])):
            raise IOError(f"Could not write RGB preview: {path}")

    @staticmethod
    def _write_flow_png(path, flow, valid_mask):
        """Write a reversible 16-bit RGB PNG containing (dx, dy, valid).

        R and G encode pixel-space displacement over the complete range that
        can occur between two in-frame pixels. B is 65535 for valid vectors.
        Invalid pixels are stored as (0, 0, 0). The float32 ``Phi.npy`` remains
        the canonical, lossless target; this PNG is a compact interchange copy.
        """
        flow = np.asarray(flow, dtype=np.float32)
        valid = np.asarray(valid_mask, dtype=bool)
        if flow.ndim != 3 or flow.shape[-1] != 2:
            raise ValueError("flow must have shape (height, width, 2)")
        if valid.shape != flow.shape[:2]:
            raise ValueError("valid_mask must match the flow image size")

        height, width = flow.shape[:2]
        x_range = float(max(width - 1, 1))
        y_range = float(max(height - 1, 1))
        encoded = np.empty((height, width, 3), dtype=np.uint16)
        encoded[..., 0] = np.rint(
            np.clip(0.5 * (flow[..., 0] / x_range + 1.0), 0.0, 1.0)
            * 65535.0
        ).astype(np.uint16)
        encoded[..., 1] = np.rint(
            np.clip(0.5 * (flow[..., 1] / y_range + 1.0), 0.0, 1.0)
            * 65535.0
        ).astype(np.uint16)
        encoded[..., 2] = np.where(valid, 65535, 0).astype(np.uint16)
        encoded[~valid] = 0

        # OpenCV preserves 16-bit RGB PNGs but expects BGR channel order.
        encoded_bgr = np.ascontiguousarray(encoded[..., ::-1])
        if not cv2.imwrite(path, encoded_bgr):
            raise IOError(f"Could not write flow PNG: {path}")

    @staticmethod
    def _preview_rgb(linear_rgb):
        """Tone-map a linear RGB render for an 8-bit diagnostic overlay."""
        linear = np.maximum(np.asarray(linear_rgb, dtype=np.float32), 0.0)
        mapped = 1.0 - np.exp(-linear)
        srgb = np.where(
            mapped <= 0.0031308,
            12.92 * mapped,
            1.055 * np.power(mapped, 1.0 / 2.4) - 0.055,
        )
        return np.rint(np.clip(srgb, 0.0, 1.0) * 255.0).astype(np.uint8)

    @classmethod
    def _write_flow_arrows(
        cls,
        path,
        flow,
        valid_mask,
        base_image,
        stride=4,
        canvas_scale=4,
        min_magnitude_px=0.25,
        max_arrows=1200,
    ):
        """Draw exact backward correspondences as arrows.

        Each arrow starts at output pixel ``(x, y)`` and ends at the clean
        background pixel ``(x + dx, y + dy)`` sampled by that output pixel.
        Green dots are output pixels and red dots are sampled background
        pixels. The canvas is enlarged, but vector endpoints are not scaled
        relative to the image coordinate system.
        """
        flow = np.asarray(flow, dtype=np.float32)
        valid = np.asarray(valid_mask, dtype=bool)
        if flow.ndim != 3 or flow.shape[-1] != 2:
            raise ValueError("flow must have shape (height, width, 2)")
        if valid.shape != flow.shape[:2]:
            raise ValueError("valid_mask must match the flow image size")

        height, width = flow.shape[:2]
        preview = cls._preview_rgb(base_image)
        if preview.shape[:2] != (height, width):
            raise ValueError("base_image must match the flow image size")

        scale = int(canvas_scale)
        canvas = cv2.resize(
            np.ascontiguousarray(preview[..., ::-1]),
            (width * scale, height * scale),
            interpolation=cv2.INTER_NEAREST,
        )
        # Darken the render so arrows remain legible over bright highlights.
        canvas = np.rint(canvas.astype(np.float32) * 0.48).astype(np.uint8)

        yy, xx = np.nonzero(valid)
        keep = (xx % int(stride) == 0) & (yy % int(stride) == 0)
        yy, xx = yy[keep], xx[keep]
        if yy.size:
            magnitude = np.linalg.norm(flow[yy, xx], axis=1)
            keep = magnitude >= float(min_magnitude_px)
            yy, xx = yy[keep], xx[keep]
        if yy.size > int(max_arrows):
            indices = np.linspace(
                0, yy.size - 1, int(max_arrows), dtype=np.int64
            )
            yy, xx = yy[indices], xx[indices]

        arrow_color = (255, 255, 0)  # cyan in BGR
        origin_color = (0, 255, 0)
        target_color = (0, 0, 255)
        thickness = max(1, scale // 2)
        radius = max(1, scale // 2)
        for y, x in zip(yy.tolist(), xx.tolist()):
            dx, dy = flow[y, x]
            start = (
                int(round((x + 0.5) * scale)),
                int(round((y + 0.5) * scale)),
            )
            target = (
                int(round((x + dx + 0.5) * scale)),
                int(round((y + dy + 0.5) * scale)),
            )
            cv2.arrowedLine(
                canvas,
                start,
                target,
                arrow_color,
                thickness,
                cv2.LINE_AA,
                0,
                0.22,
            )
            cv2.circle(canvas, start, radius, origin_color, -1, cv2.LINE_AA)
            cv2.circle(canvas, target, radius, target_color, -1, cv2.LINE_AA)

        if not cv2.imwrite(path, canvas):
            raise IOError(f"Could not write arrow visualization: {path}")

    @classmethod
    def _write_correspondence_pairs(
        cls,
        path,
        flow,
        valid_mask,
        rendered_image,
        clean_background,
        stride=4,
        canvas_scale=4,
        min_magnitude_px=0.25,
        max_arrows=1200,
    ):
        """Connect rendered pixels in I to sampled pixels in clean B."""
        flow = np.asarray(flow, dtype=np.float32)
        valid = np.asarray(valid_mask, dtype=bool)
        if flow.ndim != 3 or flow.shape[-1] != 2:
            raise ValueError("flow must have shape (height, width, 2)")
        height, width = flow.shape[:2]
        if valid.shape != (height, width):
            raise ValueError("valid_mask must match the flow image size")

        left = cls._preview_rgb(rendered_image)
        right = cls._preview_rgb(clean_background)
        if left.shape[:2] != (height, width) or right.shape[:2] != (
            height,
            width,
        ):
            raise ValueError("I and B must match the flow image size")

        scale = int(canvas_scale)
        gap = max(8, 6 * scale)
        panel_width = width * scale
        canvas = np.zeros(
            (height * scale, panel_width * 2 + gap, 3), dtype=np.uint8
        )
        canvas[:, :panel_width] = cv2.resize(
            np.ascontiguousarray(left[..., ::-1]),
            (panel_width, height * scale),
            interpolation=cv2.INTER_NEAREST,
        )
        canvas[:, panel_width + gap :] = cv2.resize(
            np.ascontiguousarray(right[..., ::-1]),
            (panel_width, height * scale),
            interpolation=cv2.INTER_NEAREST,
        )
        canvas = np.rint(canvas.astype(np.float32) * 0.55).astype(np.uint8)

        label_scale = max(0.35, scale / 10.0)
        label_y = max(14, 4 * scale)
        cv2.putText(
            canvas,
            "I: rendered pixel (x,y)",
            (5, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            label_scale,
            (255, 255, 255),
            max(1, scale // 2),
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "B: clean source pixel (u,v)",
            (panel_width + gap + 5, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            label_scale,
            (255, 255, 255),
            max(1, scale // 2),
            cv2.LINE_AA,
        )

        yy, xx = np.nonzero(valid)
        keep = (xx % int(stride) == 0) & (yy % int(stride) == 0)
        yy, xx = yy[keep], xx[keep]
        if yy.size:
            magnitude = np.linalg.norm(flow[yy, xx], axis=1)
            keep = magnitude >= float(min_magnitude_px)
            yy, xx = yy[keep], xx[keep]
        if yy.size > int(max_arrows):
            indices = np.linspace(
                0, yy.size - 1, int(max_arrows), dtype=np.int64
            )
            yy, xx = yy[indices], xx[indices]

        thickness = max(1, scale // 2)
        radius = max(1, scale // 2)
        for y, x in zip(yy.tolist(), xx.tolist()):
            dx, dy = flow[y, x]
            start = (
                int(round((x + 0.5) * scale)),
                int(round((y + 0.5) * scale)),
            )
            target = (
                panel_width
                + gap
                + int(round((x + dx + 0.5) * scale)),
                int(round((y + dy + 0.5) * scale)),
            )
            cv2.arrowedLine(
                canvas,
                start,
                target,
                (255, 255, 0),
                thickness,
                cv2.LINE_AA,
                0,
                0.012,
            )
            cv2.circle(canvas, start, radius, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, target, radius, (0, 0, 255), -1, cv2.LINE_AA)

        if not cv2.imwrite(path, canvas):
            raise IOError(f"Could not write correspondence pairs: {path}")

    def single_save(self, render_result, basename):
        """Save one frame. Signed residual-like tensors remain unclipped EXRs."""
        prefix = osp.join(self.output_folder, basename)
        self._write_exr(prefix + "_I.exr", render_result["I"])
        np.save(prefix + "_Phi.npy", render_result["Phi"])
        np.save(prefix + "_Phi_src.npy", render_result["Phi_src"])
        np.save(prefix + "_Bg_hit_src.npy", render_result["Bg_hit_src"])
        np.save(prefix + "_Bg_hit_xyz.npy", render_result["Bg_hit_xyz"])
        np.save(prefix + "_Bg_hit_normal.npy", render_result["Bg_hit_normal"])
        np.save(prefix + "_Bg_hit_uv.npy", render_result["Bg_hit_uv"])
        np.save(prefix + "_Bg_object_id.npy", render_result["Bg_object_id"])
        np.save(prefix + "_Bg_hit_albedo.npy", render_result["Bg_hit_albedo"])
        np.save(
            prefix + "_Bg_hit_distance.npy", render_result["Bg_hit_distance"]
        )
        self._write_mask(
            prefix + "_Bg_hit_valid.png", render_result["Bg_hit_valid"]
        )
        self._write_mask(
            prefix + "_Bg_projected_valid.png",
            render_result["Bg_projected_valid"],
        )
        self._write_mask(
            prefix + "_Bg_clean_visible.png",
            render_result["Bg_clean_visible"],
        )
        self._write_id_preview(
            prefix + "_Bg_object_id_vis.png",
            render_result["Bg_object_id"],
            render_result["Bg_hit_valid"],
        )
        if self.save_flow_png:
            self._write_flow_png(
                prefix + "_Phi_uv16.png",
                render_result["Phi"],
                render_result["phi_valid"],
            )
        if self.save_flow_arrows:
            self._write_flow_arrows(
                prefix + "_Phi_arrows.png",
                render_result["Phi"],
                render_result["phi_valid"],
                render_result["I"],
                stride=self.flow_arrow_stride,
                canvas_scale=self.flow_arrow_canvas_scale,
                min_magnitude_px=self.flow_arrow_min_magnitude,
                max_arrows=self.flow_arrow_max_count,
            )
            self._write_correspondence_pairs(
                prefix + "_Phi_pairs.png",
                render_result["Phi"],
                render_result["phi_valid"],
                render_result["I"],
                render_result["B"],
                stride=self.flow_arrow_stride,
                canvas_scale=self.flow_arrow_canvas_scale,
                min_magnitude_px=self.flow_arrow_min_magnitude,
                max_arrows=self.flow_arrow_max_count,
            )
        self._write_mask(prefix + "_phi_valid.png", render_result["phi_valid"])
        self._write_mask(
            prefix + "_object_mask.png", render_result["object_mask"]
        )
        self._write_mask(
            prefix + "_support_mask.png", render_result["support_mask"]
        )

        self._write_exr(prefix + "_A.exr", render_result["A"])
        np.save(prefix + "_alpha.npy", render_result["alpha"])
        self._write_exr(prefix + "_T.exr", render_result["T"])
        self._write_exr(prefix + "_CF.exr", render_result["C_F"])
        self._write_exr(prefix + "_CR.exr", render_result["C_R"])
        self._write_exr(prefix + "_F.exr", render_result["F"])
        self._write_exr(
            prefix + "_CF_material.exr", render_result["C_F_material"]
        )
        self._write_exr(
            prefix + "_F_material.exr", render_result["F_material"]
        )
        self._write_mask(prefix + "_F_valid.png", render_result["F_valid"])
        self._write_rgb_preview(
            prefix + "_CF_vis.png",
            render_result["C_F"],
            render_result["F_valid"],
        )
        self._write_rgb_preview(
            prefix + "_F_vis.png",
            render_result["F"],
            render_result["F_valid"],
        )

        # Canonical full-scene geometry GT. On object pixels N/D describe the
        # first transparent-object surface; elsewhere they describe the finite
        # clean-background plane. D is pinhole distance in scene units.
        np.save(prefix + "_N.npy", render_result["N"])
        np.save(prefix + "_D.npy", render_result["D"])
        self._write_exr(prefix + "_N.exr", render_result["N"])
        self._write_exr(prefix + "_D.exr", render_result["D"][..., None])
        self._write_normal_preview(
            prefix + "_N_vis.png",
            render_result["N"],
            render_result["N_valid"],
        )
        self._write_depth_preview(
            prefix + "_D_vis.png",
            render_result["D"],
            render_result["D_valid"],
        )
        self._write_mask(prefix + "_N_valid.png", render_result["N_valid"])
        self._write_mask(prefix + "_D_valid.png", render_result["D_valid"])

        # Geometry actually reached after the ray passes through both object
        # interfaces. D_refract is the broken-ray geometric path length.
        np.save(prefix + "_N_refract.npy", render_result["N_refract"])
        np.save(prefix + "_D_refract.npy", render_result["D_refract"])
        self._write_exr(prefix + "_N_refract.exr", render_result["N_refract"])
        self._write_exr(
            prefix + "_D_refract.exr", render_result["D_refract"][..., None]
        )
        self._write_normal_preview(
            prefix + "_N_refract_vis.png",
            render_result["N_refract"],
            render_result["Bg_hit_valid"],
        )
        self._write_depth_preview(
            prefix + "_D_refract_vis.png",
            render_result["D_refract"],
            render_result["Bg_hit_valid"],
        )

        # Explicit object-only geometry GT retained from v6.
        np.save(prefix + "_N_object.npy", render_result["N_object"])
        np.save(prefix + "_D_object.npy", render_result["D_object"])
        self._write_exr(prefix + "_N_object.exr", render_result["N_object"])
        self._write_exr(
            prefix + "_D_object.exr", render_result["D_object"][..., None]
        )
        self._write_normal_preview(
            prefix + "_N_object_vis.png",
            render_result["N_object"],
            render_result["N_object_valid"],
        )
        self._write_depth_preview(
            prefix + "_D_object_vis.png",
            render_result["D_object"],
            render_result["D_object_valid"],
        )
        self._write_mask(
            prefix + "_N_object_valid.png", render_result["N_object_valid"]
        )
        self._write_mask(
            prefix + "_D_object_valid.png", render_result["D_object_valid"]
        )
        # Backward-compatible v5/v6 alias remains object-only.
        np.save(prefix + "_normal.npy", render_result["N_object"])
        np.save(prefix + "_object_pose.npy", render_result["object_pose"])

        if self.save_debug_passes:
            self._write_exr(
                prefix + "_I_no_ref.exr", render_result["I_no_ref"]
            )
            self._write_exr(
                prefix + "_I_black.exr",
                render_result["I_black_reference"],
            )
            self._write_exr(
                prefix + "_I_black_no_ref.exr",
                render_result["I_black_reference_no_ref"],
            )
            self._write_exr(prefix + "_P.exr", render_result["P"])
            np.save(
                prefix + "_reconstruction_error.npy",
                render_result["reconstruction_error"],
            )
            if HAS_FLOW_VIS:
                phi_vis = flow_vis.flow_to_color(render_result["Phi"])
                phi_vis[render_result["phi_valid"] == 0] = 0
                imageio.imwrite(prefix + "_Phi_vis.png", phi_vis)

    def _save_sequence_static(
        self,
        sequence_prefix,
        background,
        background_normal,
        background_normal_valid,
        background_depth,
        background_depth_valid,
        sequence_meta,
    ):
        prefix = osp.join(self.output_folder, sequence_prefix)
        self._write_exr(prefix + "_background.exr", background)
        np.save(prefix + "_N_clean.npy", background_normal)
        np.save(prefix + "_D_clean.npy", background_depth)
        self._write_exr(prefix + "_N_clean.exr", background_normal)
        self._write_exr(prefix + "_D_clean.exr", background_depth[..., None])
        self._write_normal_preview(
            prefix + "_N_clean_vis.png",
            background_normal,
            background_normal_valid,
        )
        self._write_depth_preview(
            prefix + "_D_clean_vis.png",
            background_depth,
            background_depth_valid,
        )
        self._write_mask(
            prefix + "_N_clean_valid.png", background_normal_valid
        )
        self._write_mask(
            prefix + "_D_clean_valid.png", background_depth_valid
        )
        np.save(prefix + "_camera_intrinsic.npy", self.cam_intri_mat)
        np.save(prefix + "_camera_extrinsic.npy", self.cam_extri_mat)
        with open(prefix + "_sequence_meta.json", "w") as file:
            json.dump(sequence_meta, file, indent=2, sort_keys=True)

    def _frame_outputs_exist(self, basename):
        prefix = osp.join(self.output_folder, basename)
        required = (
            "_I.exr",
            "_Phi.npy",
            "_Phi_src.npy",
            "_Bg_hit_src.npy",
            "_Bg_hit_xyz.npy",
            "_Bg_hit_normal.npy",
            "_Bg_hit_uv.npy",
            "_Bg_object_id.npy",
            "_Bg_object_id_vis.png",
            "_Bg_hit_albedo.npy",
            "_Bg_hit_distance.npy",
            "_Bg_hit_valid.png",
            "_Bg_projected_valid.png",
            "_Bg_clean_visible.png",
            "_phi_valid.png",
            "_object_mask.png",
            "_support_mask.png",
            "_A.exr",
            "_alpha.npy",
            "_T.exr",
            "_CF.exr",
            "_CR.exr",
            "_F.exr",
            "_F_valid.png",
            "_CF_material.exr",
            "_F_material.exr",
            "_CF_vis.png",
            "_F_vis.png",
            "_N.npy",
            "_N.exr",
            "_N_vis.png",
            "_N_valid.png",
            "_D.npy",
            "_D.exr",
            "_D_vis.png",
            "_D_valid.png",
            "_N_refract.npy",
            "_N_refract.exr",
            "_N_refract_vis.png",
            "_D_refract.npy",
            "_D_refract.exr",
            "_D_refract_vis.png",
            "_N_object.npy",
            "_N_object.exr",
            "_N_object_vis.png",
            "_N_object_valid.png",
            "_D_object.npy",
            "_D_object.exr",
            "_D_object_vis.png",
            "_D_object_valid.png",
            "_object_pose.npy",
        )
        if self.save_flow_png:
            required = required + ("_Phi_uv16.png",)
        if self.save_flow_arrows:
            required = required + ("_Phi_arrows.png", "_Phi_pairs.png")
        return all(osp.exists(prefix + suffix) for suffix in required)

    def _sequence_static_exists(self, sequence_prefix):
        prefix = osp.join(self.output_folder, sequence_prefix)
        required = (
            "_background.exr",
            "_N_clean.npy",
            "_N_clean.exr",
            "_N_clean_vis.png",
            "_N_clean_valid.png",
            "_D_clean.npy",
            "_D_clean.exr",
            "_D_clean_vis.png",
            "_D_clean_valid.png",
            "_camera_intrinsic.npy",
            "_camera_extrinsic.npy",
            "_sequence_meta.json",
        )
        if not all(osp.exists(prefix + suffix) for suffix in required):
            return False
        try:
            with open(prefix + "_sequence_meta.json", "r") as file:
                metadata = json.load(file)
        except (OSError, ValueError):
            return False
        return metadata.get("generator_version") == self.GENERATOR_VERSION

    def _run_split(self, split, shape_list, background_list):
        """Generate one split with deterministic sequence-level sampling."""
        if not background_list:
            raise ValueError(f"No backgrounds available for split {split}")

        self.setup_output_paths(split)
        if osp.exists(self.info_file):
            with open(self.info_file, "r") as file:
                existing = {line.strip() for line in file if line.strip()}
        else:
            existing = set()

        total_frames = (
            len(shape_list)
            * self.backgrounds_per_shape
            * self.sequence_num_per_background
            * self.frames_per_sequence
        )
        print(
            f"Split: {split} | {len(existing)} checkpoint entries | "
            f"{total_frames} planned frames"
        )

        progress = tqdm(total=total_frames, desc=f"Rendering {split}")
        rendered_now = 0
        skipped_now = 0

        with open(self.info_file, "a") as checkpoint_file:
            for shape_path in shape_list:
                shape_name = osp.splitext(osp.basename(shape_path))[0]

                for background_iter in range(self.backgrounds_per_shape):
                    background_choice_seed = self._stable_seed(
                        self.dataset_seed,
                        split,
                        shape_path,
                        background_iter,
                        "background",
                    )
                    background_rng = np.random.default_rng(
                        background_choice_seed
                    )
                    background_path = background_list[
                        int(background_rng.integers(len(background_list)))
                    ]
                    background_name = osp.splitext(
                        osp.basename(background_path)
                    )[0]

                    for sequence_idx in range(
                        self.sequence_num_per_background
                    ):
                        sequence_prefix = (
                            f"{shape_name}_{background_name}"
                            f"_bg{background_iter:03d}_seq{sequence_idx:04d}"
                        )
                        frame_basenames = [
                            f"{sequence_prefix}_frame{frame_idx:04d}"
                            for frame_idx in range(self.frames_per_sequence)
                        ]
                        sequence_current = self._sequence_static_exists(
                            sequence_prefix
                        )
                        completed = [
                            basename in existing
                            and sequence_current
                            and self._frame_outputs_exist(basename)
                            for basename in frame_basenames
                        ]
                        if all(completed) and sequence_current:
                            skipped_now += self.frames_per_sequence
                            progress.update(self.frames_per_sequence)
                            continue

                        sequence_seed = self._stable_seed(
                            self.dataset_seed,
                            split,
                            shape_path,
                            background_path,
                            background_iter,
                            sequence_idx,
                        )
                        rng = np.random.default_rng(sequence_seed)
                        ior = float(
                            rng.uniform(self.ior_range[0], self.ior_range[1])
                        )

                        background_sample = self.update_background(
                            background_path, rng=rng
                        )
                        self.update_mesh(shape_path, ior)
                        reflection_scale = self.sample_reflection_scale(rng)
                        self.set_reflection_scale(reflection_scale)
                        self._random_cam_pose(rng=rng)

                        # All image-space decomposition passes must use the
                        # same primary-sample pattern.  In v10 the clean plate
                        # used a sequence-level "background" seed while each
                        # frame used a different "frame" seed.  Subtracting
                        # those independently sampled images injected a
                        # signed residual into C_F/C_F_material even where no
                        # object was present.  Keep one seed for the clean
                        # plate and every full/no-ref/object-only/transmission
                        # pass in the sequence. Object motion still changes
                        # scene content; only the camera/integrator samples are
                        # correlated.
                        decomposition_seed = self._render_seed(
                            self._stable_seed(
                                sequence_seed, "decomposition_samples"
                            )
                        )
                        background_pass = self._render_array(
                            self.background_scene,
                            self.camera,
                            decomposition_seed,
                        )
                        if background_pass.shape[-1] < 7:
                            raise RuntimeError(
                                "Expected clean RGB + normal + depth AOVs, got "
                                f"shape {background_pass.shape}"
                            )
                        background = background_pass[..., :3]
                        background_normal = np.asarray(
                            background_pass[..., 3:6], dtype=np.float32
                        )
                        background_depth = np.asarray(
                            background_pass[..., 6], dtype=np.float32
                        )
                        background_normal_valid = np.all(
                            np.isfinite(background_normal), axis=-1
                        ) & (np.linalg.norm(background_normal, axis=-1) > 1e-6)
                        background_normal[~background_normal_valid] = 0.0
                        background_depth_valid = np.isfinite(
                            background_depth
                        ) & (background_depth > 0.0)
                        background_depth[~background_depth_valid] = 0.0
                        trajectory = self.sample_object_trajectory(rng)

                        sequence_meta = {
                            "generator_version": self.GENERATOR_VERSION,
                            "sequence_seed": int(sequence_seed),
                            "decomposition_seed": int(decomposition_seed),
                            "sampling_alignment": (
                                "clean/full/no_ref/object_only/"
                                "object_only_no_ref/transmission share one "
                                "sequence-level Mitsuba seed"
                            ),
                            "shape_path": shape_path,
                            "background_path": background_path,
                            "background_index": int(background_iter),
                            "sequence_index": int(sequence_idx),
                            "frame_count": int(self.frames_per_sequence),
                            "camera_fixed": True,
                            "background_fixed": True,
                            "background_mode": self.background_mode,
                            "background_geometry": (
                                "textured_finite_plane"
                                if self.background_mode == "plane"
                                else "mitsuba_3d_scene"
                            ),
                            "background_z": (
                                self.background_z
                                if self.background_mode == "plane"
                                else None
                            ),
                            "background_sample": background_sample,
                            "background_half_size": (
                                self.background_half_size.tolist()
                                if self.background_mode == "plane"
                                else None
                            ),
                            "background_objects": (
                                {
                                    str(object_id): name
                                    for object_id, name in zip(
                                        self.background_object_ids,
                                        self.background_object_names,
                                    )
                                }
                                if self.background_mode == "scene"
                                else {"1": "background_plane"}
                            ),
                            "ambient_intensity": self.ambient_intensity,
                            "reflection_policy": (
                                "physical_full_fresnel_enabled; no random "
                                "reflection attenuation in supplied v14 configs"
                            ),
                            "object_motion": "rigid_se3",
                            "ior": ior,
                            "reflection_scale": reflection_scale,
                            "phi_type": "refractive_background_displacement",
                            "phi_unit": "pixel",
                            "phi_direction": "backward_sampling",
                            "phi_png": (
                                "RGB16: R=dx, G=dy, B=valid; "
                                "dx=(R/65535*2-1)*(width-1), "
                                "dy=(G/65535*2-1)*(height-1)"
                                if self.save_flow_png
                                else None
                            ),
                            "phi_arrows": (
                                "green output pixel -> red sampled background "
                                f"pixel; stride={self.flow_arrow_stride}; "
                                "exact endpoint scale=1"
                                if self.save_flow_arrows
                                else None
                            ),
                            "normal_gt": (
                                "N: signed world-space first-hit shading "
                                "normal for object and 3D background; "
                                "N_object: object-only; N_clean: object-free "
                                "background; N_refract: actual refracted hit"
                            ),
                            "depth_gt": (
                                "D: pinhole-to-first-visible-surface distance "
                                "for the full scene; D_object: object-only; "
                                "D_clean: clean-background first hit; "
                                "D_refract: broken-ray geometric path length"
                            ),
                            "background_hit_gt": (
                                "Bg_hit_xyz/normal/uv/object_id/albedo are the "
                                "actual 3D background intersections after two "
                                "object refractions. Bg_clean_visible marks "
                                "hits visible at their clean-camera projection."
                            ),
                            "reflection_gt": (
                                "C_R = C_F - C_F_material from paired "
                                "black-reference passes; auxiliary only and "
                                "included in canonical C_F/F"
                            ),
                            "foreground_gt": (
                                "standard matting foreground: F is straight "
                                "foreground color and C_F = alpha * F = "
                                "direct black-reference render; reflection/"
                                "highlights/TIR retained"
                            ),
                            "material_foreground_gt": (
                                "C_F_material = direct reflection-off "
                                "black-reference render; "
                                "C_F = C_F_material + C_R"
                            ),
                            "reconstruction": (
                                "I ~= alpha * F + (1-alpha) * P, where "
                                "P = warp(B, Phi) and C_F = alpha * F. The "
                                "saved "
                                "reconstruction_error is now a diagnostic of "
                                "the single-displacement background model, "
                                "not a forced algebraic identity"
                            ),
                            "A_factorization": "A = (1-alpha) * T",
                            "color_space": "linear_rgb",
                            "camera_pose": {
                                key: np.asarray(value).tolist()
                                for key, value in self.cam_pose.items()
                            },
                            "camera_x_fov_deg": self.current_camera_fov,
                        }
                        self._save_sequence_static(
                            sequence_prefix,
                            background,
                            background_normal,
                            background_normal_valid,
                            background_depth,
                            background_depth_valid,
                            sequence_meta,
                        )

                        for frame_idx, (basename, pose, is_complete) in enumerate(
                            zip(frame_basenames, trajectory, completed)
                        ):
                            if is_complete:
                                skipped_now += 1
                                progress.update(1)
                                continue

                            self.set_object_pose(pose)
                            render_result = self.single_render(
                                background=background,
                                background_depth=background_depth,
                                background_depth_valid=background_depth_valid,
                                render_seed=decomposition_seed,
                            )
                            self.single_save(render_result, basename)

                            checkpoint_file.write(f"{basename}\n")
                            checkpoint_file.flush()
                            existing.add(basename)
                            rendered_now += 1
                            progress.update(1)
                            del render_result

                gc.collect()
                dr.flush_malloc_cache()
                dr.flush_kernel_cache()

        progress.close()
        print(
            f"Split: {split} finished | rendered {rendered_now} new frames | "
            f"skipped {skipped_now} existing frames"
        )

    def run(self):
        self._run_split(
            "train", self.train_shape_list, self.train_background_list
        )
        self._run_split(
            "test", self.test_shape_list, self.test_background_list
        )
