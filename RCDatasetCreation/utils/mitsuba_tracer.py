"""GPU/CPU ray tracer built on Mitsuba scene intersections."""

from __future__ import annotations

import os
import tempfile
from typing import Union

import drjit as dr
import mitsuba as mi
import numpy as np
import trimesh

default_cache_dir = os.path.join(tempfile.gettempdir(), "drjit-cache")
os.environ.setdefault("DRJIT_CACHE_DIR", default_cache_dir)
os.makedirs(os.environ["DRJIT_CACHE_DIR"], exist_ok=True)


class MitsubaTracer:
    """Speed-first refraction tracer using Mitsuba ray intersections."""

    def __init__(self, mesh: Union[trimesh.Trimesh, str], objIOR=1.5, airIOR=1.0, debug=False):
        if isinstance(mesh, str):
            mesh = trimesh.load(mesh)
        self.mesh = mesh
        self.objIOR = objIOR
        self.airIOR = airIOR
        self.debug = debug
        self._mi_mesh = self._build_mitsuba_mesh(mesh)
        self._scene = mi.load_dict({"type": "scene", "shape": self._mi_mesh})

    def _build_mitsuba_mesh(self, mesh: trimesh.Trimesh):
        mi_mesh = mi.Mesh(
            "tracer_mesh",
            mesh.vertices.shape[0],
            mesh.faces.shape[0],
            has_vertex_normals=True,
        )
        params = mi.traverse(mi_mesh)
        params["vertex_positions"] = dr.ravel(mi.Point3f(np.asarray(mesh.vertices, dtype=np.float32).T))
        params["vertex_normals"] = dr.ravel(mi.Point3f(np.asarray(mesh.vertex_normals, dtype=np.float32).T))
        params["faces"] = dr.ravel(mi.Point3u(np.asarray(mesh.faces, dtype=np.uint32).T))
        params.update()
        mi_mesh.initialize()
        return mi_mesh

    def _as_point3f(self, rays_o):
        if isinstance(rays_o, mi.Point3f):
            return rays_o
        return mi.Point3f(rays_o[:, 0], rays_o[:, 1], rays_o[:, 2])

    def _as_vector3f(self, rays_d):
        if isinstance(rays_d, mi.Vector3f):
            return rays_d
        return mi.Vector3f(rays_d[:, 0], rays_d[:, 1], rays_d[:, 2])

    def _ray_tracing_raw(self, ray_origins, ray_directions):
        ray_origins = self._as_point3f(ray_origins)
        ray_directions = self._as_vector3f(ray_directions)
        ray = mi.Ray3f(
            o=ray_origins,
            d=ray_directions,
            time=mi.Float(0.0),
            wavelengths=[],
        )
        si = self._scene.ray_intersect(ray)
        hit_mask = si.is_valid()
        hit_idx = dr.compress(hit_mask)

        output = {
            "location": self._gather_point3f(si.p, hit_idx),
            # Speed-first choice: rely on Mitsuba's shading normals and stay on-device.
            "normal": self._gather_normal3f(si.sh_frame.n, hit_idx),
            "ray_idx": hit_idx,
            "mask": hit_mask,
        }
        if self.debug:
            output["face_idx"] = dr.gather(mi.UInt32, si.prim_index, hit_idx)
        return output

    def _gather_point3f(self, value, index):
        return mi.Point3f(
            dr.gather(mi.Float, value.x, index),
            dr.gather(mi.Float, value.y, index),
            dr.gather(mi.Float, value.z, index),
        )

    def _gather_vector3f(self, value, index):
        return mi.Vector3f(
            dr.gather(mi.Float, value.x, index),
            dr.gather(mi.Float, value.y, index),
            dr.gather(mi.Float, value.z, index),
        )

    def _gather_normal3f(self, value, index):
        return mi.Normal3f(
            dr.gather(mi.Float, value.x, index),
            dr.gather(mi.Float, value.y, index),
            dr.gather(mi.Float, value.z, index),
        )

    def _scatter_vector3f(self, target, value, index):
        if dr.width(index) == 0:
            return
        dr.scatter(target.x, value=value.x, index=index)
        dr.scatter(target.y, value=value.y, index=index)
        dr.scatter(target.z, value=value.z, index=index)

    def _scatter_point2f(self, target, value, index):
        if dr.width(index) == 0:
            return
        dr.scatter(target.x, value=value.x, index=index)
        dr.scatter(target.y, value=value.y, index=index)

    def _scatter_bool(self, target, value, index):
        if dr.width(index) == 0:
            return
        dr.scatter(target, value=value, index=index)

    def _scatter_float(self, target, value, index):
        if dr.width(index) == 0:
            return
        dr.scatter(target, value=value, index=index)

    def snell(self, rays_dir, normal, relative_IOR):
        inv_rela_IOR = 1.0 / relative_IOR
        cos_theta_i = dr.clamp(dr.dot(-rays_dir, normal), 1e-5, 1.0)
        sin2_theta_i = dr.maximum(0.0, 1.0 - cos_theta_i * cos_theta_i)
        sin2_theta_t = (inv_rela_IOR * inv_rela_IOR) * sin2_theta_i
        total_reflect = sin2_theta_t >= 1.0
        cos_theta_t = dr.sqrt(dr.clamp(1.0 - sin2_theta_t, 1e-5, 1.0))
        out_dir = dr.normalize(inv_rela_IOR * rays_dir + (inv_rela_IOR * cos_theta_i - cos_theta_t) * normal)
        return out_dir, total_reflect, cos_theta_i

    def refract_ray(self, rays_d, location, normal, inverse_mode=False):
        if not inverse_mode:
            intIOR = self.objIOR
            extIOR = self.airIOR
        else:
            intIOR = self.airIOR
            extIOR = self.objIOR
            normal = -normal

        relative_ior = mi.Float(intIOR / extIOR)
        new_dir, total_reflect, _ = self.snell(rays_d, normal, relative_ior)
        valid_mask = ~total_reflect
        valid_idx = dr.compress(valid_mask)
        new_ori = location + mi.Float(1e-2) * new_dir

        return {
            "rays_o": self._gather_point3f(new_ori, valid_idx),
            "rays_d": self._gather_vector3f(new_dir, valid_idx),
            "totalRef": total_reflect,
            "valid_idx": valid_idx,
        }

    def first_bounce(self, rays_o, rays_d):
        rays_o = self._as_point3f(rays_o)
        rays_d = self._as_vector3f(rays_d)
        tracing_output = self._ray_tracing_raw(rays_o, rays_d)
        hit_dirs = self._gather_vector3f(rays_d, tracing_output["ray_idx"])
        trans_ray = self.refract_ray(hit_dirs, tracing_output["location"], tracing_output["normal"], inverse_mode=False)

        output = {
            "trans_ray": trans_ray,
            "mask": tracing_output["mask"],
            "normal": tracing_output["normal"],
            "ray_idx": tracing_output["ray_idx"],
        }
        if self.debug and "face_idx" in tracing_output:
            output["face_idx"] = tracing_output["face_idx"]
        return output

    def second_bounce(self, rays_o, rays_d):
        rays_o = self._as_point3f(rays_o)
        rays_d = self._as_vector3f(rays_d)
        tracing_output = self._ray_tracing_raw(rays_o, rays_d)
        hit_dirs = self._gather_vector3f(rays_d, tracing_output["ray_idx"])
        trans_ray = self.refract_ray(hit_dirs, tracing_output["location"], tracing_output["normal"], inverse_mode=True)
        valid_relative_idx = trans_ray["valid_idx"]
        output_ray_idx = dr.gather(mi.UInt32, tracing_output["ray_idx"], valid_relative_idx)

        return {
            "trans_ray": trans_ray,
            "ray_idx": output_ray_idx,
        }

    def ray_tracing(self, ray_origins, ray_directions):
        output = self._ray_tracing_raw(ray_origins, ray_directions)
        result = {
            "location": np.array(output["location"], dtype=np.float32).T,
            "normal": np.array(output["normal"], dtype=np.float32).T,
            "ray_idx": np.array(output["ray_idx"], dtype=np.int32),
            "mask": np.array(output["mask"], dtype=bool),
        }
        if self.debug and "face_idx" in output:
            result["face_idx"] = np.array(output["face_idx"], dtype=np.int32)
        return result

    def trace_out_dir(self, rays_ori, rays_dir):
        ray_num = rays_ori.shape[0]
        rays_o = self._as_point3f(rays_ori)
        rays_d = self._as_vector3f(rays_dir)

        first_bounce_output = self.first_bounce(rays_o, rays_d)
        second_bounce_output = self.second_bounce(
            first_bounce_output["trans_ray"]["rays_o"],
            first_bounce_output["trans_ray"]["rays_d"],
        )
        third_bounce_output = self._ray_tracing_raw(
            second_bounce_output["trans_ray"]["rays_o"],
            second_bounce_output["trans_ray"]["rays_d"],
        )

        out_dir = dr.zeros(mi.Vector3f, ray_num)
        twice_mask = dr.zeros(mi.Bool, ray_num)
        normal = dr.zeros(mi.Vector3f, ray_num)

        self._scatter_vector3f(normal, first_bounce_output["normal"], first_bounce_output["ray_idx"])

        final_ray_idx = dr.gather(mi.UInt32, first_bounce_output["ray_idx"], second_bounce_output["ray_idx"])
        self._scatter_vector3f(out_dir, second_bounce_output["trans_ray"]["rays_d"], final_ray_idx)
        self._scatter_bool(twice_mask, mi.Bool(True), final_ray_idx)

        invalid_ray_idx = dr.gather(mi.UInt32, final_ray_idx, third_bounce_output["ray_idx"])
        self._scatter_vector3f(out_dir, mi.Vector3f(0.0), invalid_ray_idx)
        self._scatter_bool(twice_mask, mi.Bool(False), invalid_ray_idx)

        return {
            "out_dir": np.array(out_dir, dtype=np.float32).T,
            "obj_mask": np.array(first_bounce_output["mask"], dtype=bool),
            "twice_mask": np.array(twice_mask, dtype=bool),
            "normal": np.array(normal, dtype=np.float32).T,
        }

    def intersect_with_plane(self, rays_o, rays_d, z_val=0):
        z_val = mi.Float(z_val)
        t = (z_val - rays_o.z) / (rays_d.z + mi.Float(1e-5))
        x = rays_o.x + t * rays_d.x
        y = rays_o.y + t * rays_d.y
        return mi.Point2f(x, y)

    def trace_corres(self, rays_ori, rays_dir, z_val):
        ray_num = rays_ori.shape[0]
        rays_o = self._as_point3f(rays_ori)
        rays_d = self._as_vector3f(rays_dir)

        first_bounce_output = self.first_bounce(rays_o, rays_d)
        second_bounce_output = self.second_bounce(
            first_bounce_output["trans_ray"]["rays_o"],
            first_bounce_output["trans_ray"]["rays_d"],
        )
        valid_corres = self.intersect_with_plane(
            second_bounce_output["trans_ray"]["rays_o"],
            second_bounce_output["trans_ray"]["rays_d"],
            z_val=z_val,
        )

        corres = dr.zeros(mi.Point2f, ray_num)
        twice_mask = dr.zeros(mi.Float, ray_num)
        normal = dr.zeros(mi.Vector3f, ray_num)

        final_ray_idx = dr.gather(mi.UInt32, first_bounce_output["ray_idx"], second_bounce_output["ray_idx"])
        self._scatter_vector3f(normal, first_bounce_output["normal"], first_bounce_output["ray_idx"])
        self._scatter_point2f(corres, valid_corres, final_ray_idx)
        self._scatter_float(twice_mask, mi.Float(1.0), final_ray_idx)

        return {
            "corres": np.array(corres, dtype=np.float32).T,
            "obj_mask": np.array(first_bounce_output["mask"], dtype=bool),
            "twice_mask": np.array(twice_mask, dtype=np.float32),
            "normal": np.array(normal, dtype=np.float32).T,
        }
