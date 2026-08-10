import trimesh
import numpy as np
from typing import Union


class TrimeshTracer():
    def __init__(self, mesh: Union[trimesh.Trimesh, str], objIOR=1.5, airIOR=1.0, debug=False):
        if isinstance(mesh, str):
            mesh = trimesh.load(mesh, use_embree=True)
        self.mesh = mesh
        self.objIOR = objIOR
        self.airIOR = airIOR
        self.debug = debug

    def snell(self, rays_dir: np.ndarray, normal: np.ndarray, relative_IOR):
        inv_rela_IOR = 1 / relative_IOR
        cos_theta_i = np.matmul(-rays_dir.reshape(-1, 1, 3), normal.reshape(-1, 3, 1))
        cos_theta_i = cos_theta_i.clip(1e-5, 1).reshape(-1, 1)
        sin2_theta_i = (1 - cos_theta_i ** 2).clip(min=0)
        sin2_theta_t = inv_rela_IOR ** 2 * sin2_theta_i
        total_reflect = (sin2_theta_t >= 1)
        cos_theta_t = np.sqrt(1 - sin2_theta_t.clip(max=1 - 1e-5, min=1e-5))
        out_dir = inv_rela_IOR * rays_dir + (inv_rela_IOR * cos_theta_i - cos_theta_t) * normal

        out_dir = out_dir / np.linalg.norm(out_dir, axis=1).reshape(-1, 1)

        return out_dir, total_reflect.squeeze(-1), cos_theta_i

    def refract_ray(self, rays_d: np.ndarray, location, normal, inverse_mode=False):
        if not inverse_mode:
            intIOR = self.objIOR
            extIOR = self.airIOR
        else:
            intIOR = self.airIOR
            extIOR = self.objIOR
            normal = -normal

        intIOR = intIOR * np.ones([rays_d.shape[0], 1])
        extIOR = extIOR * np.ones([rays_d.shape[0], 1])
        new_dir, total_reflect, cos_theta_i = self.snell(rays_d, normal, intIOR / extIOR)

        new_ori = location + 1e-2 * new_dir

        assert (not np.isnan(new_ori).any())
        assert (not np.isnan(new_dir).any())

        output = {
            'rays_o': new_ori[~total_reflect],
            'rays_d': new_dir[~total_reflect],
            'totalRef': total_reflect,
        }

        if self.debug:
            output.update({'intIOR': intIOR, 'extIOR': extIOR})

        return output

    def first_bounce(self, rays_o, rays_d):
        """First refraction bounce (air -> object)."""
        inverse_mode = False
        tracing_output = self.ray_tracing(rays_o, rays_d)
        location = tracing_output.pop('location')
        normal = tracing_output.pop('normal')
        ray_idx = tracing_output.pop('ray_idx')
        mask = tracing_output.pop('mask')

        trans_ray = self.refract_ray(rays_d[mask], location, normal, inverse_mode=inverse_mode)
        assert (not np.any(trans_ray['totalRef']))

        output = {
            'trans_ray': trans_ray,
            'mask': mask,
            'normal': normal,
            'ray_idx': ray_idx,
        }

        if self.debug:
            output.update({'location': location, 'tracing_output': tracing_output})

        return output

    def second_bounce(self, rays_o, rays_d):
        """Second refraction bounce (object -> air), may have total internal reflection."""
        inverse_mode = True
        tracing_output = self.ray_tracing(rays_o, rays_d)
        location = tracing_output.pop('location')
        normal = tracing_output.pop('normal')
        ray_idx = tracing_output.pop('ray_idx')
        tracing_mask = tracing_output.pop('mask')

        trans_ray = self.refract_ray(rays_d[tracing_mask], location, normal, inverse_mode=inverse_mode)

        totalRef_bounce2 = trans_ray['totalRef']
        totalRef_mask = ~totalRef_bounce2

        mask = np.zeros(rays_o.shape[0], dtype=bool)
        mask[ray_idx][totalRef_mask] = 1
        output = {
            'trans_ray': trans_ray,
            'mask': mask,
            'ray_idx': ray_idx[totalRef_mask],
        }
        if self.debug:
            output.update({'location': location, 'tracing_output': tracing_output, 'normal': normal})

        return output

    def interp_normal(self, face_index, points):
        bary_coord = trimesh.triangles.points_to_barycentric(
            triangles=self.mesh.triangles[face_index], points=points)
        bary_coord = trimesh.unitize(bary_coord).reshape(-1, 3, 1)
        normal = self.mesh.vertex_normals[self.mesh.faces[face_index]] * bary_coord
        normal = trimesh.unitize(normal.sum(axis=1))
        return normal

    def ray_tracing(self, ray_origins, ray_directions):
        """
        Ray-mesh intersection using trimesh (CPU, accelerated by pyembree).
        """
        ray_num = ray_origins.shape[0]
        valid_locations, ray_idx, face_idx = self.mesh.ray.intersects_location(
            ray_origins, ray_directions, multiple_hits=False)

        valid_normals = self.interp_normal(face_idx, valid_locations)

        hit_mask = np.zeros(ray_num) == 1
        hit_mask[ray_idx] = True

        output = {
            'location': valid_locations,
            'normal': valid_normals,
            'ray_idx': ray_idx,
            'mask': hit_mask
        }

        if self.debug:
            output.update({'face_idx': face_idx})
        return output

    def trace_out_dir(self, rays_ori, rays_dir):
        """Trace two-bounce refraction and return output directions."""
        ray_num = rays_ori.shape[0]

        first_bounce_output = self.first_bounce(rays_ori, rays_dir)

        second_bounce_output = self.second_bounce(first_bounce_output['trans_ray']['rays_o'],
                                                  first_bounce_output['trans_ray']['rays_d'])

        # Check if refracted rays hit the object again (invalid)
        third_bounce_output = self.ray_tracing(second_bounce_output['trans_ray']['rays_o'],
                                               second_bounce_output['trans_ray']['rays_d'])

        out_dir = np.zeros([ray_num, 3])
        twice_mask = np.zeros(ray_num, dtype=bool)
        normal = np.zeros([ray_num, 3])

        first_ray_idx = first_bounce_output['ray_idx']
        second_ray_idx = second_bounce_output['ray_idx']
        third_ray_idx = third_bounce_output['ray_idx']
        normal[first_ray_idx] = first_bounce_output['normal']
        out_dir[first_ray_idx[second_ray_idx], :] = second_bounce_output['trans_ray']['rays_d']
        twice_mask[first_ray_idx[second_ray_idx]] = True
        twice_mask[first_ray_idx[second_ray_idx[third_ray_idx]]] = False
        out_dir[twice_mask == 0] = 0

        output = {
            'out_dir': out_dir,
            'obj_mask': first_bounce_output['mask'],
            'twice_mask': twice_mask,
            'normal': normal,
        }
        return output

    def trace_corres(self, rays_ori, rays_dir, z_val):
        """Trace refraction correspondence to a z-plane."""
        ray_num = rays_ori.shape[0]

        first_bounce_output = self.first_bounce(rays_ori, rays_dir)

        second_bounce_output = self.second_bounce(first_bounce_output['trans_ray']['rays_o'],
                                                  first_bounce_output['trans_ray']['rays_d'])

        valid_corres = self.intersect_with_plane(second_bounce_output['trans_ray']['rays_o'],
                                                 second_bounce_output['trans_ray']['rays_d'],
                                                 z_val=z_val)

        corres = np.zeros([ray_num, 2])
        twice_mask = np.zeros(ray_num)
        normal = np.zeros([ray_num, 3])

        first_ray_idx = first_bounce_output['ray_idx']
        second_ray_idx = second_bounce_output['ray_idx']
        normal[first_ray_idx] = first_bounce_output['normal']
        corres[first_ray_idx[second_ray_idx], :] = valid_corres
        twice_mask[first_ray_idx[second_ray_idx]] = 1

        output = {
            'corres': corres,
            'obj_mask': first_bounce_output['mask'],
            'twice_mask': twice_mask,
            'normal': normal,
        }

        if self.debug:
            output.update({'first_bounce_output': first_bounce_output,
                           'second_bounce_output': second_bounce_output})

        return output

    def intersect_with_plane(self, rays_o, rays_d, z_val=0):
        t = (z_val - rays_o[:, 2]) / (rays_d[:, 2] + 1e-5)
        x = rays_o[:, 0] + t * rays_d[:, 0]
        y = rays_o[:, 1] + t * rays_d[:, 1]
        return np.stack([x, y], axis=1)
