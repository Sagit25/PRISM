import numpy as np
import pytest
import trimesh


mi = pytest.importorskip("mitsuba")
pytest.importorskip("drjit")

if "llvm_ad_rgb" not in mi.variants():
    pytest.skip("Mitsuba LLVM variant is unavailable", allow_module_level=True)
mi.set_variant("llvm_ad_rgb")

from utils.mitsuba_tracer import MitsubaTracer


def _camera_rays(size: int = 41) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.linspace(-1.25, 1.25, size, dtype=np.float32)
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
    origins = np.stack(
        (xx, yy, np.zeros_like(xx)), axis=-1
    ).reshape(-1, 3)
    directions = np.zeros_like(origins)
    directions[:, 2] = 1.0
    return origins, directions


def _centroid_x(mask: np.ndarray, size: int) -> float:
    yy, xx = np.nonzero(mask.reshape(size, size))
    assert yy.size > 0
    return float(xx.mean())


def test_update_mesh_rebuilds_scene_intersection_for_new_pose() -> None:
    size = 41
    mesh = trimesh.creation.box(extents=(0.8, 1.4, 0.6))
    mesh.apply_translation((0.0, 0.0, 3.0))
    tracer = MitsubaTracer(mesh)
    origins, directions = _camera_rays(size)

    initial = tracer.ray_tracing(origins, directions)["mask"]

    moved = mesh.copy()
    moved.apply_translation((0.65, 0.0, 0.0))
    tracer.update_mesh(moved)
    translated = tracer.ray_tracing(origins, directions)["mask"]

    assert not np.array_equal(initial, translated)
    assert _centroid_x(translated, size) > _centroid_x(initial, size) + 5.0
