"""Factory for selecting the refraction tracer backend."""

from utils.mitsuba_tracer import MitsubaTracer
from utils.tool_utils import convert_to_dict
from utils.trimesh_tracer import TrimeshTracer


def build_tracer(mesh, conf=None, obj_ior=1.5, air_ior=1.0, debug=False):
    conf = convert_to_dict(conf) if conf is not None else {}
    tracer_conf = conf.get("Tracer", {}) if isinstance(conf, dict) else {}
    backend = tracer_conf.get("backend", "trimesh")

    if backend == "trimesh":
        return TrimeshTracer(mesh, objIOR=obj_ior, airIOR=air_ior, debug=debug)
    if backend == "mitsuba":
        return MitsubaTracer(mesh, objIOR=obj_ior, airIOR=air_ior, debug=debug)
    raise ValueError(f"Unknown tracer backend: {backend}")
