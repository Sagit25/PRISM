"""Project registry and builder."""
import importlib

from utils.registry import PROJECT_REGISTRY


def build_project(conf):
    name = conf['Project']
    importlib.import_module(f'.{name}', package=__name__)
    project_cls = PROJECT_REGISTRY.get(name)
    return project_cls(conf)
