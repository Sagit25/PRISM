"""Lightweight registries for projects and scene elements."""


class Registry:
    """Minimal name -> object mapping with decorator-style register."""

    def __init__(self, name):
        self._name = name
        self._items = {}

    def register(self, obj=None, name=None):
        if obj is None:
            def deco(target):
                self._store(target, name)
                return target
            return deco
        self._store(obj, name)
        return obj

    def _store(self, obj, name=None):
        key = name or obj.__name__
        if key in self._items:
            raise KeyError(f'{self._name} `{key}` already registered')
        self._items[key] = obj

    def get(self, name):
        return self._items[name]

    def keys(self):
        return list(self._items.keys())


PROJECT_REGISTRY = Registry('Project')
ELEMENT_REGISTRY = Registry('SceneElement')
CAMPOSE_REGISTRY = Registry('CamPose')
SCENE_REGISTRY = Registry('Scene')
