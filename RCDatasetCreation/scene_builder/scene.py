import os

import mitsuba as mi
from omegaconf import ListConfig

from scene_builder.elements import get_scene_element
from utils.tool_utils import convert_to_dict
from utils.registry import SCENE_REGISTRY, ELEMENT_REGISTRY


class Scene:
    def __init__(self, conf):
        self.source_path = conf['source_path']
        self.conf = conf
        self.scene_dict = {'type': 'scene'}
        self.scale = conf.get('scale', 1.0)

    def build_mitsuba_scene(self):
        return mi.load_dict(self.scene_dict)

    def attach_integrator(self):
        conf = self.conf['integrator']
        integrator_conf = convert_to_dict(conf)
        self.scene_dict['integrator'] = mi.load_dict(integrator_conf)

    def resolve_source_path(self, middle, path):
        return os.path.join(self.source_path, middle, path)


@SCENE_REGISTRY.register()
class XMLScene(Scene):
    def __init__(self, conf):
        super().__init__(conf)
        xml_path = self.resolve_source_path('', conf['filename'])
        self.scene = mi.load_file(xml_path)


@SCENE_REGISTRY.register()
class CustomScene(Scene):
    def __init__(self, conf):
        super().__init__(conf)
        element_conf = convert_to_dict(conf['element'])
        for element_name, params in self._iter_element_configs(element_conf):
            element_func = get_scene_element(element_name)
            element_nodes = element_func(self, params)
            if element_nodes:
                self.scene_dict.update(element_nodes)
        self.attach_integrator()
        self.scene = self.build_mitsuba_scene()

    def attach_integrator(self):
        """Load integrator. Use mitsuba_utils.wrap_with_aov for AOV support."""
        conf = self.conf['integrator']
        integrator_conf = convert_to_dict(conf)
        self.scene_dict['integrator'] = mi.load_dict(integrator_conf)

    def _iter_element_configs(self, elements_conf):
        """Yield (element_name, params_dict) tuples from the scene element list."""
        if not isinstance(elements_conf, (list, ListConfig)):
            raise TypeError('Scene.element must be a list of element configs.')
        for conf in elements_conf:
            element_name = conf['type']
            params = dict(conf)
            params.pop('type', None)
            if 'node' in params and 'name' not in params:
                params['name'] = params['node']
            yield element_name, params
