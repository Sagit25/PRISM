"""Miscellaneous built-in scene elements."""
from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

import scene_builder.mitsuba_utils as lib
from utils.registry import ELEMENT_REGISTRY

__all__ = ['raw']


def _resolve_raw_value(value):
    """Resolve raw payload values that encode simple Mitsuba transforms."""
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)

    if isinstance(value, dict):
        transform_type = value.get('type')
        if transform_type == 'rotation':
            return lib.transform_rotate(value['axis'], value['angle'])
        if transform_type == 'translate':
            return lib.transform_translate(value['value'])
        if transform_type == 'scale':
            return lib.transform_scale(value['value'])
        if transform_type == 'look_at':
            return lib.look_at(value['origin'], value['target'], value.get('up', [0, 0, 1]))
        return {key: _resolve_raw_value(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_resolve_raw_value(item) for item in value]

    return value


@ELEMENT_REGISTRY.register()
def raw(scene, conf):
    """Inject a Mitsuba node from raw dictionary payload."""
    node_name = conf.get('name') or conf['node']
    payload = conf['payload']
    payload = _resolve_raw_value(payload)
    return {node_name: payload}
