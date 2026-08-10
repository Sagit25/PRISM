"""Helpers for registering Mitsuba scene elements."""
from __future__ import annotations

from utils.registry import ELEMENT_REGISTRY


def get_scene_element(name: str):
    return ELEMENT_REGISTRY.get(name)
