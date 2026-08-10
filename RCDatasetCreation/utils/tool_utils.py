"""Small misc helpers shared across modules."""
import random
from typing import Any
import os
import os.path as osp

import numpy as np
from omegaconf import DictConfig, ListConfig

def _to_float_array(data):
    """Convert arbitrary array-likes to float32 numpy arrays."""
    return np.asarray(data, dtype=np.float32)


def linear_to_srgb(linear):
    """Convert linear RGB values in [0, 1] to sRGB."""
    arr = _to_float_array(linear)
    eps = np.finfo(np.float32).eps
    srgb0 = 323 / 25 * arr
    srgb1 = (211 * np.clip(arr, eps, None) ** (5 / 12) - 11) / 200
    return np.where(arr <= 0.0031308, srgb0, srgb1)


def srgb_to_linear(srgb):
    """Convert sRGB values in [0, 1] to linear RGB."""
    arr = _to_float_array(srgb)
    eps = np.finfo(np.float32).eps
    linear0 = 25 / 323 * arr
    linear1 = np.clip(((200 * arr + 11) / 211), eps, None) ** (12 / 5)
    return np.where(arr <= 0.04045, linear0, linear1)

def convert_to_dict(data: Any):
    """Recursively convert DictConfig/ListConfig into vanilla Python types."""
    if isinstance(data, DictConfig):
        return {key: convert_to_dict(value) for key, value in data.items()}
    if isinstance(data, (list, ListConfig)):
        return [convert_to_dict(item) for item in data]
    return data


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def spherical2cartesian(theta_deg, phi_deg, radius):
    """Convert spherical angles (deg) to cartesian coordinates."""
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    return np.array([x, y, z], dtype=np.float32)

def scandir(dir_path, suffix=None, recursive=False, full_path=False, getDirs=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative pathes.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if (not entry.name.startswith('.')) and (entry.is_file() or getDirs):
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)

