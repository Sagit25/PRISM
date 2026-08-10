"""YAML config loader with file-level include support."""
import os

from omegaconf import OmegaConf, DictConfig


def load_config(config_path: str) -> DictConfig:
    """Load a YAML config with file-level include support.

    If the config contains an 'includes' key (list of relative paths),
    each fragment is loaded and merged as a base. The main config
    overrides all fragment values.
    """
    config_path = os.path.abspath(config_path)
    config_dir = os.path.dirname(config_path)

    main_conf = OmegaConf.load(config_path)

    if 'includes' in main_conf:
        include_paths = main_conf.pop('includes')
        base = OmegaConf.create({})
        for inc_path in include_paths:
            if not os.path.isabs(inc_path):
                inc_path = os.path.join(config_dir, inc_path)
            fragment = OmegaConf.load(inc_path)
            base = OmegaConf.merge(base, fragment)
        main_conf = OmegaConf.merge(base, main_conf)

    return main_conf
