from pathlib import Path

import yaml


CONFIG_ROOT = Path(__file__).parents[1] / "configs"
MAIN_CONFIGS = (
    "dataset_prism_main.yaml",
    "dataset_prism_main_smoke.yaml",
    "dataset_prism_research_asset_smoke.yaml",
)


def load_config(name: str) -> dict:
    with (CONFIG_ROOT / name).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_all_main_configs_keep_full_physical_fresnel_reflection() -> None:
    for name in MAIN_CONFIGS:
        config = load_config(name)
        reflection = config["Reflection"]

        assert config["split_kind"] == "main", name
        assert reflection["enabled"] is True, name
        assert float(reflection["zero_probability"]) == 0.0, name
        assert [float(value) for value in reflection["reflection_scale_range"]] == [
            1.0,
            1.0,
        ], name


def test_main_transparent_mesh_bootstraps_with_full_reflection() -> None:
    for name in MAIN_CONFIGS:
        config = load_config(name)
        transparent = [
            element
            for element in config["Scene"]["element"]
            if element["type"] == "transparent_mesh"
        ]

        assert transparent, name
        assert all(
            float(element["reflection_scale"]) == 1.0
            for element in transparent
        ), name
