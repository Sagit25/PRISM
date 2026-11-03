"""Common mask/cutout protocol for matting baselines."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE = Path(__file__).parents[1] / "scripts" / "comparison_semantic.py"
SPEC = importlib.util.spec_from_file_location("comparison_semantic", MODULE)
assert SPEC is not None and SPEC.loader is not None
semantic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(semantic)


def test_explicit_mask_precedes_alpha_and_cutout_uses_observed_pixels() -> None:
    observed = np.array([[[0.2, 0.8]], [[0.4, 0.6]], [[0.7, 0.3]]], dtype=np.float32)
    explicit = np.array([[[1, 0]]], dtype=np.float32)
    alpha = np.array([[[0, 1]]], dtype=np.float32)
    mask, source = semantic.comparison_mask({"mask": explicit, "alpha": alpha}, (1, 2))
    assert source == "model_mask"
    np.testing.assert_array_equal(mask, explicit)
    np.testing.assert_allclose(
        semantic.observed_foreground_cutout(observed, mask), observed * explicit
    )


def test_alpha_only_mask_is_thresholded_without_scoring_alpha() -> None:
    mask, source = semantic.comparison_mask(
        {"alpha": np.array([[[0.49, 0.5]]], dtype=np.float32)}, (1, 2)
    )
    assert source == "alpha_threshold_0.5"
    np.testing.assert_array_equal(mask, np.array([[[0, 1]]], dtype=np.float32))
    assert semantic.comparison_mask({}, (1, 2)) == (None, None)
    with pytest.raises(ValueError):
        semantic.comparison_mask({"mask": np.array([[[np.nan]]])}, (1, 1))


def test_matting_preview_contains_mask_and_cutout_pairs(tmp_path: Path) -> None:
    rgb = np.full((2, 2, 3), 100, dtype=np.uint8)
    gt = np.array([[[1, 0], [0, 0]]], dtype=np.float32)
    pred = np.array([[[0, 1], [0, 0]]], dtype=np.float32)
    output = tmp_path / "preview.png"
    semantic.save_mask_cutout_preview(output, rgb, gt, pred)
    from PIL import Image
    with Image.open(output) as preview:
        assert preview.size == (2 * 3 * 5, (2 + 20) * 3)
