"""Regression checks for the generated-data contract preflight."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")


def _validator_module():
    source = (
        Path(__file__).resolve().parents[2]
        / "RCDatasetCreation/tools/validate_prism_contract.py"
    )
    spec = importlib.util.spec_from_file_location("prism_generation_validator", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_rejects_nonfinite_sequence_background(tmp_path, monkeypatch) -> None:
    validator = _validator_module()
    prefix = tmp_path / "sample_seq0000_frame000"
    sequence = tmp_path / "sample_seq0000"
    for suffix in validator.FRAME_SUFFIXES:
        Path(str(prefix) + suffix).touch()
    background_file = Path(str(sequence) + "_background.exr")
    background_file.touch()

    background = np.zeros((2, 2, 3), np.float32)
    background[0, 0, 0] = np.nan
    monkeypatch.setattr(
        validator,
        "read_exr",
        lambda path: background if path == background_file else np.zeros((2, 2, 3), np.float32),
    )
    monkeypatch.setattr(
        validator.np,
        "load",
        lambda path: (
            np.zeros((2, 2, 2), np.float32)
            if str(path).endswith(("_Phi.npy", "_u.npy"))
            else np.zeros((2, 2), np.float32)
        ),
    )
    monkeypatch.setattr(
        validator.cv2,
        "imread",
        lambda path, flags: np.ones((2, 2), np.uint8),
    )

    with pytest.raises(AssertionError, match="NaN or Inf"):
        validator.validate_frame(prefix)
