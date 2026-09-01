import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

import refractive_mam2.dataset as dataset_module
from refractive_mam2 import (
    PairedBackgroundBatchSampler,
    RCTransPRISMDataset,
    prism_collate,
)


def _write_stub_sequence(root: Path, name: str, background_path: str) -> dict[str, np.ndarray]:
    prefix = root / name
    metadata = {
        "generator_version": "v15_prism_contract",
        "split_kind": "main",
        "frame_count": 1,
        "paired_background_group_id": "object_pose_group_0",
        "background_path": background_path,
    }
    Path(str(prefix) + "_sequence_meta.json").write_text(json.dumps(metadata))
    Path(str(prefix) + "_background.exr").touch()
    frame = Path(str(prefix) + "_frame000")
    for suffix in RCTransPRISMDataset.REQUIRED_SUFFIXES:
        Path(str(frame) + suffix).touch()

    h, w = 3, 4
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    background = np.full((h, w, 3), 0.5, np.float32)
    alpha = np.full((h, w), 0.25, np.float32)
    color = np.full((h, w, 3), 0.8, np.float32)
    tau = (1.0 - alpha[..., None]) * color
    g = np.full((h, w, 3), 0.1, np.float32)
    image = g + tau * background
    return {
        str(prefix) + "_background.exr": background,
        str(frame) + "_I.exr": image,
        str(frame) + "_object_mask.png": np.ones((h, w), np.float32),
        str(frame) + "_alpha.npy": alpha,
        str(frame) + "_CF.exr": g,
        str(frame) + "_F.exr": g / alpha[..., None],
        str(frame) + "_T.exr": color,
        str(frame) + "_A.exr": tau,
        str(frame) + "_Phi.npy": np.stack((xx, yy), axis=-1),
        str(frame) + "_u.npy": np.zeros((h, w, 2), np.float32),
        str(frame) + "_R.exr": np.zeros((h, w, 3), np.float32),
        str(frame) + "_confidence.npy": np.ones((h, w), np.float32),
        str(frame) + "_phi_valid.png": np.ones((h, w), np.float32),
    }


def test_loader_maps_full_v15_contract_and_builds_pairs(tmp_path, monkeypatch) -> None:
    arrays = {}
    arrays.update(_write_stub_sequence(tmp_path, "seq_a", "background_a"))
    arrays.update(_write_stub_sequence(tmp_path, "seq_b", "background_b"))
    monkeypatch.setattr(dataset_module, "_read_exr", lambda path: arrays[str(path)])
    monkeypatch.setattr(dataset_module, "_read_mask", lambda path: arrays[str(path)])
    monkeypatch.setattr(dataset_module, "_npy", lambda path: arrays[str(path)])

    dataset = RCTransPRISMDataset(tmp_path, strict_contract=True)
    assert len(dataset) == 2
    batch = prism_collate([dataset[0], dataset[1]])
    target = batch.ground_truth
    assert target.frames.shape == (2, 1, 3, 3, 4)
    assert target.color_transmission.shape == target.frames.shape
    assert target.source_coordinates.shape == (2, 1, 2, 3, 4)
    assert target.confidence.shape == (2, 1, 1, 3, 4)
    assert target.refractive_validity.shape == (2, 1, 1, 3, 4)
    pair = next(iter(PairedBackgroundBatchSampler(dataset, shuffle=False)))
    assert set(pair) == {0, 1}
