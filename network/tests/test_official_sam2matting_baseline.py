"""Metrics used by the released-weight SAM2Matting comparison run."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch


MODULE = Path(__file__).parents[1] / "scripts" / "evaluate_official_sam2matting.py"
SPEC = importlib.util.spec_from_file_location("sam2matting_baseline", MODULE)
assert SPEC is not None and SPEC.loader is not None
baseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(baseline)


def test_prompt_frame_excluded_from_propagated_metric() -> None:
    frames = torch.ones(1, 2, 3, 2, 2)
    truth = torch.zeros(1, 2, 1, 2, 2)
    mask_truth = truth.clone()
    mask_logits = torch.full_like(truth, -1)
    mask_logits[:, 1] = 1
    all_frames = baseline.metrics_from_totals(
        baseline.score_clip(
            frames, mask_logits, mask_truth, skip_prompt_frame=False
        )
    )
    propagated = baseline.metrics_from_totals(
        baseline.score_clip(
            frames, mask_logits, mask_truth, skip_prompt_frame=True
        )
    )
    assert all_frames["observed_cutout_l1"] == 0.5
    assert propagated["observed_cutout_l1"] == 1.0
    assert propagated["mask_iou"] == 0.0
    assert propagated["frames"] == 1
    assert not any("alpha" in name for name in propagated)


def test_per_sequence_totals_use_pixel_weighting() -> None:
    destination: dict[str, float] = {}
    baseline.add_totals(destination, {"cutout_abs_sum": 1.0, "cutout_rgb_values": 2.0})
    baseline.add_totals(destination, {"cutout_abs_sum": 1.0, "cutout_rgb_values": 8.0})
    assert destination == {"cutout_abs_sum": 2.0, "cutout_rgb_values": 10.0}


def test_test_directory_accepts_main_generation_recipe(tmp_path: Path) -> None:
    split_root = tmp_path / "test"
    split_root.mkdir()
    prefix = split_root / "sample"
    (split_root / "sample_sequence_meta.json").write_text(json.dumps({
        "generator_version": "v17_pose_aligned_trace",
        "split_kind": "main",
        "frame_count": 2,
    }))
    (split_root / "sample_background.exr").touch()
    for index in range(2):
        (split_root / f"sample_frame{index:03d}_I.exr").touch()
    dataset = baseline.open_split_dataset(
        tmp_path, "test", 2, strict_contract=False,
    )
    assert len(dataset) == 1
    assert dataset.records[0].metadata["split_kind"] == "main"
