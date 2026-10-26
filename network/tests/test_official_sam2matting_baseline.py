"""Metrics used by the released-weight SAM2Matting comparison run."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


MODULE = Path(__file__).parents[1] / "scripts" / "evaluate_official_sam2matting.py"
SPEC = importlib.util.spec_from_file_location("sam2matting_baseline", MODULE)
assert SPEC is not None and SPEC.loader is not None
baseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(baseline)


def test_prompt_frame_excluded_from_propagated_metric() -> None:
    truth = torch.zeros(1, 2, 1, 2, 2)
    prediction = truth.clone()
    prediction[:, 1] = 1
    mask_truth = truth.clone()
    mask_logits = torch.full_like(truth, -1)
    mask_logits[:, 1] = 1
    all_frames = baseline.metrics_from_totals(
        baseline.score_clip(
            prediction, truth, mask_logits, mask_truth, skip_prompt_frame=False
        )
    )
    propagated = baseline.metrics_from_totals(
        baseline.score_clip(
            prediction, truth, mask_logits, mask_truth, skip_prompt_frame=True
        )
    )
    assert all_frames["alpha_mae"] == 0.5
    assert propagated["alpha_mae"] == 1.0
    assert propagated["mask_iou"] == 0.0
    assert propagated["frames"] == 1


def test_per_sequence_totals_use_pixel_weighting() -> None:
    destination: dict[str, float] = {}
    baseline.add_totals(destination, {"alpha_abs_sum": 1.0, "alpha_pixels": 2.0})
    baseline.add_totals(destination, {"alpha_abs_sum": 1.0, "alpha_pixels": 8.0})
    assert destination == {"alpha_abs_sum": 2.0, "alpha_pixels": 10.0}
