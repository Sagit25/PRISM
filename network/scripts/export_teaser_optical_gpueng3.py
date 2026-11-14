"""Export final PRISM optical factors for four selected 128px teaser videos.

Run alongside the existing GPUeng3 trace evaluator only after it exits. The
underlying checkpoint, test split, prompts, and inverse-priority rule remain
the same; the loader skips all but PRISM_TEASER_SELECTED_IDS.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch.nn.functional as F

import evaluate_inverse_priority_trace as trace


SELECTED = set(os.environ["PRISM_TEASER_SELECTED_IDS"].split(","))
if len(SELECTED) != 4 or not all(SELECTED):
    raise ValueError("Expected exactly four nonempty sequence IDs")
trace.EXPECTED = len(SELECTED)
EXPORT = Path(os.environ["PRISM_TEASER_OPTICAL_ROOT"])
EXPORT.mkdir(parents=True, exist_ok=True)


class SelectedLoader:
    def __init__(self, source):
        self.source = source

    def __iter__(self):
        for batch in self.source:
            ids = list(getattr(batch, "sequence_ids", []) or [])
            if len(ids) != 1:
                raise RuntimeError(f"Expected batch size one, got {ids}")
            if str(ids[0]) in SELECTED:
                trace.state.ids = ids
                yield batch


original_score = trace._score_trace


def object_support_for_frame(prediction, frame):
    """Use PAM support when available, else the checkpoint's tracker mask."""
    matter = prediction.matter
    support_logits = matter.object_support_logits
    if support_logits is None:
        support_logits = prediction.backbone.mask_logits
    support_logits = support_logits[0, frame]
    if support_logits.shape[-2:] != matter.alpha.shape[-2:]:
        support_logits = F.interpolate(
            support_logits[None],
            size=matter.alpha.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0]
    return support_logits >= 0


def score_and_export(prediction, target, kwargs):
    ids = list(trace.state.ids or [])
    if len(ids) != 1 or str(ids[0]) not in SELECTED:
        raise RuntimeError(f"Unexpected sequence: {ids}")
    seq = str(ids[0])
    frame = trace.FRAME
    matter = prediction.matter
    support = object_support_for_frame(prediction, frame)
    gt_background = target.counterfactual_background
    if gt_background.ndim == 5:
        gt_background = gt_background[:, 0]
    fields = {
        "sequence_id": np.asarray(seq),
        "input": trace._numpy(target.frames[0, frame]).astype(np.float32),
        "gt_background": trace._numpy(gt_background[0]).astype(np.float32),
        "alpha": trace._numpy(matter.alpha[0, frame]).astype(np.float32),
        "straight_foreground": trace._numpy(
            matter.straight_foreground[0, frame]).astype(np.float32),
        "premultiplied_foreground": trace._numpy(
            matter.premultiplied_foreground[0, frame]).astype(np.float32),
        "transmittance": trace._numpy(matter.transmittance[0, frame]).astype(np.float32),
        "refractive_flow": trace._numpy(
            matter.refractive_flow[0, frame]).astype(np.float32),
        "residual": trace._numpy(matter.residual[0, frame]).astype(np.float32),
        "background": trace._numpy(
            prediction.background.background[0]).astype(np.float32),
        "reconstruction": trace._numpy(
            prediction.reconstructed_frames[0, frame]).astype(np.float32),
        "object_support": trace._numpy(
            support).astype(bool),
    }
    for key, value in fields.items():
        if key != "sequence_id" and not np.isfinite(value).all():
            raise FloatingPointError(f"Nonfinite {key} for {seq}")
    if matter.refractive_kernel_weights is not None:
        fields["refractive_kernel_weights"] = trace._numpy(
            matter.refractive_kernel_weights[0, frame]).astype(np.float32)
    if matter.refractive_kernel_flows is not None:
        fields["refractive_kernel_flows"] = trace._numpy(
            matter.refractive_kernel_flows[0, frame]).astype(np.float32)
    original_score(prediction, target, kwargs)
    np.savez_compressed(EXPORT / f"{seq}.npz", **fields)
    print(f"PRISM_TEASER_OPTICAL_EXPORTED {seq}", flush=True)


def main():
    trace.RecordingLoader = SelectedLoader
    trace._score_trace = score_and_export
    trace.main()
    found = {p.stem for p in EXPORT.glob("*.npz")}
    if found != SELECTED:
        raise RuntimeError(f"Missing teaser fields: {sorted(SELECTED - found)}")
    print("PRISM_TEASER_OPTICAL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
