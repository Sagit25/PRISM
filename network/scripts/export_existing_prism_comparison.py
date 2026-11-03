#!/usr/bin/env python3
"""Convert saved PRISM qualitative tensors to the common comparison format.

The exporter only uses existing test predictions.  It checks sequence IDs
against the held-out manifest; it never substitutes training images or GT.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from refractive_mam2.dataset import RCTransPRISMDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, default=4)
    parser.add_argument("--max-sequences", type=int, default=8)
    parser.add_argument("--source-run", required=True)
    args = parser.parse_args()
    if not 0 <= args.frame_index < 8 or args.max_sequences <= 0:
        parser.error("frame index must be 0..7 and sequence count positive")
    dataset = RCTransPRISMDataset(args.data_root / "test", clip_length=8, strict_contract=True)
    output = args.output_root / "predictions"
    output.mkdir(parents=True, exist_ok=True)
    keys = {
        "background": "background",
        "alpha": "alpha",
        "straight_foreground": "foreground",
        "refractive_flow": "flow",
        "reconstruction": "recomposition",
    }
    count = min(len(dataset), args.max_sequences)
    for index in range(count):
        sample = dataset[index]
        sequence_id = sample["sequence_id"]
        source = args.source_root / sequence_id / "prediction.pt"
        if not source.is_file():
            raise FileNotFoundError(f"no saved PRISM prediction for {sequence_id}: {source}")
        payload = torch.load(source, map_location="cpu", weights_only=True)
        values: dict[str, np.ndarray] = {
            "sequence_id": np.array(sequence_id),
            "foreground_semantics": np.array("physical_straight_F"),
        }
        for old, new in keys.items():
            tensor = payload[old]
            if old != "background":
                tensor = tensor[args.frame_index]
            values[new] = tensor.detach().float().numpy().astype(np.float32)
        expected = {"background": 3, "alpha": 1, "foreground": 3, "flow": 2, "recomposition": 3}
        for key, channels in expected.items():
            if values[key].shape != (channels, 64, 64):
                raise ValueError(f"{sequence_id}: unexpected {key} shape {values[key].shape}")
            if not np.isfinite(values[key]).all():
                raise ValueError(f"{sequence_id}: nonfinite {key} prediction")
        np.savez_compressed(output / f"{index:04d}.npz", **values)
        print(f"PRISM_EXISTING_PREDICTION_EXPORTED {index} {sequence_id}", flush=True)
    (args.output_root / "source_manifest.json").write_text(json.dumps({
        "source_run": args.source_run,
        "frame_index": args.frame_index,
        "sequence_count": count,
        "source_root": str(args.source_root),
        "input_privilege": "PRISM inference, no oracle test target",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
