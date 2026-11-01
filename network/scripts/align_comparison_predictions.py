#!/usr/bin/env python3
"""Index sequence-ID prediction files in the canonical held-out test order."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from refractive_mam2.dataset import RCTransPRISMDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--max-sequences", type=int, default=0)
    args = parser.parse_args()
    if args.max_sequences < 0:
        parser.error("max-sequences must be nonnegative")
    dataset = RCTransPRISMDataset(args.data_root / "test", clip_length=8, strict_contract=True)
    count = min(len(dataset), args.max_sequences or len(dataset))
    output = args.prediction_root / "predictions"
    output.mkdir(parents=True, exist_ok=True)
    by_id = args.prediction_root / "predictions_by_sequence_id"
    for index in range(count):
        sequence_id = dataset[index]["sequence_id"]
        source = by_id / f"{sequence_id}.npz"
        if not source.is_file():
            raise FileNotFoundError(f"missing PRISM prediction for test index {index}: {sequence_id}")
        with np.load(source, allow_pickle=False) as archive:
            if str(archive["sequence_id"].item()) != sequence_id:
                raise ValueError(f"sequence ID mismatch in {source}")
        target = output / f"{index:04d}.npz"
        if target.is_symlink() or target.exists():
            if target.resolve() != source.resolve():
                raise FileExistsError(f"existing indexed prediction differs: {target}")
        else:
            target.symlink_to(source)
    print(f"PRISM_COMPARISON_ALIGNED sequences={count} root={output}", flush=True)


if __name__ == "__main__":
    main()
