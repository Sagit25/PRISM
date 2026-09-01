#!/usr/bin/env python3
"""Load and validate the RCTrans-v15 tensors consumed by the network."""

from __future__ import annotations

import argparse

from refractive_mam2 import (
    PairedBackgroundBatchSampler,
    RCTransPRISMDataset,
    prism_collate,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--clip-length", type=int)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--split-kind")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--check-pairs", action="store_true")
    parser.add_argument("--tolerance", type=float, default=2e-2)
    args = parser.parse_args()

    dataset = RCTransPRISMDataset(
        args.root,
        clip_length=args.clip_length,
        frame_stride=args.frame_stride,
        strict_contract=True,
        contract_tolerance=args.tolerance,
        require_split_kind=args.split_kind,
    )
    count = len(dataset) if args.limit <= 0 else min(len(dataset), args.limit)
    for index in range(count):
        sample = dataset[index]
        batch = prism_collate([sample])
        target = batch.ground_truth
        print(
            f"[{index + 1}/{count}] {batch.sequence_ids[0]} "
            f"frames={tuple(target.frames.shape)} "
            f"group={batch.paired_background_group_ids[0]}"
        )
    if args.check_pairs:
        sampler = PairedBackgroundBatchSampler(dataset, shuffle=False)
        pair_count = sum(1 for _ in sampler)
        print(f"paired groups: {pair_count}")
    print(f"validated sequences: {count}/{len(dataset)}")


if __name__ == "__main__":
    main()
