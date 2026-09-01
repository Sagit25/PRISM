#!/usr/bin/env python3
"""Validate PRISM v15 files, equations, and paired-background invariants."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np


FRAME_SUFFIXES = (
    "_I.exr",
    "_object_mask.png",
    "_alpha.npy",
    "_CF.exr",
    "_F.exr",
    "_T.exr",
    "_A.exr",
    "_Phi.npy",
    "_Phi_src.npy",
    "_u.npy",
    "_R.exr",
    "_confidence.npy",
    "_phi_valid.png",
    "_object_pose.npy",
)


def read_exr(path: Path) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if value is None:
        raise RuntimeError(f"Could not read EXR: {path}")
    if value.ndim == 2:
        value = value[..., None]
    return np.asarray(value[..., :3], dtype=np.float32)


def frame_prefixes(sequence_prefix: Path) -> list[Path]:
    paths = sorted(
        sequence_prefix.parent.glob(sequence_prefix.name + "_frame*_I.exr")
    )
    return [Path(str(path)[: -len("_I.exr")]) for path in paths]


def max_abs(value: np.ndarray) -> float:
    return float(np.max(np.abs(value))) if value.size else 0.0


def validate_frame(prefix: Path) -> dict[str, float]:
    missing = [
        str(prefix) + suffix
        for suffix in FRAME_SUFFIXES
        if not Path(str(prefix) + suffix).exists()
    ]
    if missing:
        raise FileNotFoundError("Missing frame outputs:\n" + "\n".join(missing))

    sequence_prefix = Path(str(prefix).rsplit("_frame", 1)[0])
    background_path = Path(str(sequence_prefix) + "_background.exr")
    if not background_path.exists():
        raise FileNotFoundError(background_path)

    image = read_exr(Path(str(prefix) + "_I.exr"))
    premultiplied = read_exr(Path(str(prefix) + "_CF.exr"))
    tau = read_exr(Path(str(prefix) + "_A.exr"))
    color_transmission = read_exr(Path(str(prefix) + "_T.exr"))
    residual = read_exr(Path(str(prefix) + "_R.exr"))
    background = read_exr(background_path)
    alpha = np.load(str(prefix) + "_alpha.npy").astype(np.float32)
    phi = np.load(str(prefix) + "_Phi.npy").astype(np.float32)
    phi_alias = np.load(str(prefix) + "_Phi_src.npy").astype(np.float32)
    displacement = np.load(str(prefix) + "_u.npy").astype(np.float32)
    confidence = np.load(str(prefix) + "_confidence.npy").astype(np.float32)
    valid_raw = cv2.imread(
        str(prefix) + "_phi_valid.png", cv2.IMREAD_UNCHANGED
    )
    if valid_raw is None:
        raise RuntimeError(f"Could not read validity mask for {prefix}")
    valid = valid_raw > 0

    height, width = alpha.shape
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    grid = np.stack([xx, yy], axis=-1)
    source_from_u = grid + displacement

    phi_error = (
        max_abs(phi[valid] - source_from_u[valid]) if np.any(valid) else 0.0
    )
    alias_error = max_abs(phi - phi_alias)
    factorization_error = max_abs(
        tau - (1.0 - alpha[..., None]) * color_transmission
    )

    warped_background = cv2.remap(
        background,
        source_from_u[..., 0].astype(np.float32),
        source_from_u[..., 1].astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    formation_error = max_abs(
        image - (premultiplied + tau * warped_background + residual)
    )

    arrays = (
        image,
        premultiplied,
        tau,
        color_transmission,
        residual,
        alpha,
        phi,
        displacement,
        confidence,
    )
    if not all(np.isfinite(value).all() for value in arrays):
        raise AssertionError(f"NaN or Inf found in {prefix}")
    if float(confidence.min()) < -1e-6 or float(confidence.max()) > 1.0 + 1e-6:
        raise AssertionError(f"Confidence is outside [0,1] in {prefix}")

    return {
        "phi_error": phi_error,
        "alias_error": alias_error,
        "factorization_error": factorization_error,
        "formation_error": formation_error,
        "valid_fraction": float(valid.mean()),
    }


def load_sequences(root: Path) -> list[dict]:
    sequences = []
    for meta_path in sorted(root.glob("*_sequence_meta.json")):
        with meta_path.open() as handle:
            metadata = json.load(handle)
        prefix = Path(str(meta_path)[: -len("_sequence_meta.json")])
        frames = frame_prefixes(prefix)
        if not frames:
            raise FileNotFoundError(f"No frames found for {prefix}")
        if metadata.get("generator_version") != "v15_prism_contract":
            raise AssertionError(f"Unexpected generator version: {meta_path}")
        if metadata.get("split_kind") == "main" and float(
            metadata.get("reflection_scale", 1.0)
        ) != 0.0:
            raise AssertionError(f"Main split contains reflection: {meta_path}")
        sequences.append(
            {"prefix": prefix, "frames": frames, "metadata": metadata}
        )
    if not sequences:
        raise FileNotFoundError(f"No *_sequence_meta.json under {root}")
    return sequences


def validate_dataset_manifest(result_dir: Path) -> dict:
    manifest_path = result_dir / "dataset_manifest.json"
    if not manifest_path.is_file():
        manifest_path = result_dir.parent / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"dataset_manifest.json is required beside split directories: {result_dir}"
        )
    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    resources = manifest.get("resources", {})
    required = ("train", "validation", "test")
    if any(split not in resources for split in required):
        raise AssertionError("manifest must contain train/validation/test resources")
    for kind in ("shapes", "backgrounds"):
        sets = {
            split: set(resources[split].get(kind, ()))
            for split in required
        }
        for index, first in enumerate(required):
            for second in required[index + 1 :]:
                overlap = sets[first] & sets[second]
                if overlap:
                    raise AssertionError(
                        f"{kind} leakage between {first} and {second}: "
                        f"{sorted(overlap)[:5]}"
                    )
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    print(f"MANIFEST PASS sha256={digest} path={manifest_path}")
    return manifest


def validate_sequence_partition(
    sequences: list[dict], result_dir: Path, manifest: dict
) -> None:
    split = result_dir.name
    if split not in ("train", "validation", "test"):
        return
    resources = manifest["resources"][split]
    shapes = set(resources.get("shapes", ()))
    backgrounds = set(resources.get("backgrounds", ()))
    for sequence in sequences:
        metadata = sequence["metadata"]
        if metadata.get("shape_path") not in shapes:
            raise AssertionError(
                f"{sequence['prefix']}: shape is outside manifest split {split}"
            )
        if metadata.get("background_path") not in backgrounds:
            raise AssertionError(
                f"{sequence['prefix']}: background is outside manifest split {split}"
            )


def compare_paired_groups(
    sequences: list[dict], pair_tolerance: float, require_pairs: bool
) -> int:
    groups = defaultdict(list)
    for sequence in sequences:
        group_id = sequence["metadata"].get("paired_background_group_id")
        if group_id:
            groups[group_id].append(sequence)

    paired_group_count = 0
    for group_id, members in groups.items():
        if len(members) < 2:
            continue
        paired_group_count += 1
        reference = members[0]
        reference_meta = reference["metadata"]
        for member in members[1:]:
            metadata = member["metadata"]
            for key in (
                "operator_seed",
                "ior",
                "material_transmission_rgb",
                "camera_pose",
                "camera_x_fov_deg",
            ):
                if metadata.get(key) != reference_meta.get(key):
                    raise AssertionError(
                        f"Paired group {group_id} differs in {key}"
                    )
            if len(member["frames"]) != len(reference["frames"]):
                raise AssertionError(f"Frame-count mismatch in {group_id}")
            for ref_frame, frame in zip(reference["frames"], member["frames"]):
                for suffix in (
                    "_alpha.npy",
                    "_Phi.npy",
                    "_u.npy",
                    "_object_pose.npy",
                ):
                    error = max_abs(
                        np.load(str(ref_frame) + suffix)
                        - np.load(str(frame) + suffix)
                    )
                    if error > pair_tolerance:
                        raise AssertionError(
                            f"Paired group {group_id} {suffix} error={error}"
                        )
        backgrounds = {
            member["metadata"].get("background_path") for member in members
        }
        if len(backgrounds) < 2:
            raise AssertionError(
                f"Paired group {group_id} has no background diversity"
            )

    if require_pairs and paired_group_count == 0:
        raise AssertionError("No paired-background group with >=2 members")
    return paired_group_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--formation-tol", type=float, default=2e-3)
    parser.add_argument("--factorization-tol", type=float, default=5e-4)
    parser.add_argument("--coordinate-tol", type=float, default=1e-4)
    parser.add_argument("--pair-tol", type=float, default=5e-4)
    parser.add_argument("--require-pairs", action="store_true")
    parser.add_argument(
        "--skip-manifest",
        action="store_true",
        help="allow legacy/smoke outputs without a dataset manifest",
    )
    args = parser.parse_args()

    manifest = None
    if not args.skip_manifest:
        manifest = validate_dataset_manifest(args.result_dir)
    sequences = load_sequences(args.result_dir)
    if manifest is not None:
        validate_sequence_partition(sequences, args.result_dir, manifest)
    metrics = []
    for sequence in sequences:
        metrics.extend(validate_frame(frame) for frame in sequence["frames"])

    maxima = {
        key: max(metric[key] for metric in metrics)
        for key in (
            "phi_error",
            "alias_error",
            "factorization_error",
            "formation_error",
        )
    }
    if maxima["phi_error"] > args.coordinate_tol:
        raise AssertionError(f"Phi=x+u failed: {maxima['phi_error']}")
    if maxima["alias_error"] > args.coordinate_tol:
        raise AssertionError(f"Phi_src alias failed: {maxima['alias_error']}")
    if maxima["factorization_error"] > args.factorization_tol:
        raise AssertionError(
            f"tau=(1-alpha)C failed: {maxima['factorization_error']}"
        )
    if maxima["formation_error"] > args.formation_tol:
        raise AssertionError(
            f"I=G+tau*B(Phi)+R failed: {maxima['formation_error']}"
        )

    pair_count = compare_paired_groups(
        sequences, args.pair_tol, args.require_pairs
    )
    print(
        "PASS "
        f"sequences={len(sequences)} frames={len(metrics)} "
        f"paired_groups={pair_count} "
        f"max_phi_error={maxima['phi_error']:.6g} "
        f"max_tau_error={maxima['factorization_error']:.6g} "
        f"max_formation_error={maxima['formation_error']:.6g}"
    )


if __name__ == "__main__":
    main()
