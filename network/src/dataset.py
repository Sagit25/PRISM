"""RCTrans-v15/PRISM dataset integration.

The renderer stores linear-RGB EXRs and pixel-space backward correspondences.
This module is the single authoritative mapping from that on-disk contract to
``RefractiveGroundTruth`` used by the network losses.
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler

from .losses import RefractiveGroundTruth
from .renderer import recompose
from .trimap import build_transparency_trimap


def _cv2():
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "RCTrans EXR loading requires OpenCV. Install refractive-mam2[data]."
        ) from exc
    return cv2


def _read_exr(path: Path) -> np.ndarray:
    cv2 = _cv2()
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise OSError(f"Could not read EXR: {path}")
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        image = image[..., None]
    if image.shape[-1] >= 3:
        image = image[..., :3][..., ::-1].copy()  # OpenCV BGR -> linear RGB
    return image


def _read_mask(path: Path) -> np.ndarray:
    cv2 = _cv2()
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise OSError(f"Could not read mask: {path}")
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = np.asarray(mask, dtype=np.float32)
    if mask.max(initial=0.0) > 1.0:
        mask /= 255.0
    return (mask >= 0.5).astype(np.float32)


def _npy(path: Path) -> np.ndarray:
    return np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)


def _hwc3(value: np.ndarray, name: str) -> np.ndarray:
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"{name} must have shape [H,W,3], got {value.shape}")
    return value


def _hwc2(value: np.ndarray, name: str) -> np.ndarray:
    if value.ndim != 3 or value.shape[-1] != 2:
        raise ValueError(f"{name} must have shape [H,W,2], got {value.shape}")
    return value


def _hw(value: np.ndarray, name: str) -> np.ndarray:
    value = np.squeeze(value)
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [H,W], got {value.shape}")
    return value


def _chw(value: np.ndarray) -> Tensor:
    return torch.from_numpy(np.ascontiguousarray(value.transpose(2, 0, 1)))


def _one_channel(value: np.ndarray) -> Tensor:
    return torch.from_numpy(np.ascontiguousarray(value[None]))


@dataclass(frozen=True)
class RCTransSequence:
    prefix: Path
    metadata_path: Path
    frame_prefixes: tuple[Path, ...]
    pair_group_id: str
    background_path: str
    metadata: dict[str, Any]


@dataclass
class RCTransBatch:
    ground_truth: RefractiveGroundTruth
    sequence_ids: list[str]
    paired_background_group_ids: list[str]
    metadata: list[dict[str, Any]]

    @property
    def frames(self) -> Tensor:
        return self.ground_truth.frames

    def to(self, device: torch.device | str, non_blocking: bool = False) -> "RCTransBatch":
        values: dict[str, Tensor | None] = {}
        for name in self.ground_truth.__dataclass_fields__:
            value = getattr(self.ground_truth, name)
            values[name] = (
                None
                if value is None
                else value.to(device=device, non_blocking=non_blocking)
            )
        return RCTransBatch(
            ground_truth=RefractiveGroundTruth(**values),
            sequence_ids=self.sequence_ids,
            paired_background_group_ids=self.paired_background_group_ids,
            metadata=self.metadata,
        )


class RCTransPRISMDataset(Dataset[dict[str, Any]]):
    """Load the canonical RCTrans v15 PRISM sequence contract.

    One item is one fixed-camera sequence.  ``clip_length`` optionally takes a
    deterministic prefix clip (with ``frame_stride``); this keeps pairing exact
    because both backgrounds use the same frame indices.
    """

    REQUIRED_SUFFIXES = (
        "_I.exr",
        "_object_mask.png",
        "_alpha.npy",
        "_CF.exr",
        "_F.exr",
        "_T.exr",
        "_A.exr",
        "_Phi.npy",
        "_u.npy",
        "_R.exr",
        "_confidence.npy",
        "_phi_valid.png",
    )

    def __init__(
        self,
        root: str | Path,
        *,
        clip_length: int | None = None,
        frame_stride: int = 1,
        strict_contract: bool = True,
        contract_tolerance: float = 2e-2,
        foreground_threshold: float = 0.95,
        require_generator_version: str | None = "v15_prism_contract",
        require_split_kind: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.clip_length = clip_length
        self.frame_stride = frame_stride
        self.strict_contract = strict_contract
        self.contract_tolerance = contract_tolerance
        self.foreground_threshold = foreground_threshold
        if frame_stride < 1:
            raise ValueError("frame_stride must be at least one")
        if clip_length is not None and clip_length < 1:
            raise ValueError("clip_length must be positive")

        records: list[RCTransSequence] = []
        for metadata_path in sorted(self.root.rglob("*_sequence_meta.json")):
            prefix = Path(str(metadata_path)[: -len("_sequence_meta.json")])
            with metadata_path.open("r", encoding="utf-8") as file:
                metadata = json.load(file)
            if (
                require_generator_version is not None
                and metadata.get("generator_version") != require_generator_version
            ):
                raise ValueError(
                    f"{metadata_path}: expected generator_version="
                    f"{require_generator_version!r}, got {metadata.get('generator_version')!r}"
                )
            if (
                require_split_kind is not None
                and metadata.get("split_kind") != require_split_kind
            ):
                continue
            frame_paths = sorted(prefix.parent.glob(prefix.name + "_frame*_I.exr"))
            frame_prefixes = tuple(
                Path(str(path)[: -len("_I.exr")]) for path in frame_paths
            )
            if not frame_prefixes:
                raise FileNotFoundError(f"No frames found for {metadata_path}")
            expected_count = metadata.get("frame_count")
            if expected_count is not None and len(frame_prefixes) != int(expected_count):
                raise ValueError(
                    f"{metadata_path}: metadata has {expected_count} frames, "
                    f"found {len(frame_prefixes)}"
                )
            background_file = Path(str(prefix) + "_background.exr")
            if not background_file.is_file():
                raise FileNotFoundError(background_file)
            group_id = metadata.get("paired_background_group_id")
            if not group_id:
                group_id = f"unpaired:{prefix}"
            records.append(
                RCTransSequence(
                    prefix=prefix,
                    metadata_path=metadata_path,
                    frame_prefixes=frame_prefixes,
                    pair_group_id=str(group_id),
                    background_path=str(metadata.get("background_path", background_file)),
                    metadata=metadata,
                )
            )
        if not records:
            raise FileNotFoundError(f"No *_sequence_meta.json found below {self.root}")
        self.records = records
        if self.strict_contract:
            self._validate_pair_metadata()

    def _validate_pair_metadata(self) -> None:
        invariant_keys = (
            "operator_seed",
            "shape_path",
            "frame_count",
            "camera_pose",
            "camera_x_fov_deg",
            "ior",
            "material_transmission_rgb",
        )
        for group_id, indices in self.pair_groups.items():
            if len(indices) < 2:
                continue
            anchor = self.records[indices[0]]
            for index in indices[1:]:
                paired = self.records[index]
                for key in invariant_keys:
                    if (
                        key in anchor.metadata
                        and key in paired.metadata
                        and anchor.metadata[key] != paired.metadata[key]
                    ):
                        raise ValueError(
                            f"pair group {group_id!r} disagrees on {key}"
                        )
                for first_prefix, second_prefix in zip(
                    anchor.frame_prefixes, paired.frame_prefixes
                ):
                    first_pose = Path(str(first_prefix) + "_object_pose.npy")
                    second_pose = Path(str(second_prefix) + "_object_pose.npy")
                    if first_pose.is_file() and second_pose.is_file():
                        if not np.allclose(
                            _npy(first_pose), _npy(second_pose), atol=1e-6, rtol=0.0
                        ):
                            raise ValueError(
                                f"pair group {group_id!r} has different object poses"
                            )

    def __len__(self) -> int:
        return len(self.records)

    @property
    def pair_groups(self) -> dict[str, list[int]]:
        groups: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(self.records):
            groups[record.pair_group_id].append(index)
        return dict(groups)

    def _selected_frames(self, record: RCTransSequence) -> tuple[Path, ...]:
        selected = record.frame_prefixes[:: self.frame_stride]
        if self.clip_length is not None:
            selected = selected[: self.clip_length]
        if not selected:
            raise ValueError(f"Empty clip for {record.prefix}")
        return selected

    def _assert_files(self, prefix: Path) -> None:
        missing = [
            str(prefix) + suffix
            for suffix in self.REQUIRED_SUFFIXES
            if not Path(str(prefix) + suffix).is_file()
        ]
        if missing:
            raise FileNotFoundError("Missing RCTrans outputs: " + ", ".join(missing))

    def _load_frame(self, prefix: Path) -> dict[str, Tensor]:
        self._assert_files(prefix)
        image = _hwc3(_read_exr(Path(str(prefix) + "_I.exr")), "I")
        mask = _hw(_read_mask(Path(str(prefix) + "_object_mask.png")), "mask")
        alpha = _hw(_npy(Path(str(prefix) + "_alpha.npy")), "alpha")
        g = _hwc3(_read_exr(Path(str(prefix) + "_CF.exr")), "G")
        f_std = _hwc3(_read_exr(Path(str(prefix) + "_F.exr")), "F")
        color = _hwc3(_read_exr(Path(str(prefix) + "_T.exr")), "T")
        tau = _hwc3(_read_exr(Path(str(prefix) + "_A.exr")), "A")
        phi = _hwc2(_npy(Path(str(prefix) + "_Phi.npy")), "Phi")
        flow = _hwc2(_npy(Path(str(prefix) + "_u.npy")), "u")
        residual = _hwc3(_read_exr(Path(str(prefix) + "_R.exr")), "R")
        confidence = _hw(
            _npy(Path(str(prefix) + "_confidence.npy")), "confidence"
        )
        validity = _hw(
            _read_mask(Path(str(prefix) + "_phi_valid.png")), "phi_valid"
        )
        h, w = image.shape[:2]
        arrays = (mask, alpha, g, f_std, color, tau, phi, flow, residual, confidence, validity)
        if any(value.shape[:2] != (h, w) for value in arrays):
            raise ValueError(f"Spatial shape mismatch at {prefix}")

        if self.strict_contract:
            yy, xx = np.mgrid[:h, :w].astype(np.float32)
            expected_phi = np.stack((xx, yy), axis=-1) + flow
            valid = validity > 0.5
            if valid.any():
                phi_error = np.max(np.abs(phi[valid] - expected_phi[valid]))
                if phi_error > self.contract_tolerance:
                    raise ValueError(f"{prefix}: Phi != x+u ({phi_error:.6g})")
            tau_error = np.max(np.abs(tau - (1.0 - alpha[..., None]) * color))
            if tau_error > self.contract_tolerance:
                raise ValueError(f"{prefix}: tau factorization error {tau_error:.6g}")
            identifiable = alpha > 1e-3
            if identifiable.any():
                g_error = np.max(
                    np.abs(g[identifiable] - alpha[identifiable, None] * f_std[identifiable])
                )
                if g_error > self.contract_tolerance:
                    raise ValueError(f"{prefix}: G != alpha*F_std ({g_error:.6g})")
            if confidence.min() < -1e-6 or confidence.max() > 1.0 + 1e-6:
                raise ValueError(f"{prefix}: confidence is outside [0,1]")
            invalid_confidence = confidence[~valid]
            if invalid_confidence.size and invalid_confidence.max() > self.contract_tolerance:
                raise ValueError(f"{prefix}: confidence is nonzero outside phi_valid")

        return {
            "frames": _chw(image),
            "object_mask": _one_channel(mask),
            "alpha": _one_channel(alpha),
            "straight_foreground": _chw(f_std),
            "premultiplied_foreground": _chw(g),
            "color_transmission": _chw(color),
            "transmittance": _chw(tau),
            "source_coordinates": _chw(phi),
            "refractive_flow": _chw(flow),
            "residual": _chw(residual),
            "confidence": _one_channel(confidence),
            "refractive_validity": _one_channel(validity),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        frames = [self._load_frame(prefix) for prefix in self._selected_frames(record)]
        stacked = {
            name: torch.stack([frame[name] for frame in frames], dim=0)
            for name in frames[0]
        }
        stacked["trimap"] = build_transparency_trimap(
            stacked["object_mask"],
            stacked["alpha"],
            foreground_threshold=self.foreground_threshold,
        )
        background = _hwc3(
            _read_exr(Path(str(record.prefix) + "_background.exr")), "background"
        )
        stacked["counterfactual_background"] = _chw(background)
        if self.strict_contract:
            with torch.no_grad():
                background_video = stacked["counterfactual_background"].unsqueeze(0).expand(
                    stacked["frames"].shape[0], -1, -1, -1
                )
                reconstructed, _ = recompose(
                    stacked["alpha"],
                    stacked["premultiplied_foreground"],
                    background_video,
                    stacked["refractive_flow"],
                    transmittance=stacked["transmittance"],
                    residual=stacked["residual"],
                )
                reconstruction_error = (
                    reconstructed - stacked["frames"]
                ).abs().max().item()
                if reconstruction_error > self.contract_tolerance:
                    raise ValueError(
                        f"{record.prefix}: image-formation error "
                        f"{reconstruction_error:.6g}"
                    )
        return {
            "tensors": stacked,
            "sequence_id": str(record.prefix.relative_to(self.root)),
            "paired_background_group_id": record.pair_group_id,
            "metadata": record.metadata,
        }


def prism_collate(samples: Sequence[dict[str, Any]]) -> RCTransBatch:
    if not samples:
        raise ValueError("Cannot collate an empty sample list")
    tensor_names = tuple(samples[0]["tensors"])
    tensors = {
        name: torch.stack([sample["tensors"][name] for sample in samples], dim=0)
        for name in tensor_names
    }
    return RCTransBatch(
        ground_truth=RefractiveGroundTruth(**tensors),
        sequence_ids=[sample["sequence_id"] for sample in samples],
        paired_background_group_ids=[
            sample["paired_background_group_id"] for sample in samples
        ],
        metadata=[sample["metadata"] for sample in samples],
    )


class PairedBackgroundBatchSampler(Sampler[list[int]]):
    """Yield same-operator sequences rendered on distinct backgrounds."""

    def __init__(
        self,
        dataset: RCTransPRISMDataset,
        *,
        backgrounds_per_group: int = 2,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        if backgrounds_per_group < 2:
            raise ValueError("backgrounds_per_group must be at least two")
        self.dataset = dataset
        self.backgrounds_per_group = backgrounds_per_group
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        eligible: list[list[int]] = []
        for indices in dataset.pair_groups.values():
            distinct: dict[str, int] = {}
            for index in indices:
                record = dataset.records[index]
                distinct.setdefault(record.background_path, index)
            if len(distinct) >= backgrounds_per_group:
                eligible.append(list(distinct.values()))
        if not eligible:
            raise ValueError(
                "No paired_background_group_id contains enough distinct backgrounds"
            )
        self.groups = eligible

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.groups)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        groups = [list(group) for group in self.groups]
        if self.shuffle:
            rng.shuffle(groups)
        for group in groups:
            if self.shuffle:
                rng.shuffle(group)
            yield group[: self.backgrounds_per_group]


def build_paired_prism_dataloader(
    dataset: RCTransPRISMDataset,
    *,
    backgrounds_per_group: int = 2,
    shuffle: bool = True,
    seed: int = 0,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader[RCTransBatch]:
    sampler = PairedBackgroundBatchSampler(
        dataset,
        backgrounds_per_group=backgrounds_per_group,
        shuffle=shuffle,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=prism_collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
