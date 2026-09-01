"""Manifest-driven VOS and image/video-matting data for PRISM Stage 1."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from .losses import RefractiveGroundTruth


_KINDS = {"vos", "video_matting", "image_matting", "synthetic_physics"}


def _image(path: Path) -> Tensor:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("semantic manifests require Pillow") from exc
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).permute(2, 0, 1)


def _label(path: Path) -> Tensor:
    if path.suffix.lower() == ".npy":
        array = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    else:
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover
            raise ImportError("semantic manifests require Pillow") from exc
        array = np.asarray(Image.open(path), dtype=np.float32)
    if array.ndim == 3:
        array = array[..., 0]
    return torch.from_numpy(array.copy()).unsqueeze(0)


def _paths(root: Path, values: Sequence[str]) -> list[Path]:
    paths = [Path(value) for value in values]
    return [path if path.is_absolute() else root / path for path in paths]


@dataclass
class SemanticBatch:
    ground_truth: RefractiveGroundTruth
    dataset_kinds: list[str]
    sample_ids: list[str]

    def to(self, device: torch.device | str, non_blocking: bool = False) -> "SemanticBatch":
        values: dict[str, Tensor | None] = {}
        for name in self.ground_truth.__dataclass_fields__:
            value = getattr(self.ground_truth, name)
            values[name] = (
                None
                if value is None
                else value.to(device=device, non_blocking=non_blocking)
            )
        return SemanticBatch(
            RefractiveGroundTruth(**values),
            list(self.dataset_kinds),
            list(self.sample_ids),
        )


class ManifestSemanticDataset(Dataset[dict[str, Any]]):
    """Read a JSONL manifest without imposing an upstream dataset layout.

    Each record contains ``dataset_kind``, ``frames`` and either
    ``object_masks``, ``trimaps`` or ``alpha``. Paths are relative to the
    manifest. Matting masks are derived from alpha/trimap when omitted.
    """

    def __init__(
        self,
        manifests: Sequence[str | Path],
        *,
        image_size: int = 512,
        clip_length: int | None = None,
        seed: int = 0,
        random_horizontal_flip: bool = True,
    ) -> None:
        if image_size < 1:
            raise ValueError("semantic image_size must be positive")
        self.image_size = image_size
        self.clip_length = clip_length
        self.seed = seed
        self.random_horizontal_flip = random_horizontal_flip
        self.epoch = 0
        records: list[dict[str, Any]] = []
        for manifest_value in manifests:
            manifest = Path(manifest_value)
            with manifest.open(encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    kind = record.get("dataset_kind")
                    if kind not in _KINDS:
                        raise ValueError(
                            f"{manifest}:{line_number}: invalid dataset_kind={kind!r}"
                        )
                    frames = record.get("frames")
                    if not frames:
                        raise ValueError(f"{manifest}:{line_number}: frames are required")
                    record["_root"] = manifest.parent
                    record["_id"] = str(record.get("id", f"{manifest.name}:{line_number}"))
                    records.append(record)
        if not records:
            raise ValueError("semantic manifests contain no samples")
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rng(self, sample_id: str) -> random.Random:
        stable = sum((index + 1) * ord(char) for index, char in enumerate(sample_id))
        return random.Random(self.seed + self.epoch * 1_000_003 + stable)

    def _selection(self, count: int, sample_id: str) -> list[int]:
        if self.clip_length is None:
            return list(range(count))
        if count <= self.clip_length:
            return list(range(count)) + [count - 1] * (self.clip_length - count)
        start = self._rng(sample_id).randrange(count - self.clip_length + 1)
        return list(range(start, start + self.clip_length))

    def _resize(self, value: Tensor, *, mode: str) -> Tensor:
        kwargs = {"size": (self.image_size, self.image_size), "mode": mode}
        if mode == "bilinear":
            kwargs["align_corners"] = False
            kwargs["antialias"] = True
        return F.interpolate(value, **kwargs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        root = record["_root"]
        sample_id = record["_id"]
        frame_paths = _paths(root, record["frames"])
        selection = self._selection(len(frame_paths), sample_id)
        frame_paths = [frame_paths[position] for position in selection]
        frames = self._resize(
            torch.stack([_image(path) for path in frame_paths]),
            mode="bilinear",
        )

        def selected_labels(name: str) -> Tensor | None:
            values = record.get(name)
            if values is None:
                return None
            all_paths = _paths(root, values)
            if len(all_paths) != len(record["frames"]):
                raise ValueError(f"{sample_id}: {name} count must match frames")
            paths = [all_paths[position] for position in selection]
            return torch.stack([_label(path) for path in paths])

        masks = selected_labels("object_masks")
        trimaps = selected_labels("trimaps")
        alpha = selected_labels("alpha")
        if masks is not None:
            masks = (self._resize(masks.float(), mode="nearest") > 0.5).float()
        if alpha is not None:
            if alpha.max() > 1.0:
                alpha = alpha / 255.0
            alpha = self._resize(alpha.float(), mode="bilinear").clamp(0, 1)
        if trimaps is not None:
            trimaps = self._resize(trimaps.float(), mode="nearest").long()
            if trimaps.max() > 2:
                trimaps = torch.where(
                    trimaps <= 64,
                    torch.zeros_like(trimaps),
                    torch.where(trimaps >= 192, torch.full_like(trimaps, 2), torch.ones_like(trimaps)),
                )
        if trimaps is None and alpha is not None:
            trimaps = torch.ones_like(alpha, dtype=torch.long)
            trimaps = torch.where(alpha <= 0.05, torch.zeros_like(trimaps), trimaps)
            trimaps = torch.where(alpha >= 0.95, torch.full_like(trimaps, 2), trimaps)
        if masks is None:
            if alpha is not None:
                masks = (alpha > 0.01).float()
            elif trimaps is not None:
                masks = (trimaps > 0).float()
            else:
                raise ValueError(f"{sample_id}: a promptable object label is required")
        if record["dataset_kind"] != "vos" and trimaps is None:
            raise ValueError(f"{sample_id}: matting data requires trimaps or alpha")
        if trimaps is None:
            trimaps = torch.zeros_like(masks, dtype=torch.long)

        if self.random_horizontal_flip and self._rng(sample_id).random() < 0.5:
            frames = torch.flip(frames, dims=(-1,))
            masks = torch.flip(masks, dims=(-1,))
            trimaps = torch.flip(trimaps, dims=(-1,))

        return {
            "frames": frames,
            "object_mask": masks,
            "trimap": trimaps,
            "dataset_kind": record["dataset_kind"],
            "sample_id": sample_id,
        }


def semantic_collate(samples: Sequence[dict[str, Any]]) -> SemanticBatch:
    if not samples:
        raise ValueError("cannot collate an empty semantic batch")
    frame_counts = {sample["frames"].shape[0] for sample in samples}
    if len(frame_counts) != 1:
        raise ValueError("semantic samples in one batch must have equal clip length")
    return SemanticBatch(
        ground_truth=RefractiveGroundTruth(
            frames=torch.stack([sample["frames"] for sample in samples]),
            object_mask=torch.stack([sample["object_mask"] for sample in samples]),
            trimap=torch.stack([sample["trimap"] for sample in samples]),
        ),
        dataset_kinds=[str(sample["dataset_kind"]) for sample in samples],
        sample_ids=[str(sample["sample_id"]) for sample in samples],
    )
