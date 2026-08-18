from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch import nn

from .config import PipelineConfig
from .pipeline import RefractiveMAM2, RefractiveMAM2Output
from .sam2_integration import MAM2FrameOutput, MAM2VideoPredictor
from .types import MAM2BackboneOutput


class _ExternallyPropagatedBackbone(nn.Module):
    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "this pipeline receives official SAM2 outputs through forward_from_backbone"
        )


def build_physics_pipeline_for_sam2(
    config: PipelineConfig | None = None,
) -> RefractiveMAM2:
    return RefractiveMAM2(_ExternallyPropagatedBackbone(), config)


def _select_object_record(
    records: list[MAM2FrameOutput],
    object_ids: list[Any],
    object_id: Any,
) -> MAM2FrameOutput:
    if object_id not in object_ids:
        raise KeyError(f"object_id {object_id!r} is absent from predictor output {object_ids!r}")
    if len(records) < len(object_ids):
        raise RuntimeError(
            "SAM2 produced fewer cached MSS records than object masks; "
            "disable vos_optimized and use the supported official predictor"
        )
    # Interactions before propagation may also call _track_step. The runner
    # clears the cache first; taking the final N is an additional safeguard.
    aligned = records[-len(object_ids) :]
    return aligned[object_ids.index(object_id)]


def propagate_mam2_backbone(
    predictor: MAM2VideoPredictor,
    inference_state: dict[str, Any],
    *,
    object_id: Any,
    start_frame_idx: int | None = None,
    max_frame_num_to_track: int | None = None,
    reverse: bool = False,
) -> MAM2BackboneOutput:
    """Use the official public propagation API and collect PDD/MSS tensors."""

    predictor.clear_mam2_cache()
    selected: dict[int, MAM2FrameOutput] = {}
    iterator: Iterator = predictor.propagate_in_video(
        inference_state,
        start_frame_idx=start_frame_idx,
        max_frame_num_to_track=max_frame_num_to_track,
        reverse=reverse,
    )
    for frame_index, object_ids, _ in iterator:
        ids = list(object_ids)
        records = predictor.pop_mam2_frame_outputs(int(frame_index))
        selected[int(frame_index)] = _select_object_record(records, ids, object_id)

    if not selected:
        raise RuntimeError("SAM2 propagation produced no frames; add a first-frame prompt")
    ordered_indices = sorted(selected, reverse=reverse)
    ordered = [selected[index] for index in ordered_indices]

    device = next(predictor.parameters()).device
    mask = torch.stack([item.mask_logits[0] for item in ordered], dim=0).unsqueeze(0).to(device)
    trimap = torch.stack([item.trimap_logits[0] for item in ordered], dim=0).unsqueeze(0).to(device)
    features = torch.stack(
        [item.non_memory_features[0] for item in ordered], dim=0
    ).unsqueeze(0).to(device)
    output = MAM2BackboneOutput(
        mask_logits=mask,
        trimap_logits=trimap,
        non_memory_features=features,
    )
    output.validate()
    return output


class SAM2RefractiveRunner:
    """End-to-end official SAM2 -> PDD/MSS -> refractive matting runner."""

    def __init__(
        self,
        predictor: MAM2VideoPredictor,
        physics_pipeline: RefractiveMAM2,
    ) -> None:
        self.predictor = predictor
        self.physics_pipeline = physics_pipeline

    def run(
        self,
        frames: Tensor,
        inference_state: dict[str, Any],
        *,
        object_id: Any,
    ) -> RefractiveMAM2Output:
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError("frames must have shape [T,3,H,W] in RGB [0,1]")
        semantics = propagate_mam2_backbone(
            self.predictor,
            inference_state,
            object_id=object_id,
        )
        if semantics.mask_logits.shape[1] != frames.shape[0]:
            raise ValueError(
                "frames length must equal the number of propagated frames; "
                "run full forward propagation for end-to-end decomposition"
            )
        target_device = next(self.physics_pipeline.parameters()).device
        video = frames.unsqueeze(0).to(target_device)
        semantics = MAM2BackboneOutput(
            mask_logits=semantics.mask_logits.to(target_device),
            trimap_logits=semantics.trimap_logits.to(target_device),
            non_memory_features=semantics.non_memory_features.to(target_device),
        )
        return self.physics_pipeline.forward_from_backbone(video, semantics)


def save_refractive_checkpoint(
    path: str | Path,
    predictor: MAM2VideoPredictor,
    physics_pipeline: RefractiveMAM2,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    torch.save(
        {
            "format_version": 4,
            "predictor_mam2": predictor.mam2_extension_state_dict(),
            "physics_pipeline": physics_pipeline.state_dict(),
            "pipeline_config": asdict(physics_pipeline.config),
            "metadata": metadata or {},
        },
        Path(path),
    )


def load_refractive_checkpoint(
    path: str | Path,
    predictor: MAM2VideoPredictor,
    physics_pipeline: RefractiveMAM2,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 4:
        raise RuntimeError(
            "unsupported refractive checkpoint format; version 4 records the "
            "resolution-aware refractive-flow convention. A v3 checkpoint must "
            "be migrated explicitly with flow_parameterization='fixed_pixels'."
        )
    saved_matter = payload.get("pipeline_config", {}).get("matter", {})
    current_matter = physics_pipeline.config.matter
    for key in (
        "flow_parameterization",
        "max_refractive_flow_fraction",
        "max_refractive_flow",
    ):
        current = getattr(current_matter, key)
        if key in saved_matter and saved_matter[key] != current:
            raise RuntimeError(
                f"checkpoint matter config mismatch for {key}: "
                f"saved={saved_matter[key]!r}, current={current!r}"
            )
    predictor.load_mam2_extension_state_dict(payload["predictor_mam2"], strict=strict)
    physics_pipeline.load_state_dict(payload["physics_pipeline"], strict=strict)
    return dict(payload.get("metadata", {}))
