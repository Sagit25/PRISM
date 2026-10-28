from __future__ import annotations

import os
import warnings
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch import nn

from .config import PipelineConfig
from .color import srgb_to_linear
from .pipeline import RefractiveMAM2, RefractiveMAM2Output
from .matter import PhysicsAwareMatter
from .pdd import PromptableDualModeDecoder
from .sam2_integration import MAM2FrameOutput, MAM2VideoPredictor
from .types import MAM2BackboneOutput


class _ExternallyPropagatedBackbone(nn.Module):
    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "this pipeline receives official SAM2 outputs through forward_from_backbone"
        )


def _mam2_architecture(predictor: MAM2VideoPredictor) -> str:
    """Return the serialized MAM2 contract, including for lightweight test doubles."""

    return getattr(
        predictor,
        "mam2_architecture",
        PromptableDualModeDecoder.ARCHITECTURE,
    )


def _prism_architecture(physics_pipeline: RefractiveMAM2) -> str:
    matter = getattr(physics_pipeline, "matter", None)
    return getattr(matter, "ARCHITECTURE", PhysicsAwareMatter.ARCHITECTURE)


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
        raise KeyError(
            f"object_id {object_id!r} is absent from predictor output {object_ids!r}"
        )
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
    rgb_frames: Tensor | None = None,
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
        raise RuntimeError(
            "SAM2 propagation produced no frames; add a first-frame prompt"
        )
    ordered_indices = sorted(selected, reverse=reverse)
    ordered = [selected[index] for index in ordered_indices]

    device = next(predictor.parameters()).device
    mask = (
        torch.stack([item.mask_logits[0] for item in ordered], dim=0)
        .unsqueeze(0)
        .to(device)
    )
    trimap = (
        torch.stack([item.trimap_logits[0] for item in ordered], dim=0)
        .unsqueeze(0)
        .to(device)
    )
    features = (
        torch.stack([item.non_memory_features[0] for item in ordered], dim=0)
        .unsqueeze(0)
        .to(device)
    )
    if rgb_frames is None:
        raise ValueError(
            "rgb_frames are required to run MAM2's final RGB+trimap alpha matter"
        )
    if rgb_frames.ndim == 4:
        rgb_frames = rgb_frames.unsqueeze(0)
    if rgb_frames.ndim != 5 or rgb_frames.shape[2] != 3:
        raise ValueError("rgb_frames must have shape [T,3,H,W] or [B,T,3,H,W]")
    if ordered_indices and max(ordered_indices) < rgb_frames.shape[1]:
        rgb_frames = rgb_frames[:, ordered_indices]
    if rgb_frames.shape[:2] != mask.shape[:2]:
        raise ValueError("rgb_frames must align with propagated MAM2 frames")
    rgb_frames = rgb_frames.to(device)
    alpha = predictor.predict_mam2_alpha(rgb_frames, trimap)
    output = MAM2BackboneOutput(
        mask_logits=mask,
        trimap_logits=trimap,
        alpha_matte=alpha,
        non_memory_features=features,
    )
    output.validate()
    return output


class SAM2RefractiveRunner:
    """End-to-end SAM2 -> PDD/MSS -> MAM2 alpha -> PRISM runner."""

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
        frames_are_srgb: bool = True,
    ) -> RefractiveMAM2Output:
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError("frames must have shape [T,3,H,W] in RGB [0,1]")
        semantics = propagate_mam2_backbone(
            self.predictor,
            inference_state,
            object_id=object_id,
            rgb_frames=frames,
        )
        if semantics.mask_logits.shape[1] != frames.shape[0]:
            raise ValueError(
                "frames length must equal the number of propagated frames; "
                "run full forward propagation for end-to-end decomposition"
            )
        target_device = next(self.physics_pipeline.parameters()).device
        physics_frames = srgb_to_linear(frames) if frames_are_srgb else frames
        video = physics_frames.unsqueeze(0).to(target_device)
        semantics = MAM2BackboneOutput(
            mask_logits=semantics.mask_logits.to(target_device),
            trimap_logits=semantics.trimap_logits.to(target_device),
            alpha_matte=semantics.alpha_matte.to(target_device),
            non_memory_features=semantics.non_memory_features.to(target_device),
        )
        return self.physics_pipeline.forward_from_backbone(video, semantics)


def save_refractive_checkpoint(
    path: str | Path,
    predictor: MAM2VideoPredictor,
    physics_pipeline: RefractiveMAM2,
    *,
    metadata: dict[str, Any] | None = None,
    training_state: dict[str, Any] | None = None,
) -> None:
    physics_state = {
        name: value
        for name, value in physics_pipeline.state_dict().items()
        if not name.startswith("backbone.")
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    torch.save(
        {
            "format_version": 8,
            "mam2_architecture": _mam2_architecture(predictor),
            "prism_architecture": _prism_architecture(physics_pipeline),
            "predictor_mam2": predictor.mam2_extension_state_dict(),
            "mam2_integration_config": asdict(predictor.mam2_integration_config),
            "physics_pipeline": physics_state,
            "pipeline_config": asdict(physics_pipeline.config),
            "metadata": metadata or {},
            "training_state": training_state or {},
        },
        temporary,
    )
    os.replace(temporary, destination)


def load_refractive_checkpoint(
    path: str | Path,
    predictor: MAM2VideoPredictor,
    physics_pipeline: RefractiveMAM2,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 8:
        raise RuntimeError(
            "unsupported refractive checkpoint format; version 8 adds the "
            "full multi-scale temporal PRISM head and deformable 3x3 refractive "
            "kernel. Earlier heads are intentionally incompatible and Stage 2 "
            "must be retrained."
        )
    current_architecture = _mam2_architecture(predictor)
    if payload.get("mam2_architecture") != current_architecture:
        raise RuntimeError(
            "checkpoint MAM2 architecture mismatch: "
            f"saved={payload.get('mam2_architecture')!r}, "
            f"current={current_architecture!r}"
        )
    current_prism_architecture = _prism_architecture(physics_pipeline)
    saved_prism_architecture = payload.get("prism_architecture")
    saved_matter = payload.get("pipeline_config", {}).get("matter", {})
    current_matter = physics_pipeline.config.matter
    adding_foreground_refiner = (
        not bool(saved_matter.get("refine_straight_foreground", False))
        and current_matter.refine_straight_foreground
        and current_prism_architecture
        == f"{saved_prism_architecture}-foreground-residual-v1"
    )
    adding_foreground_detail_refiner = (
        not bool(saved_matter.get("refine_straight_foreground_detail", False))
        and current_matter.refine_straight_foreground_detail
        and current_prism_architecture
        == f"{saved_prism_architecture}-foreground-coarse-fine-v1"
    )
    adding_mam2_flow_bridge = (
        not bool(saved_matter.get("flow_guided_latent_bridge", False))
        and not bool(saved_matter.get("use_mam2_foreground_prior", False))
        and current_matter.flow_guided_latent_bridge
        and current_matter.use_mam2_foreground_prior
        and not current_matter.use_fba_foreground_prior
        and current_prism_architecture
        == (
            f"{saved_prism_architecture}"
            "-flow-guided-latent-bridge-v1"
            "-pretrained-mam2-flow-bridge-v1"
        )
    )
    if (
        saved_prism_architecture != current_prism_architecture
        and not adding_foreground_refiner
        and not adding_foreground_detail_refiner
        and not adding_mam2_flow_bridge
    ):
        raise RuntimeError(
            "checkpoint PRISM architecture mismatch: "
            f"saved={saved_prism_architecture!r}, "
            f"current={current_prism_architecture!r}"
        )
    if adding_foreground_refiner:
        warnings.warn(
            "adding the foreground residual refiner to a compatible PRISM "
            "checkpoint; loading existing weights and initializing only the new head",
            UserWarning,
            stacklevel=2,
        )
    if adding_foreground_detail_refiner:
        warnings.warn(
            "adding zero-initialized coarse-to-fine foreground detail to a compatible "
            "PRISM checkpoint; preserving all donor weights",
            UserWarning,
            stacklevel=2,
        )
    if adding_mam2_flow_bridge:
        warnings.warn(
            "adding the MAM2 flow-guided foreground bridge to a compatible "
            "PRISM checkpoint; preserving existing weights and initializing "
            "only the new bridge and MAM2 projection",
            UserWarning,
            stacklevel=2,
        )
    for key in (
        "flow_parameterization",
        "max_refractive_flow_fraction",
        "max_refractive_flow",
        "refine_refractive_flow",
        "independent_physical_alpha",
        "flow_refinement_width",
        "flow_refinement_max_fraction",
        "refine_straight_foreground",
        "foreground_refinement_width",
        "foreground_refinement_logit_scale",
        "refine_straight_foreground_detail",
        "foreground_detail_width",
        "foreground_detail_max_delta",
        "flow_guided_latent_bridge",
        "use_fba_foreground_prior",
        "use_mam2_foreground_prior",
        "latent_bridge_width",
        "latent_bridge_flow_fraction",
        "latent_bridge_foreground_logit_scale",
        "latent_bridge_transmission_logit_scale",
        "refractive_kernel_size",
        "refractive_kernel_radius_fraction",
    ):
        current = getattr(current_matter, key)
        if (
            key in saved_matter
            and saved_matter[key] != current
            and not (key == "refine_straight_foreground" and adding_foreground_refiner)
            and not (
                key == "refine_straight_foreground_detail"
                and adding_foreground_detail_refiner
            )
            and not (
                key in {"flow_guided_latent_bridge", "use_mam2_foreground_prior"}
                and adding_mam2_flow_bridge
            )
        ):
            raise RuntimeError(
                f"checkpoint matter config mismatch for {key}: "
                f"saved={saved_matter[key]!r}, current={current!r}"
            )
    saved_predict_geometry = bool(saved_matter.get("predict_geometry", False))
    if saved_predict_geometry != current_matter.predict_geometry:
        raise RuntimeError(
            "checkpoint geometry-head mismatch: "
            f"saved={saved_predict_geometry!r}, "
            f"current={current_matter.predict_geometry!r}; start the geometry "
            "variant from Stage 1A so every stage shares one PAM head shape"
        )
    if saved_predict_geometry and "geometry_min_depth" in saved_matter:
        if saved_matter["geometry_min_depth"] != current_matter.geometry_min_depth:
            raise RuntimeError(
                "checkpoint geometry_min_depth mismatch: "
                f"saved={saved_matter['geometry_min_depth']!r}, "
                f"current={current_matter.geometry_min_depth!r}"
            )
    saved_mam2 = payload.get("mam2_integration_config", {})
    saved_backend = saved_mam2.get("matte", {}).get("backend")
    current_backend = predictor.mam2_integration_config.matte.backend
    allowed_eval_substitution = (
        saved_backend == "builtin" and current_backend == "external_mematte"
    )
    if (
        saved_backend is not None
        and saved_backend != current_backend
        and not allowed_eval_substitution
    ):
        raise RuntimeError(
            "checkpoint MAM2 matter backend mismatch: "
            f"saved={saved_backend!r}, current={current_backend!r}"
        )
    predictor.load_mam2_extension_state_dict(payload["predictor_mam2"], strict=strict)
    physics_state = payload["physics_pipeline"]
    saved_background = payload.get("pipeline_config", {}).get("background", {})
    # A format-v8 run may intentionally switch the Stage-3 completion ablation.
    # Preserve learned MAM2/PAM state while deliberately reinitializing only the
    # changed completion backbone.
    saved_completion = saved_background.get("completion_backbone", "dilated")
    current_completion = physics_pipeline.config.background.completion_backbone
    migrating_completion = saved_completion != current_completion
    completion_prefix = "background_model.completion."
    # Stage 2 learns optics against oracle B without a diffusion prior. When
    # Stage 3 adds a frozen released model, only its new optical adapter is
    # freshly initialized; every existing PAM/background tensor stays strict.
    adding_diffusion_adapter = (
        saved_background.get("completion_variant", "base") == "base"
        and physics_pipeline.config.background.completion_variant == "diffusion"
    )
    diffusion_adapter_prefix = "background_model.diffusion_completion.flow_latent_adapter."
    diffusion_pixel_prefix = "background_model.diffusion_completion.pixel_refiner."
    adding_flux_pixel_refiner = (
        bool(physics_pipeline.config.background.diffusion_pixel_refinement)
        and not bool(saved_background.get("diffusion_pixel_refinement", False))
    )

    def intentionally_new(name: str) -> bool:
        return (
            (adding_foreground_refiner and name.startswith("matter.foreground_refiner."))
            or (
                adding_foreground_detail_refiner
                and name.startswith("matter.foreground_detail_refiner.")
            )
            or (
                adding_mam2_flow_bridge
                and name.startswith(("matter.latent_bridge.", "matter.mam2_bridge_projection."))
            )
            or
            (migrating_completion and name.startswith(completion_prefix))
            or (
                adding_diffusion_adapter
                and name.startswith((diffusion_adapter_prefix, diffusion_pixel_prefix))
            )
            or (adding_flux_pixel_refiner and name.startswith(diffusion_pixel_prefix))
        )
    if migrating_completion:
        warnings.warn(
            "completion backbone changed from "
            f"{saved_completion!r} to {current_completion!r}; loading all compatible "
            "MAM2/PAM weights and initializing the new completion backbone",
            UserWarning,
            stacklevel=2,
        )
        physics_state = {
            name: value
            for name, value in physics_state.items()
            if not name.startswith(completion_prefix)
        }
    expected_physics = {
        name
        for name in physics_pipeline.state_dict()
        if not name.startswith("backbone.") and not intentionally_new(name)
    }
    received_physics = set(physics_state)
    if strict and expected_physics != received_physics:
        raise RuntimeError(
            "physics checkpoint mismatch; "
            f"missing={sorted(expected_physics-received_physics)}, "
            f"unexpected={sorted(received_physics-expected_physics)}"
        )
    missing, unexpected = physics_pipeline.load_state_dict(physics_state, strict=False)
    illegal_missing = [
        name
        for name in missing
        if not name.startswith("backbone.") and not intentionally_new(name)
    ]
    if strict and (illegal_missing or unexpected):
        raise RuntimeError(
            "physics checkpoint load failed; "
            f"missing={illegal_missing}, unexpected={list(unexpected)}"
        )
    return dict(payload.get("metadata", {}))


def load_stage1_mam2_checkpoint(
    path: str | Path,
    predictor: MAM2VideoPredictor,
) -> dict[str, Any]:
    """Transfer only trained Stage 1 semantics/matting into a new PAM design.

    Stage 1 checkpoints also serialize an untrained PAM.  A changed PAM output
    convention must never be loaded implicitly as if its weights had the same
    physical meaning.  This explicit Stage 2 handoff leaves PAM and background
    freshly initialized while strictly loading the trained MAM2 extension.
    """

    source = Path(path)
    if source.name != "prism_stage1b_best.pt":
        raise RuntimeError("Stage 1 transfer requires the final Stage 1B best checkpoint")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 8:
        raise RuntimeError("Stage 1 transfer requires a format-v8 checkpoint")
    architecture = _mam2_architecture(predictor)
    if payload.get("mam2_architecture") != architecture:
        raise RuntimeError(
            "Stage 1 MAM2 architecture mismatch: "
            f"saved={payload.get('mam2_architecture')!r}, current={architecture!r}"
        )
    saved_config = payload.get("mam2_integration_config", {})
    saved_backend = saved_config.get("matte", {}).get("backend")
    current_backend = predictor.mam2_integration_config.matte.backend
    if saved_backend != current_backend:
        raise RuntimeError(
            "Stage 1 matter backend mismatch: "
            f"saved={saved_backend!r}, current={current_backend!r}"
        )
    predictor.load_mam2_extension_state_dict(payload["predictor_mam2"], strict=True)
    return dict(payload.get("metadata", {}))


def load_refractive_training_state(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 8:
        raise RuntimeError("training resume requires a format-v8 full-PRISM checkpoint")
    return dict(payload.get("training_state", {}))
