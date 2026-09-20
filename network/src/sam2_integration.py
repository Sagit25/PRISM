from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .config import SAM2IntegrationConfig
from .lora import inject_lora
from .mam2_matte import build_mam2_matter
from .mss import MemorySeparableSiamese
from .types import MAM2BackboneOutput
from .vendor import activate_vendored_sam2, sam2_setup_hint


activate_vendored_sam2()

try:
    from sam2.sam2_video_predictor import SAM2VideoPredictor as _OfficialPredictor

    _SAM2_AVAILABLE = True
except ImportError:
    _OfficialPredictor = nn.Module  # type: ignore[assignment,misc]
    _SAM2_AVAILABLE = False


@dataclass
class MAM2FrameOutput:
    frame_index: int
    mask_logits: Tensor
    trimap_logits: Tensor
    non_memory_features: Tensor


def _clone_for_cache(value: Tensor, to_cpu: bool) -> Tensor:
    value = value.detach()
    return value.cpu() if to_cpu else value


class MAM2VideoPredictor(_OfficialPredictor):  # type: ignore[misc,valid-type]
    """Real official-SAM2 subclass with prompt-conditioned PDD and MSS."""

    def __init__(self, *args, **kwargs) -> None:
        if not _SAM2_AVAILABLE:
            raise ImportError(sam2_setup_hint())
        super().__init__(*args, **kwargs)
        self.configure_mam2(SAM2IntegrationConfig(), inject_image_lora=False)

    def configure_mam2(
        self,
        config: SAM2IntegrationConfig,
        *,
        inject_image_lora: bool,
    ) -> None:
        if self.hidden_dim != config.pdd.feature_channels:
            raise ValueError(
                f"PDD feature_channels={config.pdd.feature_channels} must equal "
                f"SAM2 hidden_dim={self.hidden_dim}"
            )
        self.mam2_integration_config = config
        self.mam2_mss = MemorySeparableSiamese(config=config.pdd)
        self.mam2_matter = build_mam2_matter(config.matte)
        self._mam2_output_cache: dict[int, list[MAM2FrameOutput]] = defaultdict(list)
        self.mam2_lora_modules: list[str] = []
        if inject_image_lora and config.lora_rank > 0:
            self.mam2_lora_modules = inject_lora(
                self.image_encoder,
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                target_patterns=config.lora_target_patterns,
            )

    def _encode_mam2_prompts(
        self,
        batch_size: int,
        device: torch.device,
        *,
        point_inputs: dict[str, Tensor] | None,
        mask_inputs: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if point_inputs is None:
            point_coords = torch.zeros(batch_size, 1, 2, device=device)
            point_labels = -torch.ones(
                batch_size, 1, dtype=torch.int32, device=device
            )
        else:
            point_coords = point_inputs["point_coords"]
            point_labels = point_inputs["point_labels"]
        if mask_inputs is not None:
            mask_prompt = mask_inputs.float()
            if mask_prompt.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                mask_prompt = F.interpolate(
                    mask_prompt,
                    size=self.sam_prompt_encoder.mask_input_size,
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
        else:
            mask_prompt = None
        return self.sam_prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=mask_prompt,
        )

    def _track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse,
        prev_sam_mask_logits,
    ):
        current_out, sam_outputs, high_res_features, memory_features = super()._track_step(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
        )
        non_memory_features = current_vision_feats[-1].permute(1, 2, 0).reshape(
            current_vision_feats[-1].shape[1],
            self.hidden_dim,
            *feat_sizes[-1],
        )
        values = list(sam_outputs)
        if len(values) < 5:
            raise RuntimeError("unsupported official SAM2 sam_outputs tuple")
        low_res_masks = values[3]
        high_res_masks = values[4]
        batch_size = low_res_masks.shape[0]

        mask_sparse, mask_dense = self._encode_mam2_prompts(
            batch_size,
            low_res_masks.device,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
        )
        # The encoder callback runs after pass 1, so pass 2 is conditioned on
        # the *refined* mask rather than the original SAM2 seed prediction.
        def encode_refined_mask(mask_logits: Tensor) -> tuple[Tensor, Tensor]:
            return self._encode_mam2_prompts(
                batch_size,
                low_res_masks.device,
                point_inputs=None,
                mask_inputs=mask_logits.sigmoid(),
            )

        mss = self.mam2_mss(
            memory_features=memory_features,
            non_memory_features=non_memory_features,
            seed_mask_logits=low_res_masks,
            mask_sparse_prompt_embeddings=mask_sparse,
            mask_dense_prompt_embeddings=mask_dense,
            trimap_prompt_encoder=encode_refined_mask,
            high_res_features=high_res_features,
        )

        refined_low_res = mss.mask_logits
        mask_delta = refined_low_res - low_res_masks
        refined_high_res = high_res_masks + F.interpolate(
            mask_delta,
            size=high_res_masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        if self.mam2_integration_config.replace_sam_mask_for_memory:
            values[3] = refined_low_res
            values[4] = refined_high_res
            sam_outputs = tuple(values)

        current_out["mam2_mask_logits"] = refined_low_res
        current_out["mam2_trimap_logits"] = mss.trimap_logits
        current_out["mam2_non_memory_features"] = non_memory_features
        if not self.training:
            to_cpu = self.mam2_integration_config.cache_inference_features_on_cpu
            self._mam2_output_cache[int(frame_idx)].append(
                MAM2FrameOutput(
                    frame_index=int(frame_idx),
                    mask_logits=_clone_for_cache(refined_low_res, to_cpu),
                    trimap_logits=_clone_for_cache(mss.trimap_logits, to_cpu),
                    non_memory_features=_clone_for_cache(non_memory_features, to_cpu),
                )
            )
        return current_out, sam_outputs, high_res_features, memory_features

    def forward_mam2_clip(
        self,
        normalized_frames: Tensor,
        *,
        first_frame_point_inputs: dict[str, Tensor] | None = None,
        first_frame_mask_inputs: Tensor | None = None,
        detach_memory_every: int | None = None,
        compute_alpha: bool = True,
    ) -> MAM2BackboneOutput:
        """Differentiable official-SAM2 clip forward with bounded-BPTT option."""

        if normalized_frames.ndim != 5 or normalized_frames.shape[2] != 3:
            raise ValueError("normalized_frames must have shape [B,T,3,H,W]")
        if normalized_frames.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"training frames must be resized to {self.image_size}x{self.image_size}"
            )
        if (first_frame_point_inputs is None) == (first_frame_mask_inputs is None):
            raise ValueError("pass exactly one first-frame point or mask prompt")

        batch, frames = normalized_frames.shape[:2]
        time_major = normalized_frames.transpose(0, 1).flatten(0, 1)
        use_temporal_checkpointing = (
            self.mam2_integration_config.temporal_activation_checkpointing
            and self.training
            and torch.is_grad_enabled()
        )
        checkpoint_chunk_size = (
            self.mam2_integration_config.temporal_checkpoint_chunk_size
            if use_temporal_checkpointing
            else time_major.shape[0]
        )
        if checkpoint_chunk_size < 1:
            raise ValueError("temporal_checkpoint_chunk_size must be positive")
        feature_chunks: list[list[Tensor]] = []
        position_chunks: list[list[Tensor]] = []
        feat_sizes = None
        for start in range(0, time_major.shape[0], checkpoint_chunk_size):
            frame_chunk = time_major[start : start + checkpoint_chunk_size]
            if use_temporal_checkpointing:
                backbone_out = checkpoint(
                    self.forward_image,
                    frame_chunk,
                    use_reentrant=False,
                )
            else:
                backbone_out = self.forward_image(frame_chunk)
            _, chunk_features, chunk_positions, chunk_sizes = (
                self._prepare_backbone_features(backbone_out)
            )
            if feat_sizes is None:
                feat_sizes = chunk_sizes
            elif feat_sizes != chunk_sizes:
                raise RuntimeError("SAM2 feature sizes changed across frame chunks")
            feature_chunks.append(chunk_features)
            position_chunks.append(chunk_positions)
        assert feat_sizes is not None
        vision_feats = [
            torch.cat([chunk[level] for chunk in feature_chunks], dim=1)
            for level in range(len(feature_chunks[0]))
        ]
        vision_pos = [
            torch.cat([chunk[level] for chunk in position_chunks], dim=1)
            for level in range(len(position_chunks[0]))
        ]
        if detach_memory_every is None and self.training:
            detach_memory_every = self.mam2_integration_config.temporal_detach_interval
        output_dict: dict[str, dict[int, dict[str, Tensor]]] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        mask_outputs: list[Tensor] = []
        trimap_outputs: list[Tensor] = []
        clean_outputs: list[Tensor] = []
        for frame_index in range(frames):
            start, end = frame_index * batch, (frame_index + 1) * batch
            current_out = self.track_step(
                frame_idx=frame_index,
                is_init_cond_frame=(frame_index == 0),
                current_vision_feats=[x[:, start:end] for x in vision_feats],
                current_vision_pos_embeds=[x[:, start:end] for x in vision_pos],
                feat_sizes=feat_sizes,
                point_inputs=first_frame_point_inputs if frame_index == 0 else None,
                mask_inputs=first_frame_mask_inputs if frame_index == 0 else None,
                output_dict=output_dict,
                num_frames=frames,
                track_in_reverse=False,
                run_mem_encoder=True,
                prev_sam_mask_logits=None,
            )
            storage = "cond_frame_outputs" if frame_index == 0 else "non_cond_frame_outputs"
            if detach_memory_every and frame_index and frame_index % detach_memory_every == 0:
                for key in ("maskmem_features", "obj_ptr"):
                    if current_out.get(key) is not None:
                        current_out[key] = current_out[key].detach()
            output_dict[storage][frame_index] = current_out
            mask_outputs.append(current_out["mam2_mask_logits"])
            trimap_outputs.append(current_out["mam2_trimap_logits"])
            clean_outputs.append(current_out["mam2_non_memory_features"])

        stacked_trimap = torch.stack(trimap_outputs, dim=1)
        alpha_matte = (
            self.mam2_matter(
                self.denormalize_sam2_frames(normalized_frames),
                stacked_trimap,
            )
            if compute_alpha
            else stacked_trimap.new_zeros((batch, frames, 1, 1, 1))
        )
        output = MAM2BackboneOutput(
            mask_logits=torch.stack(mask_outputs, dim=1),
            trimap_logits=stacked_trimap,
            alpha_matte=alpha_matte,
            non_memory_features=torch.stack(clean_outputs, dim=1),
        )
        output.validate()
        return output

    def clear_mam2_cache(self) -> None:
        self._mam2_output_cache.clear()

    def pop_mam2_frame_outputs(self, frame_index: int) -> list[MAM2FrameOutput]:
        return self._mam2_output_cache.pop(int(frame_index), [])

    def mam2_extension_state_dict(self) -> dict[str, Tensor]:
        backend = self.mam2_integration_config.matte.backend
        return {
            name: value
            for name, value in self.state_dict().items()
            if name.startswith("mam2_mss.")
            or (backend == "builtin" and name.startswith("mam2_matter."))
            or (
                backend == "external_mematte"
                and name.startswith("mam2_matter.external_model.decoder.")
            )
            or name.endswith("lora_A")
            or name.endswith("lora_B")
        }

    def load_mam2_extension_state_dict(
        self, state_dict: dict[str, Tensor], *, strict: bool = True
    ) -> None:
        expected_state = self.mam2_extension_state_dict()
        expected = set(expected_state)
        # A builtin-matter checkpoint can initialize an external-MEMatte run;
        # incompatible matter tensors are deliberately ignored.  Once Stage
        # 1B saves decoder deltas, those matching tensors load normally.
        state_dict = {
            name: value for name, value in state_dict.items()
            if name in expected or not name.startswith("mam2_matter.")
        }
        received = set(state_dict)
        missing = expected - received
        if self.mam2_integration_config.matte.backend == "external_mematte":
            decoder_keys = {
                name for name in expected
                if name.startswith("mam2_matter.external_model.decoder.")
            }
            if not (received & decoder_keys):
                missing -= decoder_keys
        if strict and (missing or received - expected):
            raise RuntimeError(
                f"MAM2 extension checkpoint mismatch; missing={sorted(missing)}, "
                f"unexpected={sorted(received-expected)}"
            )
        self.load_state_dict(state_dict, strict=False)

    @staticmethod
    def denormalize_sam2_frames(normalized_frames: Tensor) -> Tensor:
        """Recover RGB [0,1] from the ImageNet normalization used by SAM2."""

        mean = normalized_frames.new_tensor((0.485, 0.456, 0.406)).view(
            1, 1, 3, 1, 1
        )
        std = normalized_frames.new_tensor((0.229, 0.224, 0.225)).view(
            1, 1, 3, 1, 1
        )
        return (normalized_frames * std + mean).clamp(0.0, 1.0)

    def predict_mam2_alpha(self, frames: Tensor, trimap_logits: Tensor) -> Tensor:
        """Run the replaceable RGB+trimap matter used by full MAM2."""

        return self.mam2_matter(frames, trimap_logits)


def _load_official_checkpoint(model: MAM2VideoPredictor, path: str | Path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    illegal_missing = [
        key
        for key in missing
        if not key.startswith(("mam2_mss.", "mam2_matter."))
    ]
    if illegal_missing or unexpected:
        raise RuntimeError(
            "official SAM2 checkpoint mismatch; "
            f"missing={illegal_missing}, unexpected={list(unexpected)}"
        )


def build_mam2_video_predictor(
    config_file: str,
    sam2_checkpoint: str | Path,
    *,
    device: str | torch.device = "cuda",
    mode: str = "eval",
    integration_config: SAM2IntegrationConfig | None = None,
    mam2_checkpoint: str | Path | None = None,
    hydra_overrides_extra: list[str] | None = None,
    apply_postprocessing: bool = True,
) -> MAM2VideoPredictor:
    """Instantiate a true subclass and strictly account for checkpoint keys."""

    if not _SAM2_AVAILABLE:
        raise ImportError(sam2_setup_hint())
    sam2_checkpoint = Path(sam2_checkpoint).expanduser()
    if not sam2_checkpoint.is_file():
        raise FileNotFoundError(
            f"official SAM2 checkpoint not found: {sam2_checkpoint}; {sam2_setup_hint()}"
        )
    if mode not in {"eval", "train"}:
        raise ValueError("mode must be 'eval' or 'train'")
    from hydra import compose
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    import sam2  # noqa: F401 - registers the official Hydra config module

    signature = set(inspect.signature(_OfficialPredictor._track_step).parameters)
    required = {"current_vision_feats", "feat_sizes", "output_dict", "prev_sam_mask_logits"}
    if not required.issubset(signature):
        raise RuntimeError("installed SAM2 _track_step API is incompatible")

    overrides = [
        "++model._target_=refractive_mam2.sam2_integration.MAM2VideoPredictor"
    ]
    if apply_postprocessing:
        overrides += [
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            "++model.fill_hole_area=8",
        ]
    overrides += hydra_overrides_extra or []
    cfg = compose(config_name=config_file, overrides=overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    if not isinstance(model, MAM2VideoPredictor):
        raise TypeError("Hydra did not instantiate MAM2VideoPredictor")
    model.configure_mam2(integration_config or SAM2IntegrationConfig(), inject_image_lora=False)
    _load_official_checkpoint(model, sam2_checkpoint)
    model.to(device)
    if model.mam2_integration_config.lora_rank > 0:
        model.mam2_lora_modules = inject_lora(
            model.image_encoder,
            rank=model.mam2_integration_config.lora_rank,
            alpha=model.mam2_integration_config.lora_alpha,
            dropout=model.mam2_integration_config.lora_dropout,
            target_patterns=model.mam2_integration_config.lora_target_patterns,
        )
    if mam2_checkpoint is not None:
        payload: Any = torch.load(mam2_checkpoint, map_location="cpu", weights_only=True)
        state = payload.get("predictor_mam2", payload) if isinstance(payload, dict) else payload
        if not isinstance(state, dict):
            raise TypeError("MAM2 checkpoint must contain a tensor state dictionary")
        model.load_mam2_extension_state_dict(state)
    model.train(mode == "train")
    return model


def _set_matter_trainable(predictor: MAM2VideoPredictor, enabled: bool) -> None:
    matter = predictor.mam2_matter
    configure = getattr(matter, "configure_trainable", None)
    if callable(configure):
        configure(enabled)
    else:
        matter.requires_grad_(enabled)


def _set_lora_trainable(predictor: MAM2VideoPredictor, enabled: bool) -> None:
    for module in predictor.modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            module.lora_A.requires_grad_(enabled)
            module.lora_B.requires_grad_(enabled)


def mark_only_mam2_semantics_trainable(
    predictor: MAM2VideoPredictor,
) -> list[nn.Parameter]:
    """Stage 1A: train PDD/MSS and LoRA, but not the alpha matter."""

    predictor.requires_grad_(False)
    predictor.mam2_mss.requires_grad_(True)
    _set_lora_trainable(predictor, True)
    _set_matter_trainable(predictor, False)
    return [parameter for parameter in predictor.parameters() if parameter.requires_grad]


def mark_only_mam2_matter_trainable(
    predictor: MAM2VideoPredictor,
) -> list[nn.Parameter]:
    """Stage 1B: freeze semantics and train the matter/decoder only."""

    predictor.requires_grad_(False)
    _set_matter_trainable(predictor, True)
    return [parameter for parameter in predictor.parameters() if parameter.requires_grad]


def mark_only_mam2_trainable(predictor: MAM2VideoPredictor) -> list[nn.Parameter]:
    """Stage 4: jointly tune MAM2 adapters while SAM2/MEMatte encoders stay frozen."""

    predictor.requires_grad_(False)
    predictor.mam2_mss.requires_grad_(True)
    _set_lora_trainable(predictor, True)
    _set_matter_trainable(predictor, True)
    return [parameter for parameter in predictor.parameters() if parameter.requires_grad]
