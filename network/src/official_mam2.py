"""Released MAM2 video predictor and matter, adapted to PRISM's clip contract.

The official ICLR 2026 PDD/MSS main model is used for prompt propagation, not
the older in-tree approximation.  Its heavy semantic main is frozen, while the
released alpha matter's decoder and a flow-conditioned feature adapter can be
optimized together with PRISM-PAM and the background branch.
"""

from __future__ import annotations

import importlib
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .color import srgb_to_linear
from .config import MAM2MatteConfig, SAM2IntegrationConfig
from .mam2_matte import ExternalMEMatteMatter
from .types import MAM2BackboneOutput


_SAM2_MEAN = (0.485, 0.456, 0.406)
_SAM2_STD = (0.229, 0.224, 0.225)


class OfficialMAM2Matter(ExternalMEMatteMatter):
    """Flow-guided feature path inside the released MAM2 alpha matter."""

    def __init__(self, model: nn.Module, *, train_decoder: bool = True) -> None:
        super().__init__(
            model,
            patch_decoder=False,
            train_decoder=train_decoder,
            train_backbone=False,
            frame_chunk_size=1,
            full_activation_checkpointing=True,
            flow_feature_guidance=True,
        )

    def _run_external(
        self,
        flat_frames: Tensor,
        scalar_trimap: Tensor,
        flow_context: Tensor | None = None,
    ) -> dict[str, Tensor]:
        model = self.external_model
        if not all(hasattr(model, name) for name in ("pixel_mean", "pixel_std", "size_divisibility")):
            raise TypeError("expected the released mam2.models.matting.LightweightMatter")
        # PRISM stores linear RGB; the released MAM2 matter was trained on sRGB.
        srgb = torch.where(
            flat_frames.clamp(0, 1) <= 0.0031308,
            flat_frames.clamp(0, 1) * 12.92,
            1.055 * flat_frames.clamp(0, 1).pow(1 / 2.4) - 0.055,
        )
        height, width = srgb.shape[-2:]
        inputs = torch.cat(
            ((srgb - model.pixel_mean) / model.pixel_std, scalar_trimap), dim=1
        )
        multiple = model.size_divisibility
        inputs = F.pad(inputs, (0, (-width) % multiple, 0, (-height) % multiple))
        result = model.backbone(inputs)
        features = result[0] if isinstance(result, (tuple, list)) else result
        if flow_context is not None:
            assert self.flow_feature_modulator is not None
            modulation = self.flow_feature_modulator(flow_context.float())
            modulation = F.interpolate(
                modulation, size=features.shape[-2:], mode="bilinear", align_corners=False
            ).to(features.dtype)
            scale, shift = modulation.tanh().chunk(2, dim=1)
            features = features * (1 + 0.25 * scale) + 0.25 * shift
        decoded = model.decoder(features, inputs)
        if not isinstance(decoded, dict) or "phas" not in decoded:
            raise RuntimeError("released MAM2 matter decoder did not return alpha")
        return {"phas": decoded["phas"][..., :height, :width]}


class OfficialMAM2ClipBackbone(nn.Module):
    """Run official MAM2 on a short clip through its public prompt API.

    The 1024px PDD/MSS prior is frozen.  Input video bytes are staged as
    lossless PNGs because the released predictor accepts a frame directory;
    this preserves its preprocessing and memory-state behavior exactly.
    """

    official_mam2 = True
    mam2_architecture = "official-mam2-iclr2026-flow-v1"

    def __init__(self, main: nn.Module, matter: nn.Module) -> None:
        super().__init__()
        if not hasattr(main, "propagate_in_video") or not hasattr(main, "init_state"):
            raise TypeError("expected released MAM2 video predictor")
        self.main = main.requires_grad_(False).eval()
        self.mam2_matter = OfficialMAM2Matter(matter)
        self.image_size = int(main.image_size)
        self.mam2_integration_config = SAM2IntegrationConfig(
            matte=MAM2MatteConfig(backend="official_mam2")
        )

    @classmethod
    def from_pretrained(
        cls,
        source_root: str | Path,
        *,
        config_path: str | Path | None = None,
        device: str | torch.device = "cuda",
    ) -> "OfficialMAM2ClipBackbone":
        root = Path(source_root).resolve()
        if not (root / "mam2" / "inference.py").is_file():
            raise FileNotFoundError("source_root must be the released Matting-Anything-2 checkout")
        # PRISM's default SAM2 checkout is already imported by the training
        # package. The released MAM2 main requires its own patched SAM2 fork;
        # silently mixing the two would make strict checkpoint loading fail or
        # run the wrong PDD/MSS implementation.
        import sam2

        if not Path(sam2.__file__).resolve().is_relative_to(root / "sam2"):
            raise RuntimeError(
                "launch PRISM with PRISM_SAM2_ROOT=<official-mam2-root> "
                "before importing refractive_mam2"
            )
        sys.path.insert(0, str(root))
        try:
            config_module = importlib.import_module("mam2.config")
            build_module = importlib.import_module("mam2.models.build")
        except ImportError as exc:
            raise ImportError("install the released MAM2 package and dependencies") from exc
        finally:
            sys.path.remove(str(root))
        if not Path(config_module.__file__).resolve().is_relative_to(root):
            raise RuntimeError("imported MAM2 is not from the requested source_root")
        from omegaconf import OmegaConf

        config = config_module.load_config(
            Path(config_path) if config_path is not None else root / "configs/inference.yaml"
        )
        OmegaConf.resolve(config)
        target = torch.device(device)
        main = build_module.build_predictor(
            config.model, config.checkpoints.main, config.lora, target
        )
        matter = build_module.build_matter(
            config.matter, config.checkpoints.matter, target
        )
        return cls(main, matter).to(target)

    def train(self, mode: bool = True) -> "OfficialMAM2ClipBackbone":
        super().train(mode)
        self.main.eval()
        return self

    def configure_joint_trainable(self) -> list[nn.Parameter]:
        self.requires_grad_(False)
        self.mam2_matter.configure_trainable(True)
        self.main.eval()
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def clear_mam2_cache(self) -> None:
        # The official predictor state is disposed after every clip.
        return None

    def mam2_extension_state_dict(self) -> dict[str, Tensor]:
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if name.startswith((
                "mam2_matter.external_model.decoder.",
                "mam2_matter.flow_feature_modulator.",
            ))
        }

    def load_mam2_extension_state_dict(
        self, state: dict[str, Tensor], *, strict: bool = True
    ) -> None:
        expected = set(self.mam2_extension_state_dict())
        if strict and set(state) != expected:
            raise RuntimeError(
                f"official MAM2 adapter checkpoint mismatch: missing={sorted(expected - set(state))}, "
                f"unexpected={sorted(set(state) - expected)}"
            )
        self.load_state_dict(state, strict=False)

    @staticmethod
    def _write_frames(directory: Path, frames: Tensor) -> None:
        import numpy as np
        from PIL import Image

        for index, value in enumerate(frames):
            array = (
                value.detach().clamp(0, 1).permute(1, 2, 0).mul(255).round()
                .to(torch.uint8).cpu().numpy()
            )
            Image.fromarray(np.asarray(array), mode="RGB").save(
                directory / f"{index:05d}.png"
            )

    def _one_clip(
        self,
        normalized: Tensor,
        *,
        output_size: tuple[int, int],
        mask: Tensor | None,
        point_coords: Tensor | None,
        point_labels: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        mean = normalized.new_tensor(_SAM2_MEAN).view(1, 3, 1, 1)
        std = normalized.new_tensor(_SAM2_STD).view(1, 3, 1, 1)
        srgb = (normalized * std + mean).clamp(0, 1)
        srgb = F.interpolate(srgb, size=output_size, mode="bilinear", align_corners=False)
        with tempfile.TemporaryDirectory(prefix="prism-official-mam2-") as temporary:
            self._write_frames(Path(temporary), srgb)
            with torch.inference_mode():
                state = self.main.init_state(
                    temporary,
                    offload_video_to_cpu=True,
                    offload_state_to_cpu=True,
                    async_loading_frames=False,
                )
                try:
                    if mask is not None:
                        self.main.add_new_mask(state, 0, 1, mask[0] >= 0.5)
                    else:
                        if point_coords is None or point_labels is None:
                            raise ValueError("official MAM2 needs a first-frame prompt")
                        # The released API scales coordinates by image_size
                        # even with normalize_coords=False; pass [0,1] here.
                        self.main.add_new_points_or_box(
                            state, 0, 1,
                            points=point_coords / self.image_size,
                            labels=point_labels,
                            normalize_coords=False,
                        )
                    masks: dict[int, Tensor] = {}
                    trimaps: dict[int, Tensor] = {}
                    for frame_index, obj_ids, mask_logits, trimap_probability in self.main.propagate_in_video(
                        state, start_frame_idx=0, show_progress=False
                    ):
                        if 1 not in list(obj_ids):
                            raise RuntimeError("official MAM2 lost prompted object 1")
                        position = list(obj_ids).index(1)
                        masks[int(frame_index)] = mask_logits[position:position + 1].detach()
                        trimaps[int(frame_index)] = trimap_probability[position:position + 1].detach()
                    if set(masks) != set(range(normalized.shape[0])):
                        raise RuntimeError("official MAM2 did not cover every clip frame")
                    features: list[Tensor] = []
                    for index in range(normalized.shape[0]):
                        _, backbone, *_ = self.main._get_image_feature(state, index, 1)
                        features.append(backbone["backbone_fpn"][-1].detach())
                    mask_video = torch.stack([masks[i][0] for i in range(len(masks))])
                    trimap_video = torch.stack([trimaps[i][0] for i in range(len(trimaps))])
                    feature_video = torch.stack([value[0] for value in features])
                finally:
                    state["images"].close()
                    self.main.reset_state(state)
        return mask_video.clone(), trimap_video.clone(), feature_video.clone()

    def forward_mam2_clip(
        self,
        normalized_frames: Tensor,
        *,
        first_frame_mask_inputs: Tensor | None = None,
        first_frame_point_inputs: dict[str, Tensor] | None = None,
        compute_alpha: bool = True,
        original_size: tuple[int, int] | None = None,
    ) -> MAM2BackboneOutput:
        if normalized_frames.ndim != 5 or normalized_frames.shape[2] != 3:
            raise ValueError("normalized frames must be [B,T,3,H,W]")
        if not compute_alpha:
            raise ValueError("official MAM2 is a pretrained Stage-2/3/4 backend, not Stage-1A")
        if (first_frame_mask_inputs is None) == (first_frame_point_inputs is None):
            raise ValueError("provide exactly one MAM2 prompt type")
        if original_size is None:
            raise ValueError("official MAM2 requires the original PRISM frame size")
        batch, time = normalized_frames.shape[:2]
        masks: list[Tensor] = []
        trimaps: list[Tensor] = []
        features: list[Tensor] = []
        for index in range(batch):
            point = first_frame_point_inputs
            mask, trimap, feature = self._one_clip(
                normalized_frames[index],
                output_size=original_size,
                mask=None if first_frame_mask_inputs is None else first_frame_mask_inputs[index],
                point_coords=None if point is None else point["point_coords"][index],
                point_labels=None if point is None else point["point_labels"][index],
            )
            masks.append(mask)
            trimaps.append(trimap)
            features.append(feature)
        mask_logits = torch.stack(masks)
        trimap_prob = torch.stack(trimaps).clamp_min(1e-6)
        trimap_logits = trimap_prob.log()
        non_memory = torch.stack(features)
        if mask_logits.shape[:3] != (batch, time, 1):
            raise RuntimeError("official MAM2 mask shape mismatch")
        # MAM2 was prompted at the native PRISM size. Recover sRGB there for
        # its released matter, then convert to PRISM's linear-RGB contract.
        mean = normalized_frames.new_tensor(_SAM2_MEAN).view(1, 1, 3, 1, 1)
        std = normalized_frames.new_tensor(_SAM2_STD).view(1, 1, 3, 1, 1)
        srgb = (normalized_frames * std + mean).clamp(0, 1)
        srgb = F.interpolate(
            srgb.reshape(batch * time, 3, *srgb.shape[-2:]),
            size=original_size, mode="bilinear", align_corners=False,
        ).reshape(batch, time, 3, *original_size)
        alpha = self.mam2_matter(srgb_to_linear(srgb), trimap_logits)
        result = MAM2BackboneOutput(mask_logits, trimap_logits, alpha, non_memory)
        result.validate()
        return result

    def forward(self, frames: Tensor, prompts: Any | None = None) -> MAM2BackboneOutput:
        raise RuntimeError("use forward_mam2_clip with explicit first-frame prompts")
