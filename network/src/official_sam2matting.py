"""Released SAM2Matting with a flow-conditioned, trainable alpha branch.

The ECCV 2026 tracker is a frozen pretrained prior.  Its released progressive
alpha heads are reused (and fine-tuned) after PRISM predicts a provisional
refractive flow.  No MEMatte module is loaded by this backend.
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
from torch.utils.checkpoint import checkpoint

from .config import MAM2MatteConfig, SAM2IntegrationConfig
from .renderer import warp_background
from .types import MAM2BackboneOutput


_SAM2_MEAN = (0.485, 0.456, 0.406)
_SAM2_STD = (0.229, 0.224, 0.225)
_ALPHA_MODULES = (
    "unknown_region_predictor",
    "unknown_fusion",
    "unknown_alpha_predictor",
    "alpha_pred1",
    "alpha_pred2",
    "alpha_pred3",
)


class _FlowGuidedSAM2Matte(nn.Module):
    """Inject aligned B, render error, and u into the released 1/16 feature.

    The zero-initialized FiLM initially preserves the official prediction.
    Gradients through the released alpha heads then reach PRISM's flow and
    background estimates without updating the frozen video tracker.
    """

    flow_feature_guidance = True

    def __init__(self, official_model: nn.Module) -> None:
        super().__init__()
        object.__setattr__(self, "_official_model", official_model)
        self.flow_feature_modulator = nn.Sequential(
            nn.Conv2d(9, 48, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(48, 512, 3, padding=1),
        )
        nn.init.zeros_(self.flow_feature_modulator[-1].weight)
        nn.init.zeros_(self.flow_feature_modulator[-1].bias)
        self._cached_inputs: list[tuple[Tensor, Tensor, list[Tensor]]] = []

    @property
    def official_model(self) -> nn.Module:
        return object.__getattribute__(self, "_official_model")

    def set_cached_inputs(
        self, inputs: list[tuple[Tensor, Tensor, list[Tensor]]]
    ) -> None:
        self._cached_inputs = inputs

    def clear_cached_inputs(self) -> None:
        self._cached_inputs = []

    @staticmethod
    def trimap_from_alpha(alpha: Tensor, mask_logits: Tensor) -> Tensor:
        """Expose a differentiable PRISM trimap proxy, not an official output."""

        support = mask_logits.sigmoid()
        if support.shape[-2:] != alpha.shape[-2:]:
            batch, time = support.shape[:2]
            support = F.interpolate(
                support.reshape(batch * time, 1, *support.shape[-2:]),
                size=alpha.shape[-2:], mode="bilinear", align_corners=False,
            ).reshape(batch, time, 1, *alpha.shape[-2:])
        unknown_fraction = (4 * alpha * (1 - alpha)).clamp(0, 1)
        unknown = support * unknown_fraction
        foreground = support * (1 - unknown_fraction) * alpha
        background = (1 - support) + support * (1 - unknown_fraction) * (1 - alpha)
        return torch.cat((background, unknown, foreground), dim=2).clamp_min(1e-6).log()

    def forward_with_flow(
        self,
        frames: Tensor,
        trimap_logits: Tensor,
        background: Tensor,
        refractive_flow: Tensor,
        render_error: Tensor,
        object_support: Tensor,
    ) -> Tensor:
        del trimap_logits  # SAM2Matting uses its own learned ROI/pseudo-trimap.
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError("frames must be [B,T,3,H,W]")
        batch, time, _, height, width = frames.shape
        if background.shape != (batch, 3, height, width):
            raise ValueError("background must be [B,3,H,W]")
        if refractive_flow.shape != (batch, time, 2, height, width):
            raise ValueError("flow must be [B,T,2,H,W]")
        if render_error.shape != frames.shape:
            raise ValueError("render error must match frames")
        if object_support.shape != (batch, time, 1, height, width):
            raise ValueError("object support must be [B,T,1,H,W]")
        if len(self._cached_inputs) != batch * time:
            raise RuntimeError("SAM2Matting must run on this clip before flow refinement")

        aligned = warp_background(
            background[:, None].expand(-1, time, -1, -1, -1),
            refractive_flow,
        )
        scale = refractive_flow.new_tensor(
            (max(width - 1, 1), max(height - 1, 1))
        ).view(1, 1, 2, 1, 1)
        context = torch.cat(
            (aligned, render_error, refractive_flow / scale, object_support), dim=2
        ).reshape(batch * time, 9, height, width)

        outputs: list[Tensor] = []
        for index, (image, mask, features) in enumerate(self._cached_inputs):
            if len(features) < 3 or features[-1].shape[1] != 256:
                raise RuntimeError("released SAM2Matting feature pyramid is incompatible")

            def run(flow_context: Tensor) -> Tensor:
                delta = self.flow_feature_modulator(flow_context.float())
                delta = F.interpolate(
                    delta, size=features[-1].shape[-2:], mode="bilinear",
                    align_corners=False,
                ).to(features[-1].dtype)
                gain, bias = delta.tanh().chunk(2, dim=1)
                guided = list(features)
                guided[-1] = guided[-1] * (1 + 0.25 * gain) + 0.25 * bias
                alpha, _, _ = self.official_model._forward_alpha_heads(
                    mask_inputs=mask, high_res_features=guided, image=image,
                )
                return F.interpolate(
                    alpha.float(), size=(height, width), mode="bilinear",
                    align_corners=False,
                )

            value = context[index:index + 1]
            if self.training and torch.is_grad_enabled():
                alpha = checkpoint(run, value, use_reentrant=False)
            else:
                alpha = run(value)
            outputs.append(alpha[0].clamp(0, 1))
        return torch.stack(outputs).reshape(batch, time, 1, height, width)


class OfficialSAM2MattingClipBackbone(nn.Module):
    """Adapt the released SAM2Matting video predictor to PRISM's clip API."""

    official_mam2 = True  # Reuse PRISM's native-size clip forwarding contract.
    mam2_architecture = "official-sam2matting-eccv2026-flow-v1"

    def __init__(self, main: nn.Module) -> None:
        super().__init__()
        if not all(hasattr(main, key) for key in (
            "propagate_in_video", "init_state", "_forward_alpha_heads",
        )):
            raise TypeError("expected the released SAM2Matting video predictor")
        self.main = main.requires_grad_(False).eval()
        self.mam2_matter = _FlowGuidedSAM2Matte(main)
        self.image_size = int(main.image_size)
        self.mam2_integration_config = SAM2IntegrationConfig(
            matte=MAM2MatteConfig(backend="official_sam2matting")
        )

    @classmethod
    def from_pretrained(
        cls,
        source_root: str | Path,
        checkpoint_path: str | Path,
        *,
        variant: str = "tiny",
        device: str | torch.device = "cuda",
    ) -> "OfficialSAM2MattingClipBackbone":
        root = Path(source_root).resolve()
        if not (root / "sam2" / "sam2matting_video_predictor.py").is_file():
            raise FileNotFoundError("source_root must be the released SAM2Matting checkout")
        checkpoint_file = Path(checkpoint_path).resolve()
        if not checkpoint_file.is_file():
            raise FileNotFoundError(checkpoint_file)
        imported = sys.modules.get("sam2")
        if imported is not None and not Path(imported.__file__).resolve().is_relative_to(root):
            raise RuntimeError(
                "launch with PRISM_SAM2_ROOT=<official-sam2matting-root> "
                "before importing refractive_mam2"
            )
        sys.path.insert(0, str(root))
        try:
            build = importlib.import_module("sam2.build_sam")
        except ImportError as exc:
            raise ImportError("install released SAM2Matting dependencies") from exc
        finally:
            sys.path.remove(str(root))
        if not Path(build.__file__).resolve().is_relative_to(root):
            raise RuntimeError("imported sam2 is not from SAM2Matting source_root")
        configurations = {
            "tiny": "configs/sam2matting-sam2.1tiny.yaml",
            "base_plus": "configs/sam2matting-sam2.1base+.yaml",
        }
        if variant not in configurations:
            raise ValueError(f"unsupported released SAM2Matting variant: {variant}")
        model = build.build_sam2matting_video_predictor(
            configurations[variant], str(checkpoint_file), device=str(device)
        )
        return cls(model)

    def train(self, mode: bool = True) -> "OfficialSAM2MattingClipBackbone":
        super().train(mode)
        self.main.eval()  # Freeze running BN statistics in the released prior.
        return self

    def configure_joint_trainable(self) -> list[nn.Parameter]:
        self.requires_grad_(False)
        for name in _ALPHA_MODULES:
            module = getattr(self.main, name, None)
            if module is None:
                raise RuntimeError(f"released SAM2Matting is missing {name}")
            module.requires_grad_(True)
        self.mam2_matter.flow_feature_modulator.requires_grad_(True)
        self.main.eval()
        return [value for value in self.parameters() if value.requires_grad]

    def clear_mam2_cache(self) -> None:
        self.mam2_matter.clear_cached_inputs()

    def mam2_extension_state_dict(self) -> dict[str, Tensor]:
        prefixes = tuple(f"main.{name}." for name in _ALPHA_MODULES) + (
            "mam2_matter.flow_feature_modulator.",
        )
        return {
            key: value.detach().cpu()
            for key, value in self.state_dict().items()
            if key.startswith(prefixes)
        }

    def load_mam2_extension_state_dict(
        self, state: dict[str, Tensor], *, strict: bool = True
    ) -> None:
        expected = set(self.mam2_extension_state_dict())
        if strict and set(state) != expected:
            raise RuntimeError(
                f"SAM2Matting extension mismatch: missing={sorted(expected - set(state))}; "
                f"unexpected={sorted(set(state) - expected)}"
            )
        self.load_state_dict(state, strict=False)

    @staticmethod
    def _write_frames(path: Path, srgb: Tensor) -> None:
        from PIL import Image

        for index, frame in enumerate(srgb):
            pixels = frame.detach().clamp(0, 1).mul(255).round().byte()
            array = pixels.permute(1, 2, 0).cpu().numpy()
            Image.fromarray(array, mode="RGB").save(path / f"{index:05d}.png")

    def _one_clip(
        self,
        normalized: Tensor,
        *,
        output_size: tuple[int, int],
        mask: Tensor | None,
        point_coords: Tensor | None,
        point_labels: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, list[tuple[Tensor, Tensor, list[Tensor]]]]:
        mean = normalized.new_tensor(_SAM2_MEAN).view(1, 3, 1, 1)
        std = normalized.new_tensor(_SAM2_STD).view(1, 3, 1, 1)
        srgb = (normalized * std + mean).clamp(0, 1)
        srgb = F.interpolate(
            srgb, size=output_size, mode="bilinear", align_corners=False,
        )
        with tempfile.TemporaryDirectory(prefix="prism-official-sam2matting-") as temp:
            self._write_frames(Path(temp), srgb)
            with torch.inference_mode():
                state = self.main.init_state(
                    # The released alpha head concatenates decoded frames with
                    # CUDA backbone features inside add_new_mask. CPU video
                    # offload is therefore incompatible with this path.
                    video_path=temp, offload_video_to_cpu=False,
                    offload_state_to_cpu=False, async_loading_frames=False,
                )
                try:
                    images = state["images"]
                    if torch.is_tensor(images):
                        model_device = next(self.main.parameters()).device
                        if images.device != model_device:
                            raise RuntimeError(
                                "SAM2Matting frames and alpha features must share a device: "
                                f"frames={images.device}, model={model_device}"
                            )
                    if mask is not None:
                        self.main.add_new_mask(state, 0, 1, (mask[0] >= 0.5))
                    else:
                        if point_coords is None or point_labels is None:
                            raise ValueError("SAM2Matting needs a first-frame prompt")
                        self.main.add_new_points_or_box(
                            state, 0, 1, points=point_coords / self.image_size,
                            labels=point_labels, normalize_coords=False,
                        )
                    mask_by_frame: dict[int, Tensor] = {}
                    alpha_by_frame: dict[int, Tensor] = {}
                    for index, object_ids, logits, alpha, _ in self.main.propagate_in_video(state):
                        if 1 not in list(object_ids):
                            raise RuntimeError("SAM2Matting lost prompted object 1")
                        position = list(object_ids).index(1)
                        mask_by_frame[index] = logits[position:position + 1].detach()
                        if alpha.ndim == 3:
                            alpha = alpha.unsqueeze(0)
                        alpha_by_frame[index] = alpha[position:position + 1].detach()
                    count = normalized.shape[0]
                    if set(mask_by_frame) != set(range(count)):
                        raise RuntimeError("SAM2Matting did not cover every clip frame")
                    features: list[Tensor] = []
                    inputs: list[tuple[Tensor, Tensor, list[Tensor]]] = []
                    for index in range(count):
                        _, backbone, *_ = self.main._get_image_feature(state, index, 1)
                        pyramid = [feature.detach() for feature in backbone["backbone_fpn"]]
                        features.append(pyramid[-1][0])
                        native_mask = mask_by_frame[index]
                        inputs.append((
                            state["images"][index:index + 1].detach(),
                            (F.interpolate(
                                native_mask.float(), size=pyramid[0].shape[-2:],
                                mode="bilinear", align_corners=False,
                            ) > 0).to(pyramid[0].dtype),
                            pyramid,
                        ))
                    masks = torch.stack([mask_by_frame[i][0] for i in range(count)])
                    alphas = torch.stack([alpha_by_frame[i][0] for i in range(count)])
                    feature_video = torch.stack(features)
                finally:
                    self.main.reset_state(state)
                    close = getattr(state["images"], "close", None)
                    if callable(close):
                        close()
        released = [
            (image.clone(), mask.clone(), [value.clone() for value in pyramid])
            for image, mask, pyramid in inputs
        ]
        return masks.clone(), alphas.clone(), feature_video.clone(), released

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
            raise ValueError("SAM2Matting is a pretrained Stage-2/3/4 backend")
        if (first_frame_mask_inputs is None) == (first_frame_point_inputs is None):
            raise ValueError("provide exactly one prompt type")
        if original_size is None:
            raise ValueError("native frame size is required")
        batch = normalized_frames.shape[0]
        masks, alphas, features = [], [], []
        all_inputs: list[tuple[Tensor, Tensor, list[Tensor]]] = []
        self.mam2_matter.clear_cached_inputs()
        for index in range(batch):
            prompt = first_frame_point_inputs
            mask, alpha, feature, inputs = self._one_clip(
                normalized_frames[index], output_size=original_size,
                mask=None if first_frame_mask_inputs is None else first_frame_mask_inputs[index],
                point_coords=None if prompt is None else prompt["point_coords"][index],
                point_labels=None if prompt is None else prompt["point_labels"][index],
            )
            masks.append(mask)
            alphas.append(alpha)
            features.append(feature)
            all_inputs.extend(inputs)
        self.mam2_matter.set_cached_inputs(all_inputs)
        logits = torch.stack(masks)
        alpha = torch.stack(alphas).clamp(0, 1)
        # Adapter contract only: SAM2Matting itself has no PDD trimap output.
        trimap_logits = self.mam2_matter.trimap_from_alpha(alpha, logits)
        result = MAM2BackboneOutput(logits, trimap_logits, alpha, torch.stack(features))
        result.validate()
        return result

    def forward(self, frames: Tensor, prompts: Any | None = None) -> MAM2BackboneOutput:
        raise RuntimeError("use forward_mam2_clip with explicit prompts")
