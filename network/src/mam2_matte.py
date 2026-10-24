from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .config import MAM2MatteConfig
from .renderer import warp_background


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int, *, activation_checkpointing: bool) -> None:
        super().__init__()
        self.activation_checkpointing = activation_checkpointing
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(_group_count(channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(_group_count(channels), channels),
        )
        self.activation = nn.GELU()

    def _forward_impl(self, value: Tensor) -> Tensor:
        return self.activation(value + self.body(value))

    def forward(self, value: Tensor) -> Tensor:
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(self._forward_impl, value, use_reentrant=False)
        return self._forward_impl(value)


class MAM2TrimapMatter(nn.Module):
    """Predict MAM2's alpha matte from the RGB frame and predicted trimap.

    MAM2 deliberately keeps this boundary replaceable: the paper instantiates
    it with MEMatte, while PRISM ships this small implementation so the full
    mask -> trimap -> alpha contract is executable without vendoring a second
    research repository.  Known trimap regions are enforced analytically; the
    network estimates opacity only in the unknown region.
    """

    def __init__(self, config: MAM2MatteConfig | None = None) -> None:
        super().__init__()
        self.config = config or MAM2MatteConfig()
        width = self.config.width
        if width < 1 or self.config.depth < 1 or self.config.frame_chunk_size < 1:
            raise ValueError(
                "MAM2 matte width, depth, and frame_chunk_size must be positive"
            )
        self.stem = nn.Sequential(
            nn.Conv2d(6, width, 5, padding=2),
            nn.GroupNorm(_group_count(width), width),
            nn.GELU(),
        )
        self.encoder = nn.Sequential(
            *[
                _ResidualBlock(
                    width,
                    activation_checkpointing=self.config.activation_checkpointing,
                )
                for _ in range(self.config.depth)
            ],
            nn.Conv2d(width, width * 2, 4, stride=2, padding=1),
            nn.GroupNorm(_group_count(width * 2), width * 2),
            nn.GELU(),
            *[
                _ResidualBlock(
                    width * 2,
                    activation_checkpointing=self.config.activation_checkpointing,
                )
                for _ in range(self.config.depth)
            ],
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(width * 2, width, 4, stride=2, padding=1),
            nn.GroupNorm(_group_count(width), width),
            nn.GELU(),
            *[
                _ResidualBlock(
                    width,
                    activation_checkpointing=self.config.activation_checkpointing,
                )
                for _ in range(self.config.depth)
            ],
        )
        self.alpha_head = nn.Conv2d(width * 2, 1, 3, padding=1)

    def _forward_flat(self, flat_frames: Tensor, flat_trimap: Tensor) -> Tensor:
        height, width = flat_frames.shape[-2:]
        probabilities = F.interpolate(
            flat_trimap,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).softmax(dim=1)
        shallow = self.stem(torch.cat((flat_frames, probabilities), dim=1))
        deep = self.encoder(shallow)
        decoded = self.decoder(deep)
        if decoded.shape[-2:] != shallow.shape[-2:]:
            decoded = F.interpolate(
                decoded,
                size=shallow.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        unknown_alpha = self.alpha_head(torch.cat((shallow, decoded), dim=1)).sigmoid()

        if not self.training and self.config.hard_trimap_at_inference:
            classes = probabilities.argmax(dim=1, keepdim=True)
            foreground = (classes == 2).to(unknown_alpha.dtype)
            unknown = (classes == 1).to(unknown_alpha.dtype)
        else:
            foreground = probabilities[:, 2:3]
            unknown = probabilities[:, 1:2]
        return (foreground + unknown * unknown_alpha).clamp(0.0, 1.0)

    def forward(self, frames: Tensor, trimap_logits: Tensor) -> Tensor:
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError("frames must have shape [B,T,3,H,W]")
        if trimap_logits.ndim != 5 or trimap_logits.shape[2] != 3:
            raise ValueError("trimap_logits must have shape [B,T,3,h,w]")
        if frames.shape[:2] != trimap_logits.shape[:2]:
            raise ValueError("frames and trimap batch/time dimensions must match")

        batch, time, _, height, width = frames.shape
        flat_frames = frames.reshape(batch * time, 3, height, width)
        flat_trimap = trimap_logits.reshape(
            batch * time, 3, *trimap_logits.shape[-2:]
        )
        outputs: list[Tensor] = []
        for start in range(0, batch * time, self.config.frame_chunk_size):
            end = min(start + self.config.frame_chunk_size, batch * time)
            frame_chunk = flat_frames[start:end]
            trimap_chunk = flat_trimap[start:end]
            if (
                self.config.full_activation_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                alpha_chunk = checkpoint(
                    self._forward_flat,
                    frame_chunk,
                    trimap_chunk,
                    use_reentrant=False,
                )
            else:
                alpha_chunk = self._forward_flat(frame_chunk, trimap_chunk)
            outputs.append(alpha_chunk)
        return torch.cat(outputs, dim=0).reshape(batch, time, 1, height, width)


class ExternalMEMatteMatter(nn.Module):
    """Differentiable adapter for the official MEMatte implementation.

    Stage 1B and joint fine-tuning update the adaptive-token backbone and
    detail decoder by default, matching MAM2's trainable matter stage. A
    decoder-only mode remains available as a lower-memory ablation. Gradients
    through the soft trimap continue into PDD/MSS when that path is enabled.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        patch_decoder: bool = True,
        train_decoder: bool = True,
        train_backbone: bool = False,
        frame_chunk_size: int = 1,
        full_activation_checkpointing: bool = True,
        flow_feature_guidance: bool = False,
    ) -> None:
        super().__init__()
        if frame_chunk_size < 1:
            raise ValueError("frame_chunk_size must be positive")
        self.external_model = model.requires_grad_(False).eval()
        self.patch_decoder = patch_decoder
        self.train_decoder = train_decoder
        self.train_backbone = train_backbone
        self.frame_chunk_size = frame_chunk_size
        self.full_activation_checkpointing = full_activation_checkpointing
        self.flow_feature_guidance = flow_feature_guidance
        # Inject flow-aligned background evidence into the official MEMatte
        # backbone-to-decoder feature, rather than changing its RGB input or
        # treating a SAM2 feature as a MEMatte feature.  The final projection
        # starts at zero so loading the official checkpoint preserves its
        # original prediction until this adapter is trained.
        self.flow_feature_modulator = None
        if flow_feature_guidance:
            self.flow_feature_modulator = nn.Sequential(
                nn.Conv2d(9, 32, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(32, 2, 3, padding=1),
            )
            nn.init.zeros_(self.flow_feature_modulator[-1].weight)
            nn.init.zeros_(self.flow_feature_modulator[-1].bias)

    def configure_trainable(self, enabled: bool) -> list[nn.Parameter]:
        """Freeze MEMatte globally, then expose configured matter modules."""

        self.external_model.requires_grad_(False)
        backbone = getattr(self.external_model, "backbone", None)
        decoder = getattr(self.external_model, "decoder", None)
        if enabled and self.train_backbone:
            if backbone is None:
                raise AttributeError("official MEMatte model has no backbone module")
            backbone.requires_grad_(True)
        if enabled and self.train_decoder:
            if decoder is None:
                raise AttributeError("official MEMatte model has no decoder module")
            decoder.requires_grad_(True)
        if self.flow_feature_modulator is not None:
            self.flow_feature_modulator.requires_grad_(enabled)
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def train(self, mode: bool = True) -> "ExternalMEMatteMatter":
        super().train(mode)
        self.external_model.eval()
        if mode and self.train_backbone:
            self.external_model.backbone.train()
        if mode and self.train_decoder:
            self.external_model.decoder.train()
        return self

    def _run_external(
        self,
        flat_frames: Tensor,
        scalar_trimap: Tensor,
        flow_context: Tensor | None = None,
    ) -> dict[str, Tensor]:
        # Official MEMatte's train-mode forward returns its private criterion
        # rather than alpha. Use the same public submodules directly so PRISM's
        # paper alpha loss owns supervision while backbone/decoder keep their
        # train-mode behavior (routing, drop path, normalization).
        if (self.training or flow_context is not None) and hasattr(
            self.external_model, "preprocess_inputs"
        ):
            images, _, height, width = self.external_model.preprocess_inputs(
                {"image": flat_frames, "trimap": scalar_trimap}
            )
            backbone_result = self.external_model.backbone(images)
            features = (
                backbone_result[0]
                if isinstance(backbone_result, tuple)
                else backbone_result
            )
            if flow_context is not None:
                if self.flow_feature_modulator is None:
                    raise RuntimeError("MEMatte flow feature guidance is disabled")
                if features.ndim != 4 or features.shape[0] != flow_context.shape[0]:
                    raise RuntimeError("official MEMatte backbone returned invalid features")
                modulation = self.flow_feature_modulator(flow_context.float())
                modulation = F.interpolate(
                    modulation, size=features.shape[-2:], mode="bilinear", align_corners=False
                ).to(features.dtype)
                scale, shift = modulation.tanh().chunk(2, dim=1)
                features = features * (1.0 + 0.25 * scale) + 0.25 * shift
            outputs = self.external_model.decoder(features, images)
            if not isinstance(outputs, dict):
                raise RuntimeError("official MEMatte decoder did not return a dictionary")
            outputs = dict(outputs)
            outputs["phas"] = outputs["phas"][..., :height, :width]
            return outputs

        result = self.external_model(
            {"image": flat_frames, "trimap": scalar_trimap},
            patch_decoder=self.patch_decoder,
        )
        outputs = result[0] if isinstance(result, tuple) else result
        if not isinstance(outputs, dict):
            raise RuntimeError("official MEMatte backend did not return a dictionary")
        return outputs

    def _forward_flat(
        self, flat_frames: Tensor, flat_trimap: Tensor,
        flow_context: Tensor | None = None,
    ) -> Tensor:
        height, width = flat_frames.shape[-2:]
        probabilities = F.interpolate(
            flat_trimap,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).softmax(dim=1)
        if self.training:
            # Differentiable 0/0.5/1 expectation.  This preserves a gradient
            # from alpha supervision back to the trimap logits.
            scalar_trimap = probabilities[:, 2:3] + 0.5 * probabilities[:, 1:2]
            foreground = probabilities[:, 2:3]
            unknown = probabilities[:, 1:2]
        else:
            classes = probabilities.argmax(dim=1, keepdim=True)
            foreground = (classes == 2).to(flat_frames.dtype)
            unknown = (classes == 1).to(flat_frames.dtype)
            scalar_trimap = foreground + 0.5 * unknown
        outputs = self._run_external(flat_frames, scalar_trimap, flow_context)
        if not isinstance(outputs, dict) or "phas" not in outputs:
            raise RuntimeError("official MEMatte backend did not return outputs['phas']")
        alpha = outputs["phas"]
        if alpha.shape != (flat_frames.shape[0], 1, height, width):
            raise RuntimeError(
                "official MEMatte alpha has an unexpected shape: "
                f"{tuple(alpha.shape)}"
            )
        return (foreground + unknown * alpha).clamp(0.0, 1.0)

    def forward_with_flow(
        self,
        frames: Tensor,
        trimap_logits: Tensor,
        background: Tensor,
        refractive_flow: Tensor,
        render_error: Tensor,
        object_support: Tensor,
    ) -> Tensor:
        """Refine official MEMatte alpha using the current optical estimate.

        The feature FiLM receives B(x+u), render error, normalized u and
        object support.  Its output remains inside the official MEMatte
        decoder and is optimized jointly with the PRISM optical operator.
        """
        if self.flow_feature_modulator is None:
            raise RuntimeError("MEMatte flow feature guidance is disabled")
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError("frames must have shape [B,T,3,H,W]")
        batch, time, _, height, width = frames.shape
        if background.shape != (batch, 3, height, width):
            raise ValueError("background must have shape [B,3,H,W]")
        if refractive_flow.shape != (batch, time, 2, height, width):
            raise ValueError("refractive_flow must have shape [B,T,2,H,W]")
        if render_error.shape != frames.shape:
            raise ValueError("render_error must match frames")
        if object_support.shape != (batch, time, 1, height, width):
            raise ValueError("object_support must have shape [B,T,1,H,W]")
        if trimap_logits.shape[:3] != (batch, time, 3):
            raise ValueError("trimap_logits must have shape [B,T,3,h,w]")
        background_video = background[:, None].expand(-1, time, -1, -1, -1)
        aligned_background = warp_background(background_video, refractive_flow)
        flow_scale = refractive_flow.new_tensor((max(width - 1, 1), max(height - 1, 1)))
        normalized_flow = refractive_flow / flow_scale.view(1, 1, 2, 1, 1)
        context = torch.cat(
            (aligned_background, render_error, normalized_flow, object_support),
            dim=2,
        ).reshape(batch * time, 9, height, width)
        flat_frames = frames.reshape(batch * time, 3, height, width)
        flat_trimap = trimap_logits.reshape(batch * time, 3, *trimap_logits.shape[-2:])
        outputs: list[Tensor] = []
        for start in range(0, batch * time, self.frame_chunk_size):
            end = min(start + self.frame_chunk_size, batch * time)
            values = (flat_frames[start:end], flat_trimap[start:end], context[start:end])
            if self.full_activation_checkpointing and self.training and torch.is_grad_enabled():
                alpha_chunk = checkpoint(self._forward_flat, *values, use_reentrant=False)
            else:
                alpha_chunk = self._forward_flat(*values)
            outputs.append(alpha_chunk)
        return torch.cat(outputs, dim=0).reshape(batch, time, 1, height, width)

    def forward(self, frames: Tensor, trimap_logits: Tensor) -> Tensor:
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError("frames must have shape [B,T,3,H,W]")
        if trimap_logits.ndim != 5 or trimap_logits.shape[2] != 3:
            raise ValueError("trimap_logits must have shape [B,T,3,h,w]")
        if frames.shape[:2] != trimap_logits.shape[:2]:
            raise ValueError("frames and trimap batch/time dimensions must match")
        batch, time, _, height, width = frames.shape
        flat_frames = frames.reshape(batch * time, 3, height, width)
        flat_trimap = trimap_logits.reshape(
            batch * time, 3, *trimap_logits.shape[-2:]
        )
        outputs: list[Tensor] = []
        for start in range(0, batch * time, self.frame_chunk_size):
            end = min(start + self.frame_chunk_size, batch * time)
            frame_chunk = flat_frames[start:end]
            trimap_chunk = flat_trimap[start:end]
            if (
                self.full_activation_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                alpha_chunk = checkpoint(
                    self._forward_flat,
                    frame_chunk,
                    trimap_chunk,
                    use_reentrant=False,
                )
            else:
                alpha_chunk = self._forward_flat(frame_chunk, trimap_chunk)
            outputs.append(alpha_chunk)
        return torch.cat(outputs, dim=0).reshape(batch, time, 1, height, width)


def build_mam2_matter(config: MAM2MatteConfig) -> nn.Module:
    """Build the dependency-light fallback or the official MEMatte adapter."""

    if config.backend == "builtin":
        return MAM2TrimapMatter(config)
    if config.backend != "external_mematte":
        raise ValueError(f"unsupported MAM2 matter backend: {config.backend!r}")
    required = {
        "external_root": config.external_root,
        "external_config": config.external_config,
        "external_checkpoint": config.external_checkpoint,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(
            "external_mematte requires " + ", ".join(sorted(missing))
        )
    root = Path(config.external_root).expanduser().resolve()
    config_path = Path(config.external_config).expanduser().resolve()
    checkpoint_path = Path(config.external_checkpoint).expanduser().resolve()
    if not (root / "modeling" / "meta_arch" / "mematte.py").is_file():
        raise FileNotFoundError(f"invalid official MEMatte root: {root}")
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    root_string = str(root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    try:
        from detectron2.checkpoint import DetectionCheckpointer
        from detectron2.config import LazyConfig, instantiate
    except ImportError as exc:  # pragma: no cover - optional external stack
        raise ImportError(
            "external MEMatte requires its official dependencies, including detectron2"
        ) from exc
    lazy_config = LazyConfig.load(str(config_path))
    lazy_config.model.teacher_backbone = None
    lazy_config.model.backbone.max_number_token = config.external_max_tokens
    model = instantiate(lazy_config.model).eval()
    DetectionCheckpointer(model).load(str(checkpoint_path))
    return ExternalMEMatteMatter(
        model,
        patch_decoder=config.external_patch_decoder,
        train_decoder=config.external_train_decoder,
        train_backbone=config.external_train_backbone,
        frame_chunk_size=config.frame_chunk_size,
        full_activation_checkpointing=config.full_activation_checkpointing,
        flow_feature_guidance=config.flow_feature_guidance,
    )
