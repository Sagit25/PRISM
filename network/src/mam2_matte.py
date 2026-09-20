from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .config import MAM2MatteConfig


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
        if width < 1 or self.config.depth < 1:
            raise ValueError("MAM2 matte width and depth must be positive")
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
        alpha = (foreground + unknown * unknown_alpha).clamp(0.0, 1.0)
        return alpha.reshape(batch, time, 1, height, width)


class ExternalMEMatteMatter(nn.Module):
    """Differentiable adapter for the official MEMatte implementation.

    MEMatte's ViT encoder remains frozen.  Stage 1B and joint fine-tuning may
    update only its detail decoder, while gradients through the soft trimap
    continue into MAM2's PDD/MSS when that path is enabled.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        patch_decoder: bool = True,
        train_decoder: bool = True,
    ) -> None:
        super().__init__()
        self.external_model = model.requires_grad_(False).eval()
        self.patch_decoder = patch_decoder
        self.train_decoder = train_decoder

    def configure_trainable(self, enabled: bool) -> list[nn.Parameter]:
        """Freeze MEMatte globally, then optionally expose its decoder."""

        self.external_model.requires_grad_(False)
        decoder = getattr(self.external_model, "decoder", None)
        if enabled and self.train_decoder:
            if decoder is None:
                raise AttributeError("official MEMatte model has no decoder module")
            decoder.requires_grad_(True)
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def train(self, mode: bool = True) -> "ExternalMEMatteMatter":
        # Keep official MEMatte on its inference branch (which returns alpha
        # rather than training losses). eval() does not disable autograd.
        super().train(mode)
        self.external_model.eval()
        return self

    def forward(self, frames: Tensor, trimap_logits: Tensor) -> Tensor:
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError("frames must have shape [B,T,3,H,W]")
        if trimap_logits.ndim != 5 or trimap_logits.shape[2] != 3:
            raise ValueError("trimap_logits must have shape [B,T,3,h,w]")
        if frames.shape[:2] != trimap_logits.shape[:2]:
            raise ValueError("frames and trimap batch/time dimensions must match")
        batch, time, _, height, width = frames.shape
        flat_frames = frames.reshape(batch * time, 3, height, width)
        probabilities = F.interpolate(
            trimap_logits.reshape(
                batch * time, 3, *trimap_logits.shape[-2:]
            ),
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
        result = self.external_model(
            {"image": flat_frames, "trimap": scalar_trimap},
            patch_decoder=self.patch_decoder,
        )
        outputs = result[0] if isinstance(result, tuple) else result
        if not isinstance(outputs, dict) or "phas" not in outputs:
            raise RuntimeError("official MEMatte backend did not return outputs['phas']")
        alpha = outputs["phas"]
        if alpha.shape != (batch * time, 1, height, width):
            raise RuntimeError(
                "official MEMatte alpha has an unexpected shape: "
                f"{tuple(alpha.shape)}"
            )
        alpha = (foreground + unknown * alpha).clamp(0.0, 1.0)
        return alpha.reshape(batch, time, 1, height, width)


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
    )
