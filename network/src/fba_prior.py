"""Optional official FBA Matting prior for the flow-guided PRISM variant.

The upstream source is supplied separately (not vendored here):
https://github.com/MarcoForte/FBA_Matting
Its Adobe-trained checkpoint is noncommercial. PRISM uses a pre-fusion decoder
feature, not FBA's conventional alpha-composited F or B predictions.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .color import linear_to_srgb
from .config import FBA_PRE_FUSION_CHANNELS
from .types import MAM2BackboneOutput


class FBAPretrainedForegroundPrior(nn.Module):
    """Run a pretrained FBA matting network once per video clip.

    Return the 16-channel feature before FBA's final RGB/alpha head. FBA's
    conventional fusion formula mixes the apparent refracted background into
    its F prediction on transparent objects. PRISM learns its physical F from
    these features together with flow-aligned counterfactual-background cues.
    """

    FEATURE_CHANNELS = FBA_PRE_FUSION_CHANNELS

    def __init__(
        self,
        source_root: str | Path,
        checkpoint: str | Path,
        *,
        input_side: int = 320,
        train_decoder: bool = False,
    ) -> None:
        super().__init__()
        root = Path(source_root).resolve()
        weights = Path(checkpoint).resolve()
        if not (root / "networks" / "models.py").is_file():
            raise FileNotFoundError("official FBA networks/models.py is missing")
        if not weights.is_file():
            raise FileNotFoundError("official FBA checkpoint is missing")
        if input_side < 64 or input_side % 8:
            raise ValueError("FBA input_side must be >=64 and divisible by 8")
        self.input_side = input_side
        self.train_decoder = train_decoder

        # The official source imports its own top-level `networks` package.
        # Reject a preloaded package from elsewhere rather than silently
        # constructing a different model with the official checkpoint.
        existing = sys.modules.get("networks.models")
        if existing is not None and not Path(existing.__file__).resolve().is_relative_to(root):
            raise RuntimeError("another networks.models package is already loaded")
        sys.path.insert(0, str(root))
        try:
            fba_models = importlib.import_module("networks.models")
        finally:
            sys.path.remove(str(root))
        if not Path(fba_models.__file__).resolve().is_relative_to(root):
            raise RuntimeError("FBA source resolved outside source_root")
        self.model = fba_models.build_model("default")
        state = torch.load(weights, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.encoder.requires_grad_(False)
        self.model.decoder.requires_grad_(train_decoder)

    def configure_trainability(self) -> None:
        self.model.encoder.requires_grad_(False)
        self.model.decoder.requires_grad_(self.train_decoder)
        self.model.encoder.eval()
        self.model.decoder.train(self.train_decoder)

    def forward(self, frames: Tensor, backbone: MAM2BackboneOutput) -> Tensor:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("FBA foreground prior requires opencv-python-headless") from exc
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError("frames must be [B,T,3,H,W]")
        batch, time, _, height, width = frames.shape
        if backbone.mask_logits.shape[:2] != (batch, time):
            raise ValueError("FBA mask batch/time must match frames")
        flat_frames = frames.reshape(batch * time, 3, height, width)
        resized_image = F.interpolate(
            flat_frames,
            size=(self.input_side, self.input_side),
            mode="bilinear",
            align_corners=False,
        )
        # PRISM's renderer and labels use linear RGB, whereas FBA was trained
        # on ordinary display-space images. Convert only the FBA input;
        # returned decoder features are not RGB and must not be gamma-decoded.
        resized_image = linear_to_srgb(resized_image)
        object_probability = F.interpolate(
            backbone.mask_logits.reshape(
                batch * time, 1, *backbone.mask_logits.shape[-2:]
            ).detach(),
            size=(self.input_side, self.input_side),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()
        alpha_probability = F.interpolate(
            backbone.alpha_matte.reshape(
                batch * time, 1, *backbone.alpha_matte.shape[-2:]
            ).detach(),
            size=(self.input_side, self.input_side),
            mode="bilinear",
            align_corners=False,
        ).clamp(0, 1)
        certain_background = object_probability < 0.05
        certain_foreground = (object_probability > 0.95) & (alpha_probability > 0.95)
        two_channel_trimap = torch.cat(
            (certain_background, certain_foreground), dim=1
        ).to(resized_image.dtype)
        # Match the official six-channel distance-transform conditioning.
        # The trimap is intentionally detached; no GT or future frame is used.
        trimap_np = two_channel_trimap.detach().cpu().numpy()
        transformed = []
        for trimap in trimap_np:
            channels = []
            for channel in trimap:
                distance = cv2.distanceTransform(
                    ((1.0 - channel) * 255).astype(np.uint8),
                    cv2.DIST_L2,
                    0,
                )
                for fraction in (0.02, 0.08, 0.16):
                    channels.append(
                        np.exp(
                            -np.square(distance)
                            / (2 * (fraction * self.input_side) ** 2)
                        )
                    )
            transformed.append(np.stack(channels, axis=0))
        distance_channels = torch.as_tensor(
            np.stack(transformed, axis=0),
            device=resized_image.device,
            dtype=resized_image.dtype,
        )
        mean = resized_image.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = resized_image.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        normalized_image = (resized_image - mean) / std
        self.model.encoder.eval()
        final_features: list[Tensor] = []
        try:
            feature_layer = self.model.decoder.conv_up4[-2]
        except (AttributeError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "FBA decoder does not expose the expected pre-fusion feature layer"
            ) from exc
        hook = feature_layer.register_forward_hook(
            lambda _module, _inputs, output: final_features.append(output)
        )
        try:
            if not self.train_decoder:
                with torch.no_grad():
                    self.model(
                        resized_image,
                        two_channel_trimap,
                        normalized_image,
                        distance_channels,
                    )
            else:
                self.model(
                    resized_image,
                    two_channel_trimap,
                    normalized_image,
                    distance_channels,
                )
        finally:
            hook.remove()
        if len(final_features) != 1:
            raise RuntimeError("FBA decoder feature hook did not run exactly once")
        prior = final_features[0]
        if prior.shape != (
            batch * time,
            self.FEATURE_CHANNELS,
            self.input_side,
            self.input_side,
        ):
            raise RuntimeError("FBA pre-fusion decoder feature shape is unexpected")
        prior = F.interpolate(
            prior,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        return prior.reshape(batch, time, self.FEATURE_CHANNELS, height, width)
