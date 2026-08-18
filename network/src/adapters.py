from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import Tensor, nn

from .types import MAM2BackboneOutput


class DictionaryBackboneAdapter(nn.Module):
    """Adapt a backbone returning a dict to the normalized project contract.

    This keeps the physics code independent of eventual MAM2 repository names.
    The wrapped model must return tensors with video dimensions [B,T,...].
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        mask_key: str = "mask_logits",
        trimap_key: str = "trimap_logits",
        feature_key: str = "non_memory_features",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.mask_key = mask_key
        self.trimap_key = trimap_key
        self.feature_key = feature_key

    def forward(
        self,
        frames: Tensor,
        prompts: Any | None = None,
    ) -> MAM2BackboneOutput:
        raw = self.backbone(frames, prompts)
        if isinstance(raw, MAM2BackboneOutput):
            raw.validate()
            return raw
        if not isinstance(raw, Mapping):
            raise TypeError("wrapped backbone must return a mapping or MAM2BackboneOutput")
        try:
            output = MAM2BackboneOutput(
                mask_logits=raw[self.mask_key],
                trimap_logits=raw[self.trimap_key],
                non_memory_features=raw[self.feature_key],
            )
        except KeyError as exc:
            raise KeyError(f"missing MAM2 output key: {exc.args[0]}") from exc
        output.validate()
        return output
