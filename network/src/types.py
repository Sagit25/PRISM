from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from torch import Tensor, nn


@dataclass
class MAM2BackboneOutput:
    """Normalized contract expected from a MAM2 PDD/MSS implementation.

    `non_memory_features` must be the clean feature used by the second MSS
    decode, not the SAM2 feature after mask-memory injection.
    """

    mask_logits: Tensor
    trimap_logits: Tensor
    alpha_matte: Tensor
    non_memory_features: Tensor

    def validate(self) -> None:
        if self.mask_logits.ndim != 5 or self.mask_logits.shape[2] != 1:
            raise ValueError("mask_logits must have shape [B,T,1,H,W]")
        if self.trimap_logits.ndim != 5 or self.trimap_logits.shape[2] != 3:
            raise ValueError("trimap_logits must have shape [B,T,3,H,W]")
        if self.non_memory_features.ndim != 5:
            raise ValueError("non_memory_features must have shape [B,T,C,h,w]")
        if self.alpha_matte.ndim != 5 or self.alpha_matte.shape[2] != 1:
            raise ValueError("alpha_matte must have shape [B,T,1,H,W]")
        if self.mask_logits.shape[:2] != self.trimap_logits.shape[:2]:
            raise ValueError("mask and trimap batch/time dimensions must match")
        if self.mask_logits.shape[:2] != self.non_memory_features.shape[:2]:
            raise ValueError("features and predictions batch/time dimensions must match")
        if self.mask_logits.shape[:2] != self.alpha_matte.shape[:2]:
            raise ValueError("alpha and semantic batch/time dimensions must match")
        if not bool(((self.alpha_matte >= 0) & (self.alpha_matte <= 1)).all()):
            raise ValueError("alpha_matte values must be in [0,1]")


@runtime_checkable
class MAM2Backbone(Protocol):
    """Protocol for the SAM2 + PDD + MSS portion of the model."""

    def __call__(
        self,
        frames: Tensor,
        prompts: Any | None = None,
    ) -> MAM2BackboneOutput: ...


class BackboneModule(nn.Module):
    """Optional nominal base class for projects that prefer nn.Module typing."""

    def forward(
        self,
        frames: Tensor,
        prompts: Any | None = None,
    ) -> MAM2BackboneOutput:
        raise NotImplementedError
