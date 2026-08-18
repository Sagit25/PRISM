from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import PDDConfig


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x + self.body(x))


@dataclass
class PDDOutput:
    mask_logits: Tensor
    trimap_logits: Tensor
    decoded_features: Tensor


class PromptableDualModeDecoder(nn.Module):
    """Prompt-conditioned mask/trimap decoder for the MAM2 reproduction.

    SAM2 sparse prompt tokens are injected through cross attention and SAM2's
    dense prompt embedding is fused spatially. The mask remains a residual over
    the official SAM2 prediction. A high-resolution refinement path combines
    clean FPN detail with the mask-augmentation feature before trimap output.
    """

    def __init__(self, config: PDDConfig | None = None) -> None:
        super().__init__()
        self.config = config or PDDConfig()
        width = self.config.width
        refinement_width = self.config.refinement_width
        channels = self.config.feature_channels
        if width % self.config.prompt_heads != 0:
            raise ValueError("PDD width must be divisible by prompt_heads")

        self.feature_projection = nn.Sequential(
            nn.Conv2d(channels, width, 1),
            nn.GroupNorm(_group_count(width), width),
            nn.GELU(),
        )
        self.dense_prompt_projection = nn.Conv2d(channels, width, 1)
        self.sparse_prompt_projection = nn.Linear(channels, width)
        self.prompt_attention = nn.MultiheadAttention(
            width,
            self.config.prompt_heads,
            batch_first=True,
        )
        self.shared_decoder = nn.Sequential(
            *[_ResidualBlock(width) for _ in range(self.config.depth)]
        )
        self.mask_head = nn.Conv2d(width, 1, 1)

        self.refinement_projection = nn.Conv2d(width, refinement_width, 1)
        self.mask_augmentation = nn.Sequential(
            nn.Conv2d(1, refinement_width, 3, padding=1),
            nn.GroupNorm(_group_count(refinement_width), refinement_width),
            nn.GELU(),
        )
        # Channel-mean FPN detail is architecture-stable across released Hiera
        # variants while still preserving high-frequency spatial information.
        self.high_res_projection = nn.Sequential(
            nn.Conv2d(1, refinement_width, 3, padding=1),
            nn.GroupNorm(_group_count(refinement_width), refinement_width),
            nn.GELU(),
        )
        self.trimap_decoder = nn.Sequential(
            _ResidualBlock(refinement_width),
            _ResidualBlock(refinement_width),
        )
        self.trimap_head = nn.Conv2d(refinement_width, 3, 1)

        nn.init.zeros_(self.mask_head.weight)
        nn.init.zeros_(self.mask_head.bias)

    @staticmethod
    def _resize(value: Tensor, size: tuple[int, int]) -> Tensor:
        if value.shape[-2:] == size:
            return value
        return F.interpolate(value, size=size, mode="bilinear", align_corners=False)

    def _decode_features(
        self,
        features: Tensor,
        *,
        sparse_prompt_embeddings: Tensor | None,
        dense_prompt_embeddings: Tensor | None,
    ) -> Tensor:
        if features.ndim != 4 or features.shape[1] != self.config.feature_channels:
            raise ValueError(
                "features must have shape [B, PDDConfig.feature_channels, h, w]"
            )
        decoded = self.feature_projection(features)
        if dense_prompt_embeddings is not None:
            if dense_prompt_embeddings.shape[1] != self.config.feature_channels:
                raise ValueError("dense SAM2 prompt embedding has an invalid channel count")
            dense = self._resize(dense_prompt_embeddings, decoded.shape[-2:])
            decoded = decoded + self.dense_prompt_projection(dense)

        if sparse_prompt_embeddings is not None:
            if sparse_prompt_embeddings.ndim != 3:
                raise ValueError("sparse SAM2 prompts must have shape [B,N,C]")
            spatial = decoded.flatten(2).transpose(1, 2)
            prompt = self.sparse_prompt_projection(sparse_prompt_embeddings)
            attended, _ = self.prompt_attention(spatial, prompt, prompt, need_weights=False)
            decoded = (spatial + attended).transpose(1, 2).reshape_as(decoded)
        return self.shared_decoder(decoded)

    def decode_mask(
        self,
        features: Tensor,
        seed_mask_logits: Tensor,
        *,
        sparse_prompt_embeddings: Tensor | None = None,
        dense_prompt_embeddings: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        decoded = self._decode_features(
            features,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )
        delta = self._resize(self.mask_head(decoded), seed_mask_logits.shape[-2:])
        mask = seed_mask_logits + self.config.mask_delta_scale * delta
        return mask, decoded

    def decode_trimap(
        self,
        features: Tensor,
        mask_logits: Tensor,
        *,
        sparse_prompt_embeddings: Tensor | None = None,
        dense_prompt_embeddings: Tensor | None = None,
        high_res_features: Sequence[Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        mask = mask_logits.detach() if self.config.detach_mask_pseudo_prompt else mask_logits
        decoded = self._decode_features(
            features,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )
        target_size = mask.shape[-2:]
        refined = self._resize(self.refinement_projection(decoded), target_size)
        refined = refined + self.mask_augmentation(mask.sigmoid())
        if high_res_features:
            detail = torch.zeros_like(refined)
            for feature in high_res_features:
                scalar_detail = feature.float().mean(dim=1, keepdim=True).to(refined.dtype)
                detail = detail + self._resize(
                    self.high_res_projection(scalar_detail), target_size
                )
            refined = refined + detail / len(high_res_features)
        trimap = self.trimap_head(self.trimap_decoder(refined))
        return trimap, decoded

    def forward(
        self,
        features: Tensor,
        seed_mask_logits: Tensor,
        *,
        sparse_prompt_embeddings: Tensor | None = None,
        dense_prompt_embeddings: Tensor | None = None,
        high_res_features: Sequence[Tensor] | None = None,
    ) -> PDDOutput:
        mask, decoded = self.decode_mask(
            features,
            seed_mask_logits,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )
        trimap, _ = self.decode_trimap(
            features,
            mask,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            high_res_features=high_res_features,
        )
        return PDDOutput(mask_logits=mask, trimap_logits=trimap, decoded_features=decoded)

