from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Sequence

from torch import Tensor, nn

from .config import PDDConfig
from .pdd import PromptableDualModeDecoder


@dataclass
class MSSOutput:
    mask_logits: Tensor
    trimap_logits: Tensor
    non_memory_features: Tensor
    memory_decoded_features: Tensor
    non_memory_decoded_features: Tensor


class MemorySeparableSiamese(nn.Module):
    """Shared PDD: memory features for mask, clean features for trimap."""

    def __init__(
        self,
        pdd: PromptableDualModeDecoder | None = None,
        config: PDDConfig | None = None,
    ) -> None:
        super().__init__()
        if pdd is not None and config is not None:
            raise ValueError("pass either pdd or config, not both")
        self.pdd = pdd or PromptableDualModeDecoder(config)

    def forward(
        self,
        memory_features: Tensor,
        non_memory_features: Tensor,
        seed_mask_logits: Tensor,
        *,
        mask_sparse_prompt_embeddings: Tensor | None = None,
        mask_dense_prompt_embeddings: Tensor | None = None,
        trimap_sparse_prompt_embeddings: Tensor | None = None,
        trimap_dense_prompt_embeddings: Tensor | None = None,
        trimap_prompt_encoder: Callable[[Tensor], tuple[Tensor, Tensor]] | None = None,
        high_res_features: Sequence[Tensor] | None = None,
    ) -> MSSOutput:
        if memory_features.shape != non_memory_features.shape:
            raise ValueError("memory and non-memory features must have identical shapes")
        mask, memory_decoded = self.pdd.decode_mask(
            memory_features,
            seed_mask_logits,
            sparse_prompt_embeddings=mask_sparse_prompt_embeddings,
            dense_prompt_embeddings=mask_dense_prompt_embeddings,
        )
        if trimap_prompt_encoder is not None:
            pseudo_mask = (
                mask.detach() if self.pdd.config.detach_mask_pseudo_prompt else mask
            )
            trimap_sparse_prompt_embeddings, trimap_dense_prompt_embeddings = (
                trimap_prompt_encoder(pseudo_mask)
            )
        trimap, non_memory_decoded = self.pdd.decode_trimap(
            non_memory_features,
            mask,
            sparse_prompt_embeddings=trimap_sparse_prompt_embeddings,
            dense_prompt_embeddings=trimap_dense_prompt_embeddings,
            high_res_features=high_res_features,
        )
        return MSSOutput(
            mask_logits=mask,
            trimap_logits=trimap,
            non_memory_features=non_memory_features,
            memory_decoded_features=memory_decoded,
            non_memory_decoded_features=non_memory_decoded,
        )
