from __future__ import annotations

from collections.abc import Iterable

import torch.nn.functional as F
from torch import Tensor, nn


class LoRALinear(nn.Module):
    """Small LoRA adapter installed after the official SAM2 checkpoint loads."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        self.base.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        base = self.base(x)
        update = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return base + self.scaling * update


def inject_lora(
    root: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    target_patterns: Iterable[str] = ("attn.qkv", "attn.proj"),
) -> list[str]:
    """Replace matching linear layers and return their fully qualified names."""

    if rank <= 0:
        return []
    patterns = tuple(target_patterns)
    matches: list[tuple[str, nn.Linear]] = []
    for name, module in root.named_modules():
        if isinstance(module, nn.Linear) and any(pattern in name for pattern in patterns):
            matches.append((name, module))

    installed: list[str] = []
    for name, module in matches:
        parent_name, _, child_name = name.rpartition(".")
        parent = root.get_submodule(parent_name) if parent_name else root
        setattr(parent, child_name, LoRALinear(module, rank, alpha, dropout))
        installed.append(name)
    if not installed:
        raise ValueError(
            "no SAM2 image-encoder Linear matched lora_target_patterns=" + repr(patterns)
        )
    return installed


def lora_state_dict(module: nn.Module) -> dict[str, Tensor]:
    return {
        name: value
        for name, value in module.state_dict().items()
        if name.endswith("lora_A") or name.endswith("lora_B")
    }
