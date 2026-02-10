from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class BackboneOutput:
    global_embedding: torch.Tensor
    patch_tokens: torch.Tensor
    grid_size: Tuple[int, int]
    cls_token: Optional[torch.Tensor] = None
    register_tokens: Optional[torch.Tensor] = None


class BackboneAdapter(nn.Module):
    """Adapter interface that standardizes backbone outputs."""

    def __init__(self, name: str, variant: str) -> None:
        super().__init__()
        self.name = name
        self.variant = variant
        self.embed_dim: int = 0
        self.patch_size: int = 0
        self.num_register_tokens: int = 0
        self.model: nn.Module | None = None

    def forward(self, x: torch.Tensor) -> BackboneOutput:  # pragma: no cover - abstract
        raise NotImplementedError


def infer_patch_size(model: nn.Module, fallback: int) -> int:
    patch_size = getattr(model, "patch_size", None)
    if patch_size is None and hasattr(model, "patch_embed"):
        patch_size = getattr(model.patch_embed, "patch_size", None)
    if isinstance(patch_size, (tuple, list)):
        patch_size = patch_size[0]
    if isinstance(patch_size, int) and patch_size > 0:
        return patch_size
    return fallback


def infer_num_register_tokens(model: nn.Module) -> int:
    val = getattr(model, "num_register_tokens", None)
    if isinstance(val, int):
        return val
    reg = getattr(model, "register_tokens", None)
    if hasattr(reg, "shape") and len(reg.shape) >= 2:
        return int(reg.shape[1])
    return 0


def infer_grid_size(x: torch.Tensor, patch_size: int) -> Tuple[int, int]:
    if patch_size <= 0:
        raise ValueError("patch_size must be a positive integer")
    return x.shape[-2] // patch_size, x.shape[-1] // patch_size


def split_special_tokens(
    tokens: torch.Tensor,
    num_patches: int,
    num_register_tokens: int | None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Split [CLS][REG*][PATCH*] tokens into components."""
    if tokens.dim() != 3:
        raise ValueError(f"Expected tokens to be 3D, got shape {tuple(tokens.shape)}")
    total = tokens.shape[1]
    if total == num_patches:
        return tokens, None, None

    if num_register_tokens is None:
        if total == num_patches + 1:
            num_register_tokens = 0
        elif total > num_patches + 1:
            num_register_tokens = total - num_patches - 1
        else:
            raise ValueError(
                f"Token count {total} is smaller than expected patch tokens {num_patches}"
            )

    cls_token = tokens[:, :1]
    reg_end = 1 + max(0, int(num_register_tokens))
    register_tokens = tokens[:, 1:reg_end] if reg_end > 1 else None
    patch_tokens = tokens[:, reg_end : reg_end + num_patches]
    if patch_tokens.shape[1] != num_patches:
        raise ValueError(
            f"Patch token count mismatch: expected {num_patches}, got {patch_tokens.shape[1]}"
        )
    return patch_tokens, cls_token, register_tokens


def patch_tokens_to_grid(output: BackboneOutput) -> torch.Tensor:
    """Reshape patch tokens into (B, D, H, W) feature map."""
    bsz, tokens, dim = output.patch_tokens.shape
    grid_h, grid_w = output.grid_size
    if tokens != grid_h * grid_w:
        raise ValueError(
            f"Patch token count {tokens} does not match grid {output.grid_size}."
        )
    return output.patch_tokens.transpose(1, 2).reshape(bsz, dim, grid_h, grid_w)
