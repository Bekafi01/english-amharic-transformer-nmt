"""Attention masks. Convention everywhere in `amnmt.model`: **True = may attend**.

(This is the `F.scaled_dot_product_attention` convention; it is the *opposite* of
`nn.Transformer`'s key_padding_mask, where True means "ignore".)
"""

from __future__ import annotations

import torch
from torch import Tensor


def padding_mask(ids: Tensor, pad_id: int) -> Tensor:
    """[B, S] bool, True where the token is real (not padding)."""
    return ids != pad_id


def causal_mask(size: int, device: torch.device | None = None) -> Tensor:
    """[T, T] bool, True at (i, j) when j <= i."""
    return torch.ones(size, size, dtype=torch.bool, device=device).tril()


def attention_mask(key_mask: Tensor | None, causal: Tensor | None = None) -> Tensor | None:
    """Combine a [B, S] key mask and an optional [T, S] causal mask into [B, 1, T, S] for SDPA.

    Returns None when neither is given so SDPA can take its fastest path.
    """
    if key_mask is None and causal is None:
        return None
    mask: Tensor | None = None
    if key_mask is not None:
        mask = key_mask[:, None, None, :]  # [B, 1, 1, S]
    if causal is not None:
        c = causal[None, None, :, :]  # [1, 1, T, S]
        mask = c if mask is None else mask & c
    return mask
