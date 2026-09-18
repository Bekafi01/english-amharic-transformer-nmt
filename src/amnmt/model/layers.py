"""Transformer building blocks, written from `torch.nn` primitives.

Every attention call goes through `F.scaled_dot_product_attention` so the same code gets the
fused kernels on GPU. Layers are Pre-LN: `x + Sublayer(LayerNorm(x))`.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)
        self.pe: Tensor
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model)

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        """x: [B, T, d] token embeddings -> scaled embeddings + positions [offset, offset+T)."""
        t = x.size(1)
        if offset + t > self.pe.size(0):
            raise ValueError(f"sequence length {offset + t} exceeds max_len {self.pe.size(0)}")
        return self.dropout(x * self.scale + self.pe[offset : offset + t])


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_p = dropout

    def _split(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, T, dh]

    def forward(self, query: Tensor, key_value: Tensor, mask: Tensor | None) -> Tensor:
        """query [B, T, d], key_value [B, S, d], mask [B|1, 1, T|1, S] bool (True = attend)."""
        q = self._split(self.q_proj(query))
        k = self._split(self.k_proj(key_value))
        v = self._split(self.v_proj(key_value))
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout_p if self.training else 0.0
        )
        b, _, t, _ = out.shape
        return self.out_proj(out.transpose(1, 2).reshape(b, t, -1))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float, activation: str) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = F.relu if activation == "relu" else F.gelu

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.dropout(self.act(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float, activation: str
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        h = self.norm1(x)
        x = x + self.dropout(self.self_attn(h, h, mask))
        return x + self.dropout(self.ff(self.norm2(x)))


class DecoderLayer(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float, activation: str
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: Tensor, memory: Tensor, self_mask: Tensor | None, memory_mask: Tensor | None
    ) -> Tensor:
        h = self.norm1(x)
        x = x + self.dropout(self.self_attn(h, h, self_mask))
        x = x + self.dropout(self.cross_attn(self.norm2(x), memory, memory_mask))
        return x + self.dropout(self.ff(self.norm3(x)))


class Encoder(nn.Module):
    def __init__(
        self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float, activation: str
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            EncoderLayer(d_model, n_heads, d_ff, dropout, activation) for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(
        self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float, activation: str
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            DecoderLayer(d_model, n_heads, d_ff, dropout, activation) for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, x: Tensor, memory: Tensor, self_mask: Tensor | None, memory_mask: Tensor | None
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, memory, self_mask, memory_mask)
        return self.norm(x)
