"""Custom Sequence-to-Sequence Transformer architecture with 3-Way Weight Tying and Pre-LN topology."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as defined in Vaswani et al. (2017)."""

    def __init__(self, d_model: int = 512, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, L, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism with scaled dot-product attention."""

    def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = q.size(0)

        # 1. Linear projections and multi-head split: (B, n_heads, L, d_k)
        Q = self.w_q(q).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(k).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(v).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 2. Scaled Dot-Product: (B, n_heads, L_q, L_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 3. Apply mask if present
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 4. Context aggregation: (B, n_heads, L_q, d_k)
        context = torch.matmul(attn_weights, V)

        # 5. Concatenate heads and project output: (B, L_q, d_model)
        context = context.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.w_o(context)


class PositionwiseFeedForward(nn.Module):
    """Two-layer feed-forward network with expansion factor 4 and configurable activation."""

    def __init__(
        self,
        d_model: int = 512,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.activation_fn = F.gelu if activation == "gelu" else F.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation_fn(self.linear1(x))))


class EncoderLayer(nn.Module):
    """Pre-LN Transformer Encoder Layer with self-attention and positionwise FFN."""

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout=dropout, activation=activation)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # Pre-LN: LayerNorm before self-attention sublayer
        h = self.norm1(x)
        x = x + self.dropout(self.self_attn(h, h, h, mask=src_mask))
        # Pre-LN: LayerNorm before FFN sublayer
        h = self.norm2(x)
        x = x + self.dropout(self.ffn(h))
        return x


class Encoder(nn.Module):
    """Stack of N Pre-LN Transformer Encoder layers with terminal LayerNorm."""

    def __init__(
        self,
        n_layers: int = 6,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, n_heads, d_ff, dropout=dropout, activation=activation)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """Pre-LN Transformer Decoder Layer with causal self-attention, cross-attention, and FFN."""

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout=dropout, activation=activation)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        # 1. Masked Causal Self-Attention (Pre-LN)
        h = self.norm1(x)
        x = x + self.dropout(self.self_attn(h, h, h, mask=tgt_mask))

        # 2. Encoder-Decoder Cross-Attention (Pre-LN)
        h = self.norm2(x)
        x = x + self.dropout(self.cross_attn(h, memory, memory, mask=src_mask))

        # 3. Positionwise Feed-Forward (Pre-LN)
        h = self.norm3(x)
        x = x + self.dropout(self.ffn(h))
        return x


class Decoder(nn.Module):
    """Stack of N Pre-LN Transformer Decoder layers with terminal LayerNorm."""

    def __init__(
        self,
        n_layers: int = 6,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, n_heads, d_ff, dropout=dropout, activation=activation)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, src_mask)
        return self.norm(x)


class Seq2SeqTransformer(nn.Module):
    """Complete Sequence-to-Sequence Transformer with 3-Way Weight Tying ($E_{src} = E_{tgt} = W_{proj}^T$)."""

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 512,
        pad_id: int = 0,
        tie_weights: bool = True,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.vocab_size = vocab_size

        # 1. Tied Shared Embedding Table
        self.shared_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)

        # 2. Encoder and Decoder Stacks (Pre-LN)
        self.encoder = Encoder(
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
        )
        self.decoder = Decoder(
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
        )

        # 3. Target Output Projection Layer
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

        # 4. 3-Way Weight Tying Assertion
        if tie_weights:
            self.output_proj.weight = self.shared_embedding.weight

        # Weight Initialization
        self._init_parameters()

    def _init_parameters(self) -> None:
        """Initialize parameters with Xavier Uniform for linear weights and normal for embeddings."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """Encode source sequence into contextual memory representations."""
        x = self.shared_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        return self.encoder(x, src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decode target prefix given encoder memory."""
        x = self.shared_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        return self.decoder(x, memory, tgt_mask, src_mask)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through complete Seq2Seq Transformer.

        Returns:
            logits: (B, L_tgt, vocab_size) unnormalized token prediction scores
        """
        memory = self.encode(src, src_mask)
        dec_out = self.decode(tgt, memory, tgt_mask, src_mask)
        return self.output_proj(dec_out)
