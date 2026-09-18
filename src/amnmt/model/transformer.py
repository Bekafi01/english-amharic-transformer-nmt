"""Seq2Seq Transformer with a shared vocabulary and 3-way weight tying
(source embedding = target embedding = output projection).

Sequence conventions (built by the training/inference layers, not here):
    source: <2xx> tok tok ... </s>          (target-language tag first)
    target in : <s>  tok tok ...
    target out:      tok tok ... </s>
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from amnmt.core.config import ModelConfig
from amnmt.model.layers import Decoder, Encoder, SinusoidalPositionalEncoding
from amnmt.model.masks import attention_mask, causal_mask, padding_mask


class Seq2SeqTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig, vocab_size: int, pad_id: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab_size, cfg.d_model, padding_idx=pad_id)
        self.pos = SinusoidalPositionalEncoding(cfg.d_model, cfg.max_len, cfg.dropout)
        args = (cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout, cfg.activation)
        self.encoder = Encoder(cfg.n_encoder_layers, *args)
        self.decoder = Decoder(cfg.n_decoder_layers, *args)
        self._init_weights()

    def _init_weights(self) -> None:
        # Tied embedding/output: keep logits O(1) at init with a d_model^-0.5 scale.
        nn.init.normal_(self.embed.weight, mean=0.0, std=self.cfg.d_model**-0.5)
        with torch.no_grad():
            self.embed.weight[self.pad_id].zero_()
        for name, p in self.named_parameters():
            if p.dim() > 1 and not name.startswith("embed."):
                nn.init.xavier_uniform_(p)

    # ------------------------------------------------------------------ pieces

    def encode(self, src: Tensor) -> tuple[Tensor, Tensor]:
        """src [B, S] ids -> (memory [B, S, d], src_mask [B, S] True=real)."""
        src_mask = padding_mask(src, self.pad_id)
        x = self.pos(self.embed(src))
        return self.encoder(x, attention_mask(src_mask)), src_mask

    def decode(self, tgt_in: Tensor, memory: Tensor, src_mask: Tensor) -> Tensor:
        """tgt_in [B, T] ids, memory [B, S, d] -> logits [B, T, V] (full causal pass)."""
        t = tgt_in.size(1)
        self_mask = attention_mask(
            padding_mask(tgt_in, self.pad_id), causal_mask(t, device=tgt_in.device)
        )
        x = self.pos(self.embed(tgt_in))
        h = self.decoder(x, memory, self_mask, attention_mask(src_mask))
        return self.output(h)

    def output(self, h: Tensor) -> Tensor:
        return F.linear(h, self.embed.weight)  # tied projection, no bias

    def forward(self, src: Tensor, tgt_in: Tensor) -> Tensor:
        memory, src_mask = self.encode(src)
        return self.decode(tgt_in, memory, src_mask)

    # ------------------------------------------------------------------ info

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)
