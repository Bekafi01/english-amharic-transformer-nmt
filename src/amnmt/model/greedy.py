"""Greedy decoding on raw tensors. Lives in the model layer so the trainer can log samples;
beam search (with its tokenizer-aware plumbing) is in `amnmt.inference`.
"""

from __future__ import annotations

import torch
from torch import Tensor

from amnmt.model.transformer import Seq2SeqTransformer


@torch.no_grad()
def greedy_decode(
    model: Seq2SeqTransformer, src: Tensor, bos_id: int, eos_id: int, max_len: int
) -> Tensor:
    """src [B, S] -> generated ids [B, <=max_len] after <s>; pad_id fills positions after </s>."""
    model.eval()
    memory, src_mask = model.encode(src)
    b = src.size(0)
    ys = torch.full((b, 1), bos_id, dtype=torch.long, device=src.device)
    finished = torch.zeros(b, dtype=torch.bool, device=src.device)
    for _ in range(max_len):
        logits = model.decode(ys, memory, src_mask)[:, -1]
        next_tok = logits.argmax(-1)
        next_tok = torch.where(finished, torch.full_like(next_tok, model.pad_id), next_tok)
        ys = torch.cat([ys, next_tok[:, None]], dim=1)
        finished |= next_tok == eos_id
        if bool(finished.all()):
            break
    return ys[:, 1:]
