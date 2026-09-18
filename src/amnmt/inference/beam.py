"""Batched beam search over the model's `encode`/`decode` API.

No KV cache: each step re-runs the decoder on the full prefix. Simple and correct; fast enough for
FLORES-sized evaluation. Length normalisation follows GNMT: score / ((5 + len) / 6) ** alpha.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor

from amnmt.model.transformer import Seq2SeqTransformer


@dataclass
class BeamConfig:
    beam_size: int = 4
    length_penalty: float = 1.0
    max_len_a: float = 1.5  # max output length = a * src_len + b, capped by the model's max_len
    max_len_b: int = 10
    forbidden_ids: tuple[int, ...] = field(default_factory=tuple)  # never generated (pad/bos/tags)


@dataclass
class Hypothesis:
    tokens: list[int]  # without <s>, without </s>
    score: float  # length-normalised log-prob


def _length_penalty(length: int, alpha: float) -> float:
    return ((5.0 + length) / 6.0) ** alpha


@torch.no_grad()
def beam_search(
    model: Seq2SeqTransformer, src: Tensor, bos_id: int, eos_id: int, cfg: BeamConfig
) -> list[Hypothesis]:
    """src [B, S] -> best hypothesis per batch item."""
    model.eval()
    device = src.device
    b, k = src.size(0), cfg.beam_size
    src_len = int((src != model.pad_id).sum(1).max())
    max_len = min(model.cfg.max_len - 2, int(cfg.max_len_a * src_len + cfg.max_len_b))

    memory, src_mask = model.encode(src)
    memory = memory.repeat_interleave(k, dim=0)  # [B*K, S, d]
    src_mask = src_mask.repeat_interleave(k, dim=0)

    seqs = torch.full((b * k, 1), bos_id, dtype=torch.long, device=device)
    # Only beam 0 is alive at t=0, otherwise the K beams would be identical copies.
    scores = torch.zeros(b, k, device=device)
    scores[:, 1:] = float("-inf")
    finished: list[list[Hypothesis]] = [[] for _ in range(b)]
    done = torch.zeros(b, dtype=torch.bool, device=device)
    forbid = torch.tensor(cfg.forbidden_ids, dtype=torch.long, device=device)

    for step in range(max_len):
        logits = model.decode(seqs, memory, src_mask)[:, -1].float()
        if forbid.numel():
            logits[:, forbid] = float("-inf")
        logp = F.log_softmax(logits, dim=-1)  # [B*K, V]
        vocab = logp.size(-1)
        cand = (scores.view(-1, 1) + logp).view(b, k * vocab)
        top_scores, top_idx = cand.topk(2 * k, dim=-1)
        beam_idx = top_idx // vocab  # [B, 2K]
        tok_idx = top_idx % vocab

        next_beams = torch.zeros(b, k, dtype=torch.long, device=device)
        next_tokens = torch.zeros(b, k, dtype=torch.long, device=device)
        next_scores = torch.full((b, k), float("-inf"), device=device)
        is_eos = (tok_idx == eos_id).tolist()
        top_scores_l = top_scores.tolist()
        for bi in range(b):
            if done[bi]:
                next_tokens[bi] = model.pad_id
                continue
            slot = 0
            for ci in range(2 * k):
                sc = top_scores_l[bi][ci]
                if sc == float("-inf"):
                    break
                if is_eos[bi][ci]:
                    # An EOS candidate only counts if it ranks within the top-K; otherwise a
                    # worse hypothesis could close the beam ahead of a better alive one.
                    if ci < k and len(finished[bi]) < k:
                        src_row = seqs[bi * k + int(beam_idx[bi, ci])]
                        finished[bi].append(
                            Hypothesis(
                                src_row[1:].tolist(),
                                sc / _length_penalty(step + 1, cfg.length_penalty),
                            )
                        )
                    continue
                if slot < k:
                    next_beams[bi, slot] = beam_idx[bi, ci]
                    next_tokens[bi, slot] = tok_idx[bi, ci]
                    next_scores[bi, slot] = sc
                    slot += 1
            if len(finished[bi]) >= k:
                done[bi] = True
        if bool(done.all()):
            break
        flat = (torch.arange(b, device=device)[:, None] * k + next_beams).view(-1)
        seqs = torch.cat([seqs[flat], next_tokens.view(-1, 1)], dim=1)
        scores = next_scores

    # Items that never produced K finished hypotheses: close their best alive beams.
    for bi in range(b):
        if len(finished[bi]) < k:
            alive = scores[bi].tolist()
            for j in range(k):
                if alive[j] != float("-inf"):
                    finished[bi].append(
                        Hypothesis(
                            seqs[bi * k + j, 1:].tolist(),
                            alive[j] / _length_penalty(seqs.size(1) - 1, cfg.length_penalty),
                        )
                    )
        if not finished[bi]:
            finished[bi].append(Hypothesis([], float("-inf")))
    return [max(h, key=lambda x: x.score) for h in finished]
