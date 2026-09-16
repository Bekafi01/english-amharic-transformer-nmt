"""Autoregressive decoding algorithms (Greedy and Beam Search) with penalty adjustments."""

import torch
import torch.nn.functional as F

from src.nmt_engine.models.transformer import Seq2SeqTransformer


@torch.no_grad()
def greedy_decode(
    model: Seq2SeqTransformer,
    src_ids: list[int],
    max_len: int = 128,
    pad_id: int = 0,
    bos_id: int = 2,
    eos_id: int = 3,
    device: torch.device | None = None,
) -> list[int]:
    """Generates target token sequence autoregressively using greedy argmax selection.

    Args:
        model: Trained Seq2SeqTransformer model instance.
        src_ids: Source token IDs including direction prefix and delimiters.
        max_len: Maximum target sequence generation limit.
        pad_id: Padding token index (default: 0).
        bos_id: Beginning-of-sequence token index (default: 2).
        eos_id: End-of-sequence token index (default: 3).
        device: PyTorch compute device. If None, inferred from model parameters.

    Returns:
        List of generated token IDs (including bos_id and eos_id if generated).
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_mask = (src != pad_id).unsqueeze(1).unsqueeze(2)
    memory = model.encode(src, src_mask)

    ys = torch.tensor([[bos_id]], dtype=torch.long, device=device)

    for _ in range(max_len - 1):
        l_t = ys.size(1)
        tgt_mask = (
            torch.tril(torch.ones((l_t, l_t), dtype=torch.bool, device=device))
            .unsqueeze(0)
            .unsqueeze(0)
        )
        out = model.decode(ys, memory, tgt_mask, src_mask)
        logits = model.output_proj(out[:, -1])
        next_token = torch.argmax(logits, dim=-1).item()

        ys = torch.cat([ys, torch.tensor([[next_token]], dtype=torch.long, device=device)], dim=1)
        if next_token == eos_id:
            break

    return ys.squeeze(0).tolist()


@torch.no_grad()
def beam_search_decode(
    model: Seq2SeqTransformer,
    src_ids: list[int],
    beam_size: int = 4,
    max_len: int = 128,
    length_penalty: float = 0.6,
    repetition_penalty: float = 1.2,
    pad_id: int = 0,
    bos_id: int = 2,
    eos_id: int = 3,
    device: torch.device | None = None,
) -> list[int]:
    """Generates target token sequence using Beam Search with length normalization and repetition penalties.

    Length Normalization:
        Score(Y) = CumulativeLogProb / ((5 + len(Y)) / 6) ** alpha

    Repetition Penalty (Keskar et al., 2019):
        logits[t] = logits[t] / theta if logits[t] > 0 else logits[t] * theta

    Args:
        model: Trained Seq2SeqTransformer model.
        src_ids: Source token IDs.
        beam_size: Number of concurrent hypothesis beams (k).
        max_len: Maximum decoded sequence length cutoff.
        length_penalty: Alpha parameter for length normalization (default: 0.6).
        repetition_penalty: Multiplier penalizing repeated subwords (default: 1.2, 1.0 disables).
        pad_id: Padding token ID.
        bos_id: Beginning-of-sequence token ID.
        eos_id: End-of-sequence token ID.
        device: PyTorch device.

    Returns:
        List of generated token IDs for the highest-scoring beam hypothesis.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_mask = (src != pad_id).unsqueeze(1).unsqueeze(2)
    memory = model.encode(src, src_mask)

    # Hypotheses: list of (token_list, cumulative_log_prob, is_finished)
    beams: list[tuple[list[int], float, bool]] = [([bos_id], 0.0, False)]

    for _ in range(max_len - 1):
        if all(finished for _, _, finished in beams):
            break

        active_beams = [(seq, score) for seq, score, fin in beams if not fin]
        finished_beams = [b for b in beams if b[2]]

        candidates: list[tuple[list[int], float, bool]] = list(finished_beams)

        if active_beams:
            ys = torch.tensor([seq for seq, _ in active_beams], dtype=torch.long, device=device)
            l_t = ys.size(1)
            tgt_mask = (
                torch.tril(torch.ones((l_t, l_t), dtype=torch.bool, device=device))
                .unsqueeze(0)
                .unsqueeze(0)
            )
            n_active = len(active_beams)
            mem_exp = memory.expand(n_active, -1, -1)
            src_mask_exp = src_mask.expand(n_active, -1, -1, -1)

            out = model.decode(ys, mem_exp, tgt_mask, src_mask_exp)
            logits = model.output_proj(out[:, -1])  # Shape: (n_active, vocab_size)

            for b_idx, (seq, score) in enumerate(active_beams):
                b_logits = logits[b_idx].clone()
                if repetition_penalty != 1.0:
                    for token_id in set(seq):
                        if b_logits[token_id] > 0:
                            b_logits[token_id] /= repetition_penalty
                        else:
                            b_logits[token_id] *= repetition_penalty

                log_probs = F.log_softmax(b_logits, dim=-1)
                topk_log_probs, topk_indices = torch.topk(log_probs, k=beam_size)

                for lp, idx in zip(topk_log_probs.tolist(), topk_indices.tolist(), strict=True):
                    new_seq = seq + [idx]
                    new_score = score + lp
                    is_done = idx == eos_id
                    candidates.append((new_seq, new_score, is_done))

        # Length normalization ranking function
        def rank_score(cand: tuple[list[int], float, bool]) -> float:
            c_seq, c_score, _ = cand
            norm = ((5 + len(c_seq)) / 6.0) ** length_penalty
            return c_score / max(1e-8, norm)

        candidates.sort(key=rank_score, reverse=True)
        beams = candidates[:beam_size]

    return beams[0][0]
