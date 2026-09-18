"""Label-smoothed cross-entropy over non-pad target tokens."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


class LabelSmoothedCrossEntropy:
    def __init__(self, pad_id: int, smoothing: float) -> None:
        self.pad_id = pad_id
        self.smoothing = smoothing

    def __call__(self, logits: Tensor, target: Tensor) -> tuple[Tensor, Tensor, int]:
        """logits [B, T, V], target [B, T] -> (smoothed loss sum, nll sum, n_tokens).

        Sums (not means) so gradient accumulation and multi-batch eval can normalise exactly.
        """
        logits = logits.reshape(-1, logits.size(-1)).float()
        target = target.reshape(-1)
        mask = target != self.pad_id
        n_tokens = int(mask.sum())
        logp = F.log_softmax(logits[mask], dim=-1)
        tgt = target[mask]
        nll = -logp.gather(1, tgt[:, None]).squeeze(1)
        if self.smoothing > 0:
            smooth = -logp.mean(dim=-1)
            loss = (1 - self.smoothing) * nll + self.smoothing * smooth
        else:
            loss = nll
        return loss.sum(), nll.sum().detach(), n_tokens

    @staticmethod
    def perplexity(nll_sum: float, n_tokens: int) -> float:
        return float(torch.exp(torch.tensor(nll_sum / max(n_tokens, 1))))
