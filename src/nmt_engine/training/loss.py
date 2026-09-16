"""Loss functions with label smoothing and padding exclusion for Neural Machine Translation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    """Cross-Entropy loss with label smoothing regularization (Szegedy et al., 2016).

    Replaces one-hot target distributions with smoothed distributions:
        q(k) = (1 - eps) * I(k == y) + eps / (K - 1)
    Masks out padding tokens so they do not contribute to gradients or loss values.

    Args:
        vocab_size: Total vocabulary size K.
        pad_id: Padding token index to mask out.
        smoothing: Label smoothing regularization parameter eps (default: 0.1).
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        pad_id: int = 0,
        smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        assert 0.0 <= smoothing < 1.0, f"smoothing must be in [0, 1), got {smoothing}"
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes label-smoothed cross-entropy loss over unpadded target tokens.

        Args:
            logits: Predicted unnormalized scores of shape (B, L, V) or (N, V).
            target: Ground truth token indices of shape (B, L) or (N,).

        Returns:
            Scalar tensor representing the mean loss over non-pad tokens.
        """
        # Ensure flat shape: (N, V) and (N,)
        if logits.dim() == 3:
            logits = logits.view(-1, self.vocab_size)
            target = target.contiguous().view(-1)

        log_probs = F.log_softmax(logits, dim=-1)

        # Filter out padding tokens
        non_pad_mask = target != self.pad_id
        active_log_probs = log_probs[non_pad_mask]
        active_target = target[non_pad_mask]

        if active_target.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Negative log-likelihood of true class
        nll_loss = -active_log_probs.gather(dim=-1, index=active_target.unsqueeze(1)).squeeze(1)

        # Uniform smooth loss: average log probability across all vocabulary classes
        smooth_loss = -active_log_probs.mean(dim=-1)

        # Convex combination of hard NLL and smooth uniform loss
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
