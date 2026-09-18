"""Inverse-square-root learning-rate schedule with linear warmup ("Noam", peak-LR form)."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def inverse_sqrt_factor(step: int, warmup_steps: int) -> float:
    """Multiplier on the peak LR: linear ramp to 1.0 at `warmup_steps`, then sqrt(warmup / step)."""
    step = max(step, 1)
    if step < warmup_steps:
        return step / warmup_steps
    return math.sqrt(warmup_steps / step)


def build_scheduler(optimizer: Optimizer, warmup_steps: int) -> LambdaLR:
    return LambdaLR(optimizer, lambda s: inverse_sqrt_factor(s, warmup_steps))
