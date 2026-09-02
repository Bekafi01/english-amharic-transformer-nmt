"""Reproducibility and random seed utilities."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = True) -> int:
    """Sets random seeds across Python, NumPy, and PyTorch for reproducible runs.

    Args:
        seed: Integer seed value.
        deterministic: If True, sets PyTorch/cuDNN flags for strict determinism.

    Returns:
        The seed integer set.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Avoid warnings on platforms where set_deterministic is supported
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass

    return seed
