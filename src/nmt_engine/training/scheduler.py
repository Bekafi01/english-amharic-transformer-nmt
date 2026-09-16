"""Learning rate schedulers for Transformer model optimization."""

from typing import Any

import torch.optim as optim


class NoamScheduler:
    """Noam Learning Rate Scheduler (Vaswani et al., 2017).

    The learning rate increases linearly for the first `warmup_steps` training steps,
    and subsequently decreases proportionally to the inverse square root of the step number:

        lr = factor * (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))

    Args:
        optimizer: Wrapped PyTorch optimizer whose learning rate will be adjusted.
        d_model: Hidden dimension size of the Transformer model (default: 512).
        warmup_steps: Number of linear warmup steps (default: 4000).
        factor: Global learning rate multiplier scale factor (default: 1.0).
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        d_model: int = 512,
        warmup_steps: int = 4000,
        factor: float = 1.0,
    ) -> None:
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.step_num = 0

    def step(self) -> float:
        """Advance the scheduler by one step, update optimizer learning rates, and return current lr."""
        self.step_num += 1
        lr = self.calculate_lr(self.step_num)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def calculate_lr(self, step: int) -> float:
        """Compute the learning rate at a given step index."""
        step = max(1, step)
        return (
            self.factor
            * (self.d_model**-0.5)
            * min(
                step**-0.5,
                step * (self.warmup_steps**-1.5),
            )
        )

    def get_lr(self) -> float:
        """Return the current learning rate from the optimizer's first parameter group."""
        return self.optimizer.param_groups[0]["lr"]

    def state_dict(self) -> dict[str, Any]:
        """Serialize scheduler state for checkpointing."""
        return {
            "step_num": self.step_num,
            "d_model": self.d_model,
            "warmup_steps": self.warmup_steps,
            "factor": self.factor,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore scheduler state from a checkpoint dictionary."""
        self.step_num = state_dict.get("step_num", 0)
        self.d_model = state_dict.get("d_model", self.d_model)
        self.warmup_steps = state_dict.get("warmup_steps", self.warmup_steps)
        self.factor = state_dict.get("factor", self.factor)
        # Update optimizer with current step's lr
        current_lr = self.calculate_lr(self.step_num)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr
