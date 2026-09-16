"""Production training and validation manager for Seq2Seq Transformer."""

import logging
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.nmt_engine.models.transformer import Seq2SeqTransformer
from src.nmt_engine.utils.logging import get_logger


class TransformerTrainer:
    """Production training manager for Sequence-to-Sequence Transformer models.

    Integrates:
      - Automatic Mixed Precision (AMP) with modern PyTorch 2.x `torch.amp.GradScaler`
      - Gradient accumulation for larger effective batch sizes
      - Gradient norm clipping to prevent attention explosion
      - Length-normalized perplexity computation
      - Best and latest checkpoint persistence with early stopping

    Args:
        model: Seq2SeqTransformer neural architecture instance.
        train_loader: DataLoader supplying training batches (src, tgt, src_mask, tgt_mask).
        val_loader: DataLoader supplying validation batches.
        criterion: Loss function module (e.g. LabelSmoothingLoss).
        optimizer: PyTorch optimizer instance.
        scheduler: Learning rate scheduler (e.g. NoamScheduler) or None.
        device: PyTorch compute device (CUDA, MPS, or CPU).
        grad_accum_steps: Number of micro-batches to accumulate before weight updates.
        clip_grad_norm: Maximum gradient norm for clipping (0 to disable).
        checkpoint_dir: Path to directory where checkpoints are saved.
        mixed_precision: Mixed precision mode ('fp16', 'bf16', or 'no').
        logger: Optional logging.Logger instance.
    """

    def __init__(
        self,
        model: Seq2SeqTransformer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Any | None = None,
        device: torch.device | None = None,
        grad_accum_steps: int = 2,
        clip_grad_norm: float = 1.0,
        checkpoint_dir: str | Path = Path("./checkpoints"),
        mixed_precision: str = "fp16",
        logger: logging.Logger | None = None,
    ) -> None:
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.grad_accum_steps = max(1, grad_accum_steps)
        self.clip_grad_norm = clip_grad_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or get_logger("nmt_engine.trainer")

        # AMP Configuration (Modern torch.amp API)
        self.mixed_precision = mixed_precision.lower()
        self.use_amp = self.device.type == "cuda" and self.mixed_precision in ("fp16", "bf16")
        self.amp_dtype = torch.bfloat16 if self.mixed_precision == "bf16" else torch.float16

        # torch.amp.GradScaler handles device_type in PyTorch 2.1+
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=(self.use_amp and self.mixed_precision == "fp16")
        )

        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.global_step = 0

    def train_epoch(self, epoch: int) -> float:
        """Executes a single training epoch across all train_loader batches.

        Args:
            epoch: Current 1-indexed epoch number.

        Returns:
            Mean unaccumulated training loss across all batches in the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        self.optimizer.zero_grad()

        for step, (src, tgt, src_mask, tgt_mask) in enumerate(self.train_loader, 1):
            self.global_step += 1
            src = src.to(self.device, non_blocking=True)
            tgt = tgt.to(self.device, non_blocking=True)
            src_mask = src_mask.to(self.device, non_blocking=True)
            tgt_mask = tgt_mask.to(self.device, non_blocking=True)

            # Teacher forcing alignment:
            #   Input: tokens 0 to L-1 (e.g., <bos>, t_1, ..., t_{L-1})
            #   Target: tokens 1 to L   (e.g., t_1, ..., t_{L-1}, <eos>)
            tgt_input = tgt[:, :-1]
            tgt_target = tgt[:, 1:]
            tgt_input_mask = tgt_mask[:, :, :-1, :-1]

            with torch.amp.autocast(
                device_type=self.device.type, enabled=self.use_amp, dtype=self.amp_dtype
            ):
                logits = self.model(src, tgt_input, src_mask, tgt_input_mask)
                loss = self.criterion(logits, tgt_target) / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            # Parameter update on accumulation boundary or epoch end
            if step % self.grad_accum_steps == 0 or step == num_batches:
                self.scaler.unscale_(self.optimizer)
                if self.clip_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                if self.scheduler is not None:
                    self.scheduler.step()

            batch_loss = loss.item() * self.grad_accum_steps
            total_loss += batch_loss

            if step % 200 == 0 or step == 1 or step == num_batches:
                curr_lr = (
                    self.scheduler.get_lr()
                    if hasattr(self.scheduler, "get_lr")
                    else self.optimizer.param_groups[0]["lr"]
                )
                self.logger.info(
                    f"Epoch {epoch:02d} | Step {step:05d}/{num_batches:05d} "
                    f"| Loss: {batch_loss:.4f} | LR: {curr_lr:.6f}"
                )

        return total_loss / max(1, num_batches)

    @torch.no_grad()
    def evaluate(self) -> tuple[float, float]:
        """Evaluates model performance over the validation dataset.

        Returns:
            Tuple of (mean validation loss, validation perplexity).
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)

        for src, tgt, src_mask, tgt_mask in self.val_loader:
            src = src.to(self.device, non_blocking=True)
            tgt = tgt.to(self.device, non_blocking=True)
            src_mask = src_mask.to(self.device, non_blocking=True)
            tgt_mask = tgt_mask.to(self.device, non_blocking=True)

            tgt_input = tgt[:, :-1]
            tgt_target = tgt[:, 1:]
            tgt_input_mask = tgt_mask[:, :, :-1, :-1]

            with torch.amp.autocast(
                device_type=self.device.type, enabled=self.use_amp, dtype=self.amp_dtype
            ):
                logits = self.model(src, tgt_input, src_mask, tgt_input_mask)
                loss = self.criterion(logits, tgt_target)

            total_loss += loss.item()

        avg_loss = total_loss / max(1, num_batches)
        # Cap perplexity exponent to prevent overflow on early random predictions
        perplexity = math.exp(min(avg_loss, 20.0))
        return avg_loss, perplexity

    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        val_ppl: float,
        is_best: bool = False,
    ) -> Path:
        """Persists model parameters, optimizer, and scheduler state to disk."""
        checkpoint: dict[str, Any] = {
            "epoch": epoch,
            "global_step": self.global_step,
            "val_loss": val_loss,
            "val_ppl": val_ppl,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }
        if self.scheduler is not None and hasattr(self.scheduler, "state_dict"):
            checkpoint["scheduler_state"] = self.scheduler.state_dict()
        if self.scaler.is_enabled():
            checkpoint["scaler_state"] = self.scaler.state_dict()

        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(
                f"Saved new best model checkpoint (Val Loss: {val_loss:.4f}, PPL: {val_ppl:.2f}) -> {best_path}"
            )
        return latest_path

    def load_checkpoint(self, checkpoint_path: str | Path) -> dict[str, Any]:
        """Restores model, optimizer, scheduler, and scaler states from checkpoint."""
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path.resolve()}")

        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])

        if "optimizer_state" in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if (
            "scheduler_state" in checkpoint
            and self.scheduler is not None
            and hasattr(self.scheduler, "load_state_dict")
        ):
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        if "scaler_state" in checkpoint and self.scaler.is_enabled():
            self.scaler.load_state_dict(checkpoint["scaler_state"])

        self.global_step = checkpoint.get("global_step", 0)
        self.logger.info(
            f"Loaded checkpoint from {ckpt_path} (epoch {checkpoint.get('epoch', '?')})"
        )
        return checkpoint

    def train(
        self,
        max_epochs: int = 30,
        patience: int = 5,
        save_interval: int = 1,
    ) -> dict[str, list[float]]:
        """Executes complete training loop with validation, checkpointing, and early stopping.

        Args:
            max_epochs: Maximum number of epochs to train.
            patience: Number of epochs without validation loss improvement before stopping.
            save_interval: Interval of epochs at which checkpoints are persisted.

        Returns:
            Dictionary containing loss and perplexity history curves.
        """
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_ppl": []}
        epochs_without_improvement = 0

        self.logger.info(
            f"Starting training: {max_epochs} epochs | Device: {self.device} "
            f"| AMP: {self.mixed_precision} | Accumulation: {self.grad_accum_steps}"
        )

        for epoch in range(1, max_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, val_ppl = self.evaluate()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_ppl"].append(val_ppl)

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            self.logger.info(
                f"--> Epoch {epoch:02d}/{max_epochs:02d} Finished | Train Loss: {train_loss:.4f} "
                f"| Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} "
                f"{'⭐ BEST' if is_best else f'(no gain for {epochs_without_improvement}/{patience})'}"
            )

            if epoch % save_interval == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, val_ppl, is_best=is_best)

            if epochs_without_improvement >= patience:
                self.logger.info(
                    f"Early stopping triggered: No validation improvement for {patience} consecutive epochs. "
                    f"Best Epoch: {self.best_epoch} (Val Loss: {self.best_val_loss:.4f})"
                )
                break

        return history
