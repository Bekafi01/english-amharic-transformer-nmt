"""Unit tests for training engine: LabelSmoothingLoss, NoamScheduler, and TransformerTrainer."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.nmt_engine.models.transformer import Seq2SeqTransformer
from src.nmt_engine.training.loss import LabelSmoothingLoss
from src.nmt_engine.training.scheduler import NoamScheduler
from src.nmt_engine.training.trainer import TransformerTrainer


def test_label_smoothing_loss() -> None:
    vocab_size = 100
    pad_id = 0
    criterion = LabelSmoothingLoss(vocab_size=vocab_size, pad_id=pad_id, smoothing=0.1)

    logits = torch.randn(2, 5, vocab_size, requires_grad=True)
    targets = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])

    loss = criterion(logits, targets)
    assert loss.item() > 0.0
    loss.backward()
    assert logits.grad is not None


def test_label_smoothing_loss_all_pad() -> None:
    vocab_size = 100
    pad_id = 0
    criterion = LabelSmoothingLoss(vocab_size=vocab_size, pad_id=pad_id, smoothing=0.1)

    logits = torch.randn(1, 3, vocab_size, requires_grad=True)
    targets = torch.tensor([[0, 0, 0]])

    loss = criterion(logits, targets)
    assert loss.item() == 0.0


def test_noam_scheduler() -> None:
    linear = nn.Linear(10, 10)
    optimizer = optim.Adam(linear.parameters(), lr=1.0)
    scheduler = NoamScheduler(optimizer=optimizer, d_model=512, warmup_steps=100)

    # Warmup phase: learning rate should increase strictly
    lrs = []
    for _ in range(100):
        lr = scheduler.step()
        lrs.append(lr)

    for i in range(len(lrs) - 1):
        assert lrs[i] < lrs[i + 1]

    # Decay phase: learning rate should decrease strictly
    for _ in range(50):
        lr = scheduler.step()
        lrs.append(lr)

    assert lrs[-1] < lrs[99]  # step 150 < step 100

    # Test serialization
    state = scheduler.state_dict()
    assert state["step_num"] == 150

    new_scheduler = NoamScheduler(optimizer=optimizer, d_model=512, warmup_steps=100)
    new_scheduler.load_state_dict(state)
    assert new_scheduler.step_num == 150
    assert new_scheduler.get_lr() == scheduler.get_lr()


class SyntheticTranslationDataLoader:
    """Mock dataloader returning (src, tgt, src_mask, tgt_mask) batches."""

    def __init__(
        self, vocab_size: int = 200, seq_len: int = 8, batch_size: int = 4, num_batches: int = 3
    ):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_batches = num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            src = torch.randint(1, self.vocab_size, (self.batch_size, self.seq_len))
            tgt = torch.randint(1, self.vocab_size, (self.batch_size, self.seq_len))
            src_mask = torch.ones(self.batch_size, 1, 1, self.seq_len, dtype=torch.bool)
            tgt_mask = (
                torch.tril(torch.ones(self.seq_len, self.seq_len, dtype=torch.bool))
                .unsqueeze(0)
                .unsqueeze(0)
            )
            yield src, tgt, src_mask, tgt_mask

    def __len__(self):
        return self.num_batches


def test_transformer_trainer(tmp_path: Path) -> None:
    vocab_size = 200
    device = torch.device("cpu")

    model = Seq2SeqTransformer(
        vocab_size=vocab_size,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        pad_id=0,
        tie_weights=True,
    )

    train_loader = SyntheticTranslationDataLoader(vocab_size=vocab_size, num_batches=2)
    val_loader = SyntheticTranslationDataLoader(vocab_size=vocab_size, num_batches=1)

    criterion = LabelSmoothingLoss(vocab_size=vocab_size, pad_id=0, smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = NoamScheduler(optimizer=optimizer, d_model=64, warmup_steps=10)

    trainer = TransformerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        grad_accum_steps=1,
        clip_grad_norm=1.0,
        checkpoint_dir=tmp_path / "checkpoints",
        mixed_precision="no",
    )

    history = trainer.train(max_epochs=2, patience=2)
    assert len(history["train_loss"]) == 2
    assert len(history["val_loss"]) == 2
    assert (tmp_path / "checkpoints" / "latest.pt").exists()
    assert (tmp_path / "checkpoints" / "best_model.pt").exists()

    # Verify loading checkpoint
    loaded = trainer.load_checkpoint(tmp_path / "checkpoints" / "best_model.pt")
    assert "model_state" in loaded
    assert "optimizer_state" in loaded
