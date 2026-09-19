import json
from pathlib import Path

import numpy as np
import pytest
import torch

from amnmt.tokenization.tokenizer import Tokenizer
from amnmt.training.data import (
    Collate,
    PairDataset,
    TokenBudgetBatchSampler,
    TokenizedCorpus,
    load_or_tokenize,
)
from amnmt.training.loss import LabelSmoothedCrossEntropy
from amnmt.training.scheduler import inverse_sqrt_factor
from amnmt.training.trainer import Trainer
from conftest import AM, EN
from conftest import training_config as _config


def test_inverse_sqrt_schedule() -> None:
    assert inverse_sqrt_factor(1, 100) == pytest.approx(0.01)
    assert inverse_sqrt_factor(100, 100) == pytest.approx(1.0)
    assert inverse_sqrt_factor(400, 100) == pytest.approx(0.5)


def test_label_smoothed_loss_matches_torch() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 11)
    target = torch.randint(1, 11, (2, 5))
    target[0, -2:] = 0  # pad
    crit = LabelSmoothedCrossEntropy(pad_id=0, smoothing=0.1)
    loss, nll, n = crit(logits, target)
    assert n == 8
    ref = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 11),
        target.reshape(-1),
        ignore_index=0,
        label_smoothing=0.1,
        reduction="sum",
    )
    ref_nll = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 11), target.reshape(-1), ignore_index=0, reduction="sum"
    )
    assert torch.allclose(loss, ref, atol=1e-5) and torch.allclose(nll, ref_nll, atol=1e-5)


def test_token_budget_sampler_respects_budget_and_covers_everything() -> None:
    rng = np.random.default_rng(0)
    lengths = rng.integers(3, 30, size=500)
    sampler = TokenBudgetBatchSampler(lengths, batch_tokens=120, seed=1)
    batches = list(sampler)
    seen = sorted(i for b in batches for i in b)
    assert seen == list(range(500))
    for b in batches:
        assert len(b) * lengths[b].max() <= 120
    # deterministic per epoch, different across epochs
    assert list(sampler) == batches
    sampler.set_epoch(1)
    assert list(sampler) != batches
    sampler.set_epoch(0, skip=3)
    assert list(sampler) == batches[3:]
    with pytest.raises(ValueError, match="longest sample"):
        TokenBudgetBatchSampler(np.array([200]), batch_tokens=100, seed=0)


def test_pair_dataset_builds_both_directions(workspace: Path) -> None:
    tok = Tokenizer.from_file(workspace / "artifacts" / "tokenizer" / "tokenizer.json")
    corpus = TokenizedCorpus(
        *[np.array(x) for x in ([1, 2, 3, 4], [0, 2, 4], [5, 6, 7], [0, 1, 3])]
    )
    ds = PairDataset(corpus, tok, ["en-am", "am-en"])
    assert len(ds) == 4
    src, tgt_in, tgt_out = ds[0]  # pair 0, en->am
    assert src.tolist() == [tok.lang_id("am"), 1, 2, tok.eos_id]
    assert tgt_in.tolist() == [tok.bos_id, 5]
    assert tgt_out.tolist() == [5, tok.eos_id]
    src, _, tgt_out = ds[1]  # pair 0, am->en
    assert src.tolist() == [tok.lang_id("en"), 5, tok.eos_id]
    assert tgt_out.tolist() == [1, 2, tok.eos_id]
    assert ds.lengths().tolist() == [4, 4, 4, 4]
    src_b, tgt_in_b, tgt_out_b = Collate(tok.pad_id)([ds[0], ds[2]])
    assert src_b.shape == (2, 4) and tgt_in_b.shape == (2, 3) and tgt_out_b.shape == (2, 3)
    assert tgt_in_b[0, -1] == tok.pad_id  # pair 0 target is shorter than pair 1 target


def test_tokenized_cache_roundtrip_and_reuse(workspace: Path) -> None:
    tok_path = workspace / "artifacts" / "tokenizer" / "tokenizer.json"
    tok = Tokenizer.from_file(tok_path)
    from amnmt.core.config import SubsetConfig

    args = (tok, tok_path, workspace / "data/processed/train.parquet", workspace / "cache", "train")
    c1 = load_or_tokenize(*args, SubsetConfig(), 40, None, 0)
    assert len(c1) == 80
    assert tok.decode(c1.seq("am", 0).tolist()) in AM
    files = list((workspace / "cache").glob("train_*.npz"))
    assert len(files) == 1
    c2 = load_or_tokenize(*args, SubsetConfig(), 40, None, 0)  # cache hit
    assert np.array_equal(c1.en, c2.en) and len(files) == 1
    c3 = load_or_tokenize(*args, SubsetConfig(), 40, 16, 0)  # different key -> new file
    assert len(c3) == 16 and len(list((workspace / "cache").glob("train_*.npz"))) == 2
    mid = int(np.median(np.maximum(c1.lengths("en"), c1.lengths("am")))) + 2
    c4 = load_or_tokenize(*args, SubsetConfig(), mid, None, 0)  # max_len at the median drops some
    assert 0 < len(c4) < 80


# ----------------------------------------------------------------------------- end to end


def test_train_end_to_end_loss_decreases(workspace: Path) -> None:
    cfg = _config(workspace)
    run_dir = workspace / "artifacts" / "runs" / "e2e"
    state = Trainer(cfg, run_dir).train()
    assert state.step == 24
    assert (run_dir / "last.pt").exists() and (run_dir / "best.pt").exists()
    events = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    train_losses = [e["loss"] for e in events if e["event"] == "train"]
    evals = [e for e in events if e["event"] == "eval"]
    assert len(train_losses) == 6 and len(evals) == 2
    assert train_losses[-1] < train_losses[0]
    assert evals[-1]["holdout"]["nll"] < evals[0]["holdout"]["nll"]
    assert {"holdout", "flores_dev"} <= set(evals[0])
    sample = evals[0]["samples"][0]
    assert set(sample) == {"direction", "src", "ref", "hyp"} and sample["ref"] in AM + EN


def test_resume_reproduces_uninterrupted_run(workspace: Path) -> None:
    full_dir = workspace / "artifacts" / "runs" / "full"
    full = Trainer(_config(workspace), full_dir)
    full.train()

    part_dir = workspace / "artifacts" / "runs" / "part"
    Trainer(_config(workspace, max_steps=12), part_dir).train()  # stops at the save point
    resumed = Trainer(_config(workspace), part_dir)
    state = resumed.train(resume=True)
    assert state.step == 24

    for (n1, p1), (n2, p2) in zip(
        full.model.named_parameters(), resumed.model.named_parameters(), strict=True
    ):
        assert n1 == n2 and torch.allclose(p1, p2, atol=1e-6), n1
    steps = [
        json.loads(line)["step"]
        for line in (part_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert steps == sorted(steps) and 24 in steps


def test_time_limit_stops_early_and_checkpoints(workspace: Path) -> None:
    run_dir = workspace / "artifacts" / "runs" / "timed"
    cfg = _config(workspace, max_steps=10_000, time_limit_minutes=0)
    state = Trainer(cfg, run_dir).train()
    assert 0 < state.step < 10_000
    assert (run_dir / "last.pt").exists()
