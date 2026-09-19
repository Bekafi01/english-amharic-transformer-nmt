"""Training loop: AMP, gradient accumulation, clipping, inverse-sqrt LR, eval, checkpoint/resume.

Run directory layout (`<artifacts>/runs/<run>/`):
    last.pt        rolling checkpoint (model + optimizer + scheduler + sampler position + RNG)
    best.pt        model with the lowest holdout NLL
    metrics.jsonl  one JSON object per log/eval event
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from amnmt.core.config import Config, SubsetConfig
from amnmt.core.logging import get_logger
from amnmt.model.greedy import greedy_decode
from amnmt.model.transformer import Seq2SeqTransformer
from amnmt.tokenization.tokenizer import Tokenizer
from amnmt.training.data import (
    Collate,
    PairDataset,
    TokenBudgetBatchSampler,
    TokenizedCorpus,
    load_or_tokenize,
)
from amnmt.training.loss import LabelSmoothedCrossEntropy
from amnmt.training.scheduler import build_scheduler

log = get_logger(__name__)


@dataclass
class TrainState:
    step: int = 0  # optimizer steps
    epoch: int = 0
    batch_in_epoch: int = 0  # micro-batches consumed in the current epoch (for resume)
    tokens_seen: int = 0
    best_holdout_nll: float = float("inf")
    elapsed_seconds: float = 0.0


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


class Trainer:
    def __init__(self, cfg: Config, run_dir: Path) -> None:
        if cfg.model is None or cfg.training is None:
            raise ValueError("config needs `model` and `training` sections")
        self.cfg = cfg
        self.tcfg = cfg.training
        self.run_dir = run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        self.device = resolve_device(cfg.project.device)
        torch.manual_seed(cfg.project.seed)

        artifacts = cfg.paths.resolve("artifacts")
        processed = cfg.paths.resolve("data_processed")
        self.tok_path = artifacts / "tokenizer" / "tokenizer.json"
        self.tok = Tokenizer.from_file(self.tok_path)
        cache = artifacts / "cache"
        t = self.tcfg

        def corpus(split: str, subset: SubsetConfig, max_pairs: int | None) -> TokenizedCorpus:
            return load_or_tokenize(
                self.tok, self.tok_path, processed / f"{split}.parquet", cache, split,
                subset, t.max_len, max_pairs, cfg.project.seed,
            )  # fmt: skip

        self.train_corpus = corpus("train", t.subset, t.max_pairs)
        self.eval_corpora = {
            "holdout": corpus("train_holdout", SubsetConfig(), None),
            "flores_dev": corpus("valid", SubsetConfig(), None),
        }
        self.train_ds = PairDataset(self.train_corpus, self.tok, list(t.directions))
        self.sampler = TokenBudgetBatchSampler(
            self.train_ds.lengths(), t.batch_tokens, cfg.project.seed
        )
        self.collate = Collate(self.tok.pad_id)

        self.model = Seq2SeqTransformer(cfg.model, self.tok.vocab_size, self.tok.pad_id).to(
            self.device
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=t.peak_lr,
            betas=t.adam_betas,
            eps=t.adam_eps,
            weight_decay=t.weight_decay,
        )
        self.scheduler = build_scheduler(self.optimizer, t.warmup_steps)
        self.criterion = LabelSmoothedCrossEntropy(self.tok.pad_id, t.label_smoothing)

        use_amp = t.amp and self.device.type == "cuda"
        # Native bf16 needs Ampere (sm_80+). is_bf16_supported() also says True for *emulated*
        # bf16 on Turing (T4), which is several times slower than fp16 there.
        native_bf16 = use_amp and torch.cuda.get_device_capability(self.device) >= (8, 0)
        self.amp_dtype = torch.bfloat16 if native_bf16 else torch.float16
        self.use_amp = use_amp
        self.scaler = torch.amp.GradScaler(enabled=use_amp and self.amp_dtype == torch.float16)
        self.state = TrainState()
        if self.device.type == "cuda":
            major, minor = torch.cuda.get_device_capability(self.device)
            log.info("gpu %s (sm_%d%d)", torch.cuda.get_device_name(self.device), major, minor)
        log.info(
            "device=%s amp=%s params=%s train_pairs=%s samples=%s batches/epoch=%s",
            self.device,
            self.amp_dtype if use_amp else "off",
            f"{self.model.num_parameters():,}",
            f"{len(self.train_corpus):,}",
            f"{len(self.train_ds):,}",
            f"{len(self.sampler):,}",
        )

    # ------------------------------------------------------------------ checkpoints

    def _checkpoint(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "state": asdict(self.state),
            "config": self.cfg.model_dump(mode="json"),
            "vocab_size": self.tok.vocab_size,
            "pad_id": self.tok.pad_id,
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }

    def save(self, name: str) -> Path:
        path = self.run_dir / f"{name}.pt"
        tmp = path.with_suffix(".pt.tmp")
        torch.save(self._checkpoint(), tmp)
        tmp.replace(path)
        return path

    def load(self, path: Path) -> None:
        # Load on CPU: load_state_dict() moves tensors to the module/optimizer device itself,
        # and torch.set_rng_state requires a *CPU* ByteTensor (map_location=cuda broke resume).
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.state = TrainState(**ckpt["state"])
        torch.set_rng_state(ckpt["rng"]["torch"].cpu())
        if ckpt["rng"]["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["rng"]["cuda"]])
        log.info("resumed from %s at step %s", path.name, self.state.step)

    # ------------------------------------------------------------------ eval

    def _eval_loader(self, corpus: TokenizedCorpus) -> DataLoader[tuple[Tensor, Tensor, Tensor]]:
        ds = PairDataset(corpus, self.tok, list(self.tcfg.directions))
        sampler = TokenBudgetBatchSampler(ds.lengths(), self.tcfg.batch_tokens, seed=0)
        return DataLoader(
            ds,
            batch_sampler=sampler,
            collate_fn=self.collate,
            num_workers=0,
            generator=torch.Generator().manual_seed(0),
        )

    @torch.no_grad()
    def evaluate(self, corpus: TokenizedCorpus) -> dict[str, float]:
        self.model.eval()
        loss_sum = nll_sum = 0.0
        n_tok = 0
        for src, tgt_in, tgt_out in self._eval_loader(corpus):
            src, tgt_in, tgt_out = (x.to(self.device) for x in (src, tgt_in, tgt_out))
            with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                logits = self.model(src, tgt_in)
            loss, nll, n = self.criterion(logits, tgt_out)
            loss_sum += float(loss)
            nll_sum += float(nll)
            n_tok += n
        self.model.train()
        return {
            "loss": loss_sum / max(n_tok, 1),
            "nll": nll_sum / max(n_tok, 1),
            "ppl": self.criterion.perplexity(nll_sum, n_tok),
        }

    @torch.no_grad()
    def sample_translations(self, corpus: TokenizedCorpus, n: int) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for direction in self.tcfg.directions:
            ds = PairDataset(corpus, self.tok, [direction])
            items = [ds[i] for i in range(min(n, len(ds)))]
            src, _, tgt_out = self.collate(items)
            hyp = greedy_decode(
                self.model, src.to(self.device), self.tok.bos_id, self.tok.eos_id, self.tcfg.max_len
            )
            self.model.train()
            for s, r, h in zip(src, tgt_out, hyp.cpu(), strict=True):
                out.append(
                    {
                        "direction": direction,
                        "src": self.tok.decode(s.tolist()),
                        "ref": self.tok.decode(r.tolist()),
                        "hyp": self.tok.decode(h.tolist()),
                    }
                )
        return out

    # ------------------------------------------------------------------ train

    def _log_metrics(self, record: dict[str, Any]) -> None:
        record = {"step": self.state.step, "time": time.time(), **record}
        with (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _run_eval(self) -> None:
        results = {name: self.evaluate(c) for name, c in self.eval_corpora.items()}
        samples = self.sample_translations(self.eval_corpora["holdout"], self.tcfg.n_samples)
        self._log_metrics({"event": "eval", **results, "samples": samples})
        for name, r in results.items():
            log.info("eval %-10s loss=%.4f nll=%.4f ppl=%.2f", name, r["loss"], r["nll"], r["ppl"])
        for s in samples[: self.tcfg.n_samples]:
            log.info("  [%s] SRC %s", s["direction"], s["src"])
            log.info("  %s REF %s", " " * len(s["direction"]), s["ref"])
            log.info("  %s HYP %s", " " * len(s["direction"]), s["hyp"])
        holdout_nll = results["holdout"]["nll"]
        if holdout_nll < self.state.best_holdout_nll:
            self.state.best_holdout_nll = holdout_nll
            self.save("best")
            log.info("new best holdout nll %.4f -> best.pt", holdout_nll)

    def train(self, resume: bool = False) -> TrainState:
        last = self.run_dir / "last.pt"
        if resume and last.exists():
            self.load(last)
        elif resume:
            log.info("no checkpoint at %s; starting fresh", last)
        t = self.tcfg
        self.model.train()
        # elapsed_seconds is cumulative across sessions; the time limit is per session.
        session_start = time.perf_counter()
        t_start = session_start - self.state.elapsed_seconds
        deadline = (
            None if t.time_limit_minutes is None else session_start + 60 * t.time_limit_minutes
        )
        window_loss = window_tok = 0.0
        window_t0 = time.perf_counter()
        micro = 0
        stop = self.state.step >= t.max_steps

        while not stop:
            self.sampler.set_epoch(self.state.epoch, skip=self.state.batch_in_epoch)
            loader = DataLoader(
                self.train_ds,
                batch_sampler=self.sampler,
                collate_fn=self.collate,
                num_workers=t.num_workers if self.device.type == "cuda" else 0,
                pin_memory=self.device.type == "cuda",
                persistent_workers=False,
                # Own generator: constructing a loader must not consume the global RNG, or a
                # resumed run would draw different dropout masks than an uninterrupted one.
                generator=torch.Generator().manual_seed(self.cfg.project.seed + self.state.epoch),
            )
            for src, tgt_in, tgt_out in loader:
                src, tgt_in, tgt_out = (
                    x.to(self.device, non_blocking=True) for x in (src, tgt_in, tgt_out)
                )
                with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                    logits = self.model(src, tgt_in)
                loss_sum, _, n_tok = self.criterion(logits, tgt_out)
                loss = loss_sum / max(n_tok, 1) / t.accumulation_steps
                self.scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
                window_loss += float(loss_sum.detach())
                window_tok += n_tok
                self.state.tokens_seen += n_tok
                self.state.batch_in_epoch += 1
                micro += 1
                if micro % t.accumulation_steps:
                    continue

                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), t.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.state.step += 1
                self.state.elapsed_seconds = time.perf_counter() - t_start

                if self.state.step % t.log_every == 0:
                    dt = time.perf_counter() - window_t0
                    lr = self.scheduler.get_last_lr()[0]
                    rec = {
                        "event": "train",
                        "loss": window_loss / max(window_tok, 1),
                        "lr": lr,
                        "tok_per_s": window_tok / max(dt, 1e-9),
                        "epoch": self.state.epoch,
                        "elapsed_min": self.state.elapsed_seconds / 60,
                    }
                    self._log_metrics(rec)
                    log.info(
                        "step %6d ep %d loss %.4f lr %.2e %.0f tok/s %.1f min",
                        self.state.step, rec["epoch"], rec["loss"], lr, rec["tok_per_s"],
                        rec["elapsed_min"],
                    )  # fmt: skip
                    window_loss = window_tok = 0.0
                    window_t0 = time.perf_counter()
                if self.state.step % t.eval_every == 0:
                    self._run_eval()
                if self.state.step % t.save_every == 0:
                    self.save("last")
                out_of_time = deadline is not None and time.perf_counter() > deadline
                if self.state.step >= t.max_steps or out_of_time:
                    if out_of_time:
                        log.info("time limit reached at step %d", self.state.step)
                    stop = True
                    break
            else:
                self.state.epoch += 1
                self.state.batch_in_epoch = 0
                log.info("epoch %d complete", self.state.epoch)

        if self.state.step % t.eval_every:
            self._run_eval()
        self.save("last")
        log.info(
            "done: step %d, best holdout nll %.4f", self.state.step, self.state.best_holdout_nll
        )
        return self.state
