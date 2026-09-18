"""Tokenized training data: cached flat arrays, bidirectional pair dataset, token-budget batching.

Cache layout (`<artifacts>/cache/<key>.npz`): for each language a flat `uint16` token array plus
`int64` offsets. A sample is (pair index, direction); sequences are assembled on the fly:
    src     = <2xx> X </s>
    tgt_in  = <s>  Y
    tgt_out =      Y </s>
"""

from __future__ import annotations

import hashlib
import itertools
import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from amnmt.core.config import SubsetConfig
from amnmt.core.logging import get_logger
from amnmt.tokenization.tokenizer import Tokenizer

log = get_logger(__name__)

_FETCH = 50_000
_SPECIALS = 2  # per side: tag/bos + eos
DIRECTION_LANGS = {"en-am": ("en", "am"), "am-en": ("am", "en")}


# ---------------------------------------------------------------------------- tokenized corpus


@dataclass
class TokenizedCorpus:
    en: NDArray[np.uint16]
    en_off: NDArray[np.int64]
    am: NDArray[np.uint16]
    am_off: NDArray[np.int64]

    def __len__(self) -> int:
        return len(self.en_off) - 1

    def seq(self, lang: str, i: int) -> NDArray[np.uint16]:
        flat, off = (self.en, self.en_off) if lang == "en" else (self.am, self.am_off)
        return flat[off[i] : off[i + 1]]

    def lengths(self, lang: str) -> NDArray[np.int64]:
        off = self.en_off if lang == "en" else self.am_off
        return np.diff(off)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, en=self.en, en_off=self.en_off, am=self.am, am_off=self.am_off)

    @classmethod
    def load(cls, path: Path) -> TokenizedCorpus:
        z = np.load(path)
        return cls(z["en"], z["en_off"], z["am"], z["am_off"])


class _Packer:
    """Accumulates variable-length id lists into one flat uint16 array + offsets, chunk by chunk."""

    def __init__(self) -> None:
        self._flat: list[NDArray[np.uint16]] = []
        self._lens: list[NDArray[np.int64]] = []

    def add(self, seqs: list[list[int]]) -> None:
        if not seqs:
            return
        self._lens.append(np.fromiter((len(s) for s in seqs), dtype=np.int64, count=len(seqs)))
        self._flat.append(np.fromiter(itertools.chain.from_iterable(seqs), dtype=np.uint16))

    def finish(self) -> tuple[NDArray[np.uint16], NDArray[np.int64]]:
        lens = np.concatenate(self._lens) if self._lens else np.zeros(0, dtype=np.int64)
        off = np.zeros(len(lens) + 1, dtype=np.int64)
        np.cumsum(lens, out=off[1:])
        flat = np.concatenate(self._flat) if self._flat else np.zeros(0, dtype=np.uint16)
        return flat, off


def _iter_rows(
    parquet: Path, where: str, max_pairs: int | None, seed: int
) -> Iterator[list[tuple[str, str]]]:
    p = str(parquet).replace("\\", "/")
    sql = f"SELECT en, am FROM read_parquet('{p}') WHERE {where}"
    if max_pairs is not None:
        # Deterministic cap: DuckDB's reservoir sample is seeded but not order-stable; hashing is.
        sql += f" ORDER BY hash(en || am || '{seed}') LIMIT {max_pairs}"
    cur = duckdb.connect().execute(sql)
    while rows := cur.fetchmany(_FETCH):
        yield rows


def tokenize_parquet(
    tok: Tokenizer, parquet: Path, where: str, max_len: int, max_pairs: int | None, seed: int
) -> tuple[TokenizedCorpus, dict[str, int]]:
    """Encode both columns; drop pairs where either side exceeds max_len (incl. specials)."""
    if tok.vocab_size > np.iinfo(np.uint16).max + 1:
        raise ValueError("vocab too large for uint16 cache")
    en_pack, am_pack = _Packer(), _Packer()
    limit = max_len - _SPECIALS
    read = dropped = kept = 0
    t0 = time.perf_counter()
    for rows in _iter_rows(parquet, where, max_pairs, seed):
        en_ids = tok.encode_batch([r[0] for r in rows])
        am_ids = tok.encode_batch([r[1] for r in rows])
        en_keep: list[list[int]] = []
        am_keep: list[list[int]] = []
        for e, a in zip(en_ids, am_ids, strict=True):
            read += 1
            if len(e) > limit or len(a) > limit or not e or not a:
                dropped += 1
                continue
            en_keep.append(e)
            am_keep.append(a)
        en_pack.add(en_keep)
        am_pack.add(am_keep)
        kept += len(en_keep)
        if read % 1_000_000 < _FETCH:
            log.info("  tokenized %s pairs (%.0fs)", f"{read:,}", time.perf_counter() - t0)
    en, en_off = en_pack.finish()
    am, am_off = am_pack.finish()
    return TokenizedCorpus(en, en_off, am, am_off), {"read": read, "dropped": dropped, "kept": kept}


def cache_key(
    tokenizer_path: Path,
    split: str,
    subset: SubsetConfig,
    max_len: int,
    max_pairs: int | None,
    seed: int,
) -> str:
    ident = json.dumps(
        {
            # size + mtime identify tokenizer.json well enough without hashing 2 MB each run
            "tok": tokenizer_path.stat().st_size,
            "tok_mtime": int(tokenizer_path.stat().st_mtime),
            "split": split,
            "where": subset.sql_where(),
            "max_len": max_len,
            "max_pairs": max_pairs,
            "seed": seed,
        },
        sort_keys=True,
    )
    return f"{split}_{hashlib.blake2b(ident.encode(), digest_size=6).hexdigest()}"


def load_or_tokenize(
    tok: Tokenizer,
    tokenizer_path: Path,
    parquet: Path,
    cache_dir: Path,
    split: str,
    subset: SubsetConfig,
    max_len: int,
    max_pairs: int | None,
    seed: int,
) -> TokenizedCorpus:
    path = cache_dir / f"{cache_key(tokenizer_path, split, subset, max_len, max_pairs, seed)}.npz"
    if path.exists():
        corpus = TokenizedCorpus.load(path)
        log.info("cache   %s: %s pairs from %s", split, f"{len(corpus):,}", path.name)
        return corpus
    log.info("tokenize %s from %s where %s", split, parquet.name, subset.sql_where())
    corpus, stats = tokenize_parquet(tok, parquet, subset.sql_where(), max_len, max_pairs, seed)
    corpus.save(path)
    log.info(
        "  %s: kept %s / %s (dropped %s > max_len) -> %s",
        split,
        f"{stats['kept']:,}",
        f"{stats['read']:,}",
        stats["dropped"],
        path.name,
    )
    return corpus


# ---------------------------------------------------------------------------- dataset


class PairDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Index space is (pair, direction) flattened: idx = pair * n_dirs + d."""

    def __init__(self, corpus: TokenizedCorpus, tok: Tokenizer, directions: list[str]) -> None:
        self.corpus = corpus
        self.directions = directions
        self.bos, self.eos = tok.bos_id, tok.eos_id
        self.tags = {d: tok.lang_id(DIRECTION_LANGS[d][1]) for d in directions}

    def __len__(self) -> int:
        return len(self.corpus) * len(self.directions)

    def lengths(self) -> NDArray[np.int64]:
        """Padded length of each sample = max(src, tgt) incl. specials; used for bucketing."""
        per_dir = []
        for d in self.directions:
            s, t = DIRECTION_LANGS[d]
            per_dir.append(np.maximum(self.corpus.lengths(s), self.corpus.lengths(t)) + _SPECIALS)
        return np.stack(per_dir, axis=1).reshape(-1)  # interleaved to match idx layout

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        pair, d = divmod(idx, len(self.directions))
        direction = self.directions[d]
        s_lang, t_lang = DIRECTION_LANGS[direction]
        x = torch.from_numpy(self.corpus.seq(s_lang, pair).astype(np.int64))
        y = torch.from_numpy(self.corpus.seq(t_lang, pair).astype(np.int64))
        tag = torch.tensor([self.tags[direction]])
        bos, eos = torch.tensor([self.bos]), torch.tensor([self.eos])
        return torch.cat([tag, x, eos]), torch.cat([bos, y]), torch.cat([y, eos])


class Collate:
    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, batch: list[tuple[Tensor, Tensor, Tensor]]) -> tuple[Tensor, Tensor, Tensor]:
        src, tgt_in, tgt_out = zip(*batch, strict=True)
        pad = torch.nn.utils.rnn.pad_sequence
        return (
            pad(list(src), batch_first=True, padding_value=self.pad_id),
            pad(list(tgt_in), batch_first=True, padding_value=self.pad_id),
            pad(list(tgt_out), batch_first=True, padding_value=self.pad_id),
        )


# ---------------------------------------------------------------------------- batching


class TokenBudgetBatchSampler(Sampler[list[int]]):
    """Length-bucketed batches with `n_seqs * max_len <= batch_tokens`, deterministic per epoch.

    Shuffle indices -> cut into pools of ~100 batches -> sort each pool by length -> pack ->
    shuffle the batch order. `set_epoch()` changes the permutation; `skip` supports resume.
    """

    def __init__(self, lengths: NDArray[np.int64], batch_tokens: int, seed: int) -> None:
        self.lengths = lengths
        self.batch_tokens = batch_tokens
        self.seed = seed
        self.epoch = 0
        self.skip = 0
        if lengths.max(initial=0) > batch_tokens:
            raise ValueError("batch_tokens smaller than the longest sample")

    def set_epoch(self, epoch: int, skip: int = 0) -> None:
        self.epoch, self.skip = epoch, skip

    def batches(self) -> list[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        perm = rng.permutation(len(self.lengths))
        pool_size = max(self.batch_tokens // max(int(self.lengths.mean()), 1) * 100, 1)
        batches: list[list[int]] = []
        for start in range(0, len(perm), pool_size):
            pool = perm[start : start + pool_size]
            pool = pool[np.argsort(self.lengths[pool], kind="stable")]
            cur: list[int] = []
            cur_max = 0
            for idx in pool:
                length = int(self.lengths[idx])
                new_max = max(cur_max, length)
                if cur and new_max * (len(cur) + 1) > self.batch_tokens:
                    batches.append(cur)
                    cur, new_max = [], length
                cur.append(int(idx))
                cur_max = new_max
            if cur:
                batches.append(cur)
        order = rng.permutation(len(batches))
        return [batches[i] for i in order]

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self.batches()[self.skip :])

    def __len__(self) -> int:
        return len(self.batches()) - self.skip
