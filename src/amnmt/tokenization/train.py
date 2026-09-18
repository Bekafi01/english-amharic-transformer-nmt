"""Train the joint English+Amharic BPE tokenizer from a sample of train.parquet.

Recipe (SentencePiece-like): NFC -> Metaspace word split + individual digits -> BPE with byte
fallback. Byte fallback means no text is ever `<unk>`: unseen glyphs become `<0x..>` byte tokens.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import duckdb
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import decoders, models, normalizers, pre_tokenizers, trainers

from amnmt.core.config import Config
from amnmt.core.logging import get_logger
from amnmt.tokenization.tokenizer import BYTE_TOKENS, SPECIAL_TOKENS, UNK, Tokenizer

log = get_logger(__name__)

_META = "\u2581"  # ▁
_FETCH = 20_000


def build_untrained() -> HFTokenizer:
    tok = HFTokenizer(models.BPE(unk_token=UNK, byte_fallback=True))
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Metaspace(replacement=_META, prepend_scheme="always"),
            pre_tokenizers.Punctuation(behavior="isolated"),  # ነው። -> ነው ። (no ▁, so lossless)
            pre_tokenizers.Digits(individual_digits=True),
        ]
    )
    tok.decoder = decoders.Sequence(
        [decoders.ByteFallback(), decoders.Metaspace(replacement=_META, prepend_scheme="always")]
    )
    return tok


def train_from_texts(texts: Iterator[str], vocab_size: int, min_frequency: int) -> HFTokenizer:
    tok = build_untrained()
    trainer = trainers.BpeTrainer(  # type: ignore[no-untyped-call]
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[*SPECIAL_TOKENS, *BYTE_TOKENS],
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


# ---------------------------------------------------------------------------- sampling


def _sql_path(p: Path) -> str:
    return str(p).replace("\\", "/")


def sample_column(
    con: duckdb.DuckDBPyConnection, parquet: Path, column: str, where: str, n: int, seed: int
) -> Iterator[str]:
    """Deterministic reservoir sample of one text column; yields all rows if fewer than n."""
    cur = con.execute(
        f"SELECT {column} FROM (SELECT {column} FROM read_parquet('{_sql_path(parquet)}') "
        f"WHERE {where}) USING SAMPLE reservoir({n} ROWS) REPEATABLE ({seed})"
    )
    while rows := cur.fetchmany(_FETCH):
        for (text,) in rows:
            yield text


# ---------------------------------------------------------------------------- stats


def _column_stats(tok: Tokenizer, texts: list[str]) -> dict[str, float]:
    ids = tok.encode_batch(texts)
    n_tok = sum(len(x) for x in ids)
    n_chr = sum(len(t) for t in texts)
    n_byte = sum(1 for x in ids for i in x if tok.is_byte_token(i))
    n_unk = sum(1 for x in ids for i in x if i == tok.unk_id)
    return {
        "sentences": len(texts),
        "tokens_per_sentence": round(n_tok / max(len(texts), 1), 3),
        "chars_per_token": round(n_chr / max(n_tok, 1), 3),
        "byte_fallback_rate": round(n_byte / max(n_tok, 1), 6),
        "unk_rate": round(n_unk / max(n_tok, 1), 6),
    }


def compute_stats(tok: Tokenizer, holdout: Path) -> dict[str, Any]:
    con = duckdb.connect()
    en, am = (
        [
            r[0]
            for r in con.execute(f"SELECT {c} FROM read_parquet('{_sql_path(holdout)}')").fetchall()
        ]
        for c in ("en", "am")
    )
    s_en, s_am = _column_stats(tok, en), _column_stats(tok, am)
    return {
        "vocab_size": tok.vocab_size,
        "en": s_en,
        "am": s_am,
        "am_en_length_ratio": round(s_am["tokens_per_sentence"] / s_en["tokens_per_sentence"], 3),
        "roundtrip_exact": sum(tok.decode(tok.encode(t)) == t for t in en + am)
        / max(len(en + am), 1),
    }


# ---------------------------------------------------------------------------- entry


def train(cfg: Config) -> dict[str, Any]:
    if cfg.tokenizer is None:
        raise ValueError("config has no `tokenizer` section")
    tcfg = cfg.tokenizer
    processed = cfg.paths.resolve("data_processed")
    out_dir = cfg.paths.resolve("artifacts") / "tokenizer"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_parquet = processed / "train.parquet"
    where = tcfg.subset.sql_where()
    t0 = time.perf_counter()

    con = duckdb.connect()
    log.info(
        "sampling %s sentences/lang from %s where %s",
        f"{tcfg.sample_per_lang:,}",
        train_parquet,
        where,
    )
    texts: list[str] = []
    for col in ("en", "am"):
        texts.extend(
            sample_column(con, train_parquet, col, where, tcfg.sample_per_lang, cfg.project.seed)
        )
    log.info("training BPE vocab=%s on %s sentences", f"{tcfg.vocab_size:,}", f"{len(texts):,}")
    hf = train_from_texts(iter(texts), tcfg.vocab_size, tcfg.min_frequency)
    tok_path = out_dir / "tokenizer.json"
    hf.save(str(tok_path))

    tok = Tokenizer.from_file(tok_path)
    stats = compute_stats(tok, processed / "train_holdout.parquet")
    stats.update(
        {
            "trained_on_sentences": len(texts),
            "subset_where": where,
            "seconds": round(time.perf_counter() - t0, 1),
            "path": str(tok_path),
        }
    )
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log.info(
        "done: vocab=%s tok/sent en=%.1f am=%.1f ratio=%.2f byte-fallback am=%.4f%% (%.0fs)",
        stats["vocab_size"],
        stats["en"]["tokens_per_sentence"],
        stats["am"]["tokens_per_sentence"],
        stats["am_en_length_ratio"],
        100 * stats["am"]["byte_fallback_rate"],
        stats["seconds"],
    )
    return stats
