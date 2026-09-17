"""Corpus build: stream -> normalize -> filter -> per-source parquet shards -> dedup -> splits.

Memory is bounded by `shard_flush_rows`; the dedup/shuffle step runs in DuckDB on disk.
Outputs (under `paths.data_processed`):
    shards/<source>.parquet, shards/<source>.stats.json
    valid.parquet (FLORES dev), test.parquet (FLORES devtest)
    train.parquet, train_holdout.parquet, data_card.json
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from amnmt.core.config import Config, DataConfig, SourceConfig
from amnmt.core.logging import get_logger
from amnmt.data.filters import reject_reason
from amnmt.data.flores import load_flores
from amnmt.data.normalize import dedup_key, normalize_amharic, normalize_english
from amnmt.data.sources import iter_source

log = get_logger(__name__)

SHARD_SCHEMA = pa.schema(
    [
        ("en", pa.string()),
        ("am", pa.string()),
        ("source", pa.string()),
        ("score", pa.float32()),  # source alignment score (NLLB LASER margin); null if none
        ("priority", pa.int16()),
        ("key_hash", pa.int64()),
    ]
)
EVAL_SCHEMA = pa.schema(
    [("en", pa.string()), ("am", pa.string()), ("en_raw", pa.string()), ("am_raw", pa.string())]
)
_LOG_EVERY = 500_000


@dataclass
class SourceStats:
    name: str
    read: int = 0
    kept: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0


def _key_hash(key_en: str, key_am: str) -> int:
    digest = hashlib.blake2b(f"{key_en}\t{key_am}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=True)


class _Normalizer:
    def __init__(self, cfg: DataConfig) -> None:
        self._fold = cfg.normalize.fold_homophones
        self._digits = cfg.normalize.numerals_to_digits

    def __call__(self, en: str, am: str) -> tuple[str, str]:
        return normalize_english(en), normalize_amharic(am, fold=self._fold, digits=self._digits)


# ---------------------------------------------------------------------------- FLORES


def _write_eval_splits(
    cfg: DataConfig, raw_dir: Path, out_dir: Path, norm: _Normalizer
) -> tuple[dict[str, int], set[str]]:
    flores = load_flores(cfg.flores, raw_dir)
    sizes: dict[str, int] = {}
    keys: set[str] = set()
    for split, fname in (("dev", "valid.parquet"), ("devtest", "test.parquet")):
        rows: dict[str, list[str]] = {"en": [], "am": [], "en_raw": [], "am_raw": []}
        for en_raw, am_raw in flores[split]:
            en, am = norm(en_raw, am_raw)
            rows["en"].append(en)
            rows["am"].append(am)
            rows["en_raw"].append(en_raw)
            rows["am_raw"].append(am_raw)
            keys.update((dedup_key(en), dedup_key(am), dedup_key(en_raw), dedup_key(am_raw)))
        pq.write_table(pa.table(rows, schema=EVAL_SCHEMA), out_dir / fname)
        sizes[fname.removesuffix(".parquet")] = len(flores[split])
    keys.discard("")
    return sizes, keys


# ---------------------------------------------------------------------------- shards


def _build_shard(
    src: SourceConfig,
    priority: int,
    cfg: DataConfig,
    raw_dir: Path,
    shard_path: Path,
    norm: _Normalizer,
    excluded_keys: set[str],
) -> SourceStats:
    stats = SourceStats(name=src.name)
    t0 = time.perf_counter()
    buf: dict[str, list[Any]] = {k: [] for k in SHARD_SCHEMA.names}
    writer = pq.ParquetWriter(shard_path, SHARD_SCHEMA, compression="zstd")

    def flush() -> None:
        if buf["en"]:
            writer.write_table(pa.table(buf, schema=SHARD_SCHEMA))
            for col in buf.values():
                col.clear()

    try:
        for en_raw, am_raw, score in iter_source(src, raw_dir):
            stats.read += 1
            if stats.read % _LOG_EVERY == 0:
                log.info("  %s: %s read, %s kept", src.name, f"{stats.read:,}", f"{stats.kept:,}")
            en, am = norm(en_raw, am_raw)
            reason = reject_reason(en, am, cfg.filters)
            if reason is None:
                key_en, key_am = dedup_key(en), dedup_key(am)
                if key_en in excluded_keys or key_am in excluded_keys:
                    reason = "flores"
            if reason is not None:
                stats.rejected[reason] = stats.rejected.get(reason, 0) + 1
                continue
            stats.kept += 1
            buf["en"].append(en)
            buf["am"].append(am)
            buf["source"].append(src.name)
            buf["score"].append(score)
            buf["priority"].append(priority)
            buf["key_hash"].append(_key_hash(key_en, key_am))
            if len(buf["en"]) >= cfg.shard_flush_rows:
                flush()
        flush()
    finally:
        writer.close()
    stats.seconds = round(time.perf_counter() - t0, 1)
    return stats


def _stats_path(shard_path: Path) -> Path:
    return shard_path.with_suffix(".stats.json")


def _save_stats(shard_path: Path, stats: SourceStats) -> None:
    _stats_path(shard_path).write_text(json.dumps(asdict(stats), indent=2), encoding="utf-8")


def _load_stats(shard_path: Path) -> SourceStats:
    raw = json.loads(_stats_path(shard_path).read_text(encoding="utf-8"))
    return SourceStats(**raw)


# ---------------------------------------------------------------------------- dedup + split


def _scalar(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    row = con.execute(sql).fetchone()
    assert row is not None
    return int(row[0])


def _dedup_and_split(
    shards_dir: Path, out_dir: Path, work_dir: Path, holdout: int, seed: int
) -> dict[str, Any]:
    # Scratch DB lives in work_dir (local disk on Colab); out_dir may be a slow Drive mount.
    db_path = work_dir / "_build.duckdb"
    db_path.unlink(missing_ok=True)
    con = duckdb.connect(str(db_path))
    glob = str(shards_dir / "*.parquet").replace("\\", "/")
    try:
        con.execute("SET preserve_insertion_order = false")
        con.execute("SET memory_limit = '6GB'")
        before = _scalar(con, f"SELECT count(*) FROM read_parquet('{glob}', union_by_name = true)")
        # Same normalized pair in several sources: keep the most trusted source, and within a
        # source the best-scored copy.
        con.execute(
            f"""
            CREATE TABLE pairs AS
            SELECT en, am, source, score, key_hash,
                   row_number() OVER (ORDER BY hash(xor(key_hash, {seed}::BIGINT))) AS rn
            FROM (
                SELECT en, am, source, score, key_hash
                FROM read_parquet('{glob}', union_by_name = true)
                QUALIFY row_number() OVER (
                    PARTITION BY key_hash ORDER BY priority, score DESC NULLS LAST
                ) = 1
            )
            """
        )
        after = _scalar(con, "SELECT count(*) FROM pairs")
        by_source: dict[str, int] = dict(
            con.execute("SELECT source, count(*) FROM pairs GROUP BY source ORDER BY 1").fetchall()
        )
        score_quantiles: dict[str, dict[str, float]] = {}
        for source, *qs in con.execute(
            """
            SELECT source, min(score),
                   quantile_cont(score, 0.10), quantile_cont(score, 0.25),
                   quantile_cont(score, 0.50), quantile_cont(score, 0.75),
                   quantile_cont(score, 0.90), max(score)
            FROM pairs WHERE score IS NOT NULL GROUP BY source ORDER BY 1
            """
        ).fetchall():
            labels = ("min", "p10", "p25", "p50", "p75", "p90", "max")
            score_quantiles[source] = {
                k: round(float(v), 4) for k, v in zip(labels, qs, strict=True)
            }
        holdout = min(holdout, after // 10)
        for fname, cond in (("train_holdout", f"rn <= {holdout}"), ("train", f"rn > {holdout}")):
            target = str(out_dir / f"{fname}.parquet").replace("\\", "/")
            con.execute(
                f"COPY (SELECT en, am, source, score FROM pairs WHERE {cond} ORDER BY rn) "
                f"TO '{target}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
            )
    finally:
        con.close()
        db_path.unlink(missing_ok=True)
        for leftover in work_dir.glob("_build.duckdb*"):
            leftover.unlink(missing_ok=True)
    return {
        "before": before,
        "after": after,
        "removed": before - after,
        "by_source": by_source,
        "score_quantiles": score_quantiles,
        "train": after - holdout,
        "train_holdout": holdout,
    }


# ---------------------------------------------------------------------------- entry


def build(cfg: Config, *, rebuild: set[str] | None = None) -> dict[str, Any]:
    """Build the corpus. Existing shards are reused unless named in `rebuild` (or `{"all"}`)."""
    if cfg.data is None:
        raise ValueError("config has no `data` section")
    data = cfg.data
    rebuild = rebuild or set()
    unknown = rebuild - {s.name for s in data.sources} - {"all"}
    if unknown:
        raise ValueError(f"--rebuild names unknown sources: {sorted(unknown)}")
    raw_dir = cfg.paths.resolve("data_raw")
    out_dir = cfg.paths.resolve("data_processed")
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    norm = _Normalizer(data)
    t0 = time.perf_counter()

    log.info("FLORES-200 -> valid/test")
    eval_sizes, excluded = _write_eval_splits(data, raw_dir, out_dir, norm)

    per_source: list[SourceStats] = []
    for priority, src in enumerate(data.sources):
        shard = shards_dir / f"{src.name}.parquet"
        reuse = not ({"all", src.name} & rebuild)
        if reuse and shard.exists() and _stats_path(shard).exists():
            log.info("shard   %s exists, skipping", src.name)
            per_source.append(_load_stats(shard))
            continue
        log.info("source  %s (%s)", src.name, src.kind)
        stats = _build_shard(src, priority, data, raw_dir, shard, norm, excluded)
        _save_stats(shard, stats)
        per_source.append(stats)
        log.info(
            "  %s: read %s, kept %s (%.1f%%) in %ss",
            src.name,
            f"{stats.read:,}",
            f"{stats.kept:,}",
            100 * stats.kept / max(stats.read, 1),
            stats.seconds,
        )

    log.info("dedup + shuffle + split (DuckDB)")
    dedup = _dedup_and_split(shards_dir, out_dir, raw_dir, data.holdout_size, cfg.project.seed)

    card: dict[str, Any] = {
        "project": cfg.project.name,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seconds": round(time.perf_counter() - t0, 1),
        "sources": [asdict(s) for s in per_source],
        "dedup": dedup,
        "splits": {
            "train": dedup["train"],
            "train_holdout": dedup["train_holdout"],
            **eval_sizes,
        },
        "config": data.model_dump(mode="json"),
    }
    (out_dir / "data_card.json").write_text(
        json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    splits = card["splits"]
    log.info(
        "done: train=%s holdout=%s valid=%s test=%s (%.0fs)",
        f"{splits['train']:,}",
        splits["train_holdout"],
        splits["valid"],
        splits["test"],
        card["seconds"],
    )
    return card
