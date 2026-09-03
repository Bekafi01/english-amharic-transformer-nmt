"""Production-ready parallel corpus collector and multi-source dispatcher for English-Amharic NMT."""

from __future__ import annotations

import gc
import gzip
import io
import re
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from src.nmt_engine.config import DataConfig, SourceItemConfig
from src.nmt_engine.utils.io import ensure_dir, save_json
from src.nmt_engine.utils.logging import get_logger

logger = get_logger("nmt_engine.data.collector")

# Unicode ranges for Ge'ez / Ethiopic Fidel and Latin scripts
ETHIOPIC_REGEX = re.compile(r"[\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF]")
LATIN_REGEX = re.compile(r"[a-zA-Z]")


def extract_pair(row: dict[str, Any]) -> tuple[str | None, str | None]:
    """Extracts English and Amharic sentence pairs using schema lookup with script fallback.

    Args:
        row: Dictionary representation of a single corpus record.

    Returns:
        Tuple of (english_text, amharic_text) or (None, None) if extraction fails.
    """
    en = (
        row.get("english")
        or row.get("English")
        or row.get("eng")
        or row.get("eng_Latn")
        or row.get("sentence_eng_Latn")
    )
    am = (
        row.get("amharic")
        or row.get("Amharic")
        or row.get("amh")
        or row.get("amh_Ethi")
        or row.get("sentence_amh_Ethi")
    )

    # Check nested translation dictionary if present (e.g., OPUS format)
    if not en and not am and "translation" in row and isinstance(row["translation"], dict):
        trans = row["translation"]
        en = trans.get("en") or trans.get("eng") or trans.get("eng_Latn")
        am = trans.get("am") or trans.get("amh") or trans.get("amh_Ethi")

    # Fallback 1: Dynamic key scan
    if not en or not am:
        for k, v in row.items():
            if not isinstance(v, str):
                continue
            k_low = k.lower()
            if not en and ("eng" in k_low or k_low.startswith("en") or "en_" in k_low):
                en = v
            elif not am and ("amh" in k_low or k_low.startswith("am") or "am_" in k_low):
                am = v

    # Fallback 2: Content script detection (Amharic has Fidel, English has Latin)
    if not en or not am:
        for val in row.values():
            if isinstance(val, str) and val.strip():
                if not am and ETHIOPIC_REGEX.search(val):
                    am = val
                elif not en and LATIN_REGEX.search(val) and not ETHIOPIC_REGEX.search(val):
                    en = val

    if en and am:
        en_str, am_str = str(en).strip(), str(am).strip()
        if en_str and am_str:
            return en_str, am_str
    return None, None


class DataCollector:
    """Orchestrates multi-source parallel corpus acquisition and streaming serialization."""

    def __init__(self, config: DataConfig | None = None) -> None:
        """Initializes DataCollector with optional configuration.

        Args:
            config: Validated DataConfig instance. If None, default DataConfig is used.
        """
        self.config = config or DataConfig()

    def collect_source(
        self,
        source_id: str,
        item_config: SourceItemConfig,
        max_samples: int | None = None,
    ) -> pd.DataFrame:
        """Dispatches collection to the appropriate protocol handler.

        Args:
            source_id: Unique source identifier (e.g., 'nllb', 'opus100', 'mt560').
            item_config: Configuration for the source item.
            max_samples: Maximum number of sentence pairs to collect.

        Returns:
            DataFrame with columns ['english', 'amharic', 'source'].
        """
        limit = max_samples or item_config.max_samples
        protocol = item_config.type

        logger.info(f"Ingesting source [{source_id}] via protocol '{protocol}'...")

        if protocol == "gcs_stream":
            if not item_config.url:
                raise ValueError(f"Source [{source_id}] requires 'url' for gcs_stream")
            return self._load_nllb_gcs_stream(
                url=item_config.url,
                source_name=source_id,
                max_samples=limit,
            )
        elif protocol == "url":
            if not item_config.url:
                raise ValueError(f"Source [{source_id}] requires 'url'")
            return self._load_opus_archive(
                name=source_id,
                url=item_config.url,
                max_samples=limit,
            )
        elif protocol == "huggingface":
            if not item_config.path:
                raise ValueError(f"Source [{source_id}] requires 'path'")
            return self._load_hf_dataset(
                path=item_config.path,
                subset=item_config.subset,
                source_name=source_id,
                max_samples=limit,
            )
        elif protocol in ("local_parallel_text", "local"):
            return self._load_local_parallel(
                am_path=item_config.am_path or item_config.path,
                en_path=item_config.en_path,
                source_name=source_id,
                max_samples=limit,
            )
        else:
            logger.warning(f"Unsupported ingestion protocol '{protocol}' for source [{source_id}]")
            return pd.DataFrame(columns=["english", "amharic", "source"])

    def _load_nllb_gcs_stream(
        self,
        url: str,
        source_name: str = "nllb",
        max_samples: int | None = None,
    ) -> pd.DataFrame:
        """Streams gzipped TSV parallel pairs directly from Google Cloud Storage.

        Args:
            url: GCS bucket archive URL.
            source_name: Name identifier for the source.
            max_samples: Optional sample cap.

        Returns:
            DataFrame containing extracted sentence pairs.
        """
        t0 = time.time()
        logger.info(f"Connecting to GCS stream: {url}")

        try:
            response = requests.get(url, stream=True, timeout=60)
            if response.status_code != 200:
                logger.error(f"HTTP status {response.status_code} received from GCS stream")
                return pd.DataFrame(columns=["english", "amharic", "source"])

            pairs: list[tuple[str, str]] = []
            with gzip.GzipFile(fileobj=response.raw) as gz:
                for line in gz:
                    decoded = line.decode("utf-8", errors="replace").strip()
                    if not decoded:
                        continue
                    parts = decoded.split("\t")
                    if len(parts) >= 2:
                        am_text, en_text = parts[0].strip(), parts[1].strip()
                        # Verify orientation by script (Fidel vs Latin)
                        if not ETHIOPIC_REGEX.search(am_text) and ETHIOPIC_REGEX.search(en_text):
                            am_text, en_text = en_text, am_text

                        if am_text and en_text:
                            pairs.append((en_text, am_text))

                    if max_samples and len(pairs) >= max_samples:
                        break

            df = pd.DataFrame(pairs, columns=["english", "amharic"])
            df["source"] = source_name
            elapsed = time.time() - t0
            rate = len(df) / max(elapsed, 0.1)
            logger.info(
                f"Loaded {len(df):,} pairs from {source_name} in {elapsed:.1f}s ({rate:,.0f} pairs/sec)"
            )
            return df
        except Exception as e:
            logger.error(f"Failed to stream from GCS ({url}): {e}")
            return pd.DataFrame(columns=["english", "amharic", "source"])

    def _load_opus_archive(
        self,
        name: str,
        url: str,
        max_samples: int | None = None,
    ) -> pd.DataFrame:
        """Downloads and extracts parallel text directly from an OPUS zip archive.

        Args:
            name: Archive identifier.
            url: Download URL for the zip archive.
            max_samples: Optional sample cap.

        Returns:
            DataFrame containing extracted sentence pairs.
        """
        t0 = time.time()
        logger.info(f"Downloading OPUS archive '{name}' from {url}...")
        try:
            response = requests.get(url, stream=True, timeout=60)
            if response.status_code != 200:
                logger.error(f"HTTP status {response.status_code} from {url}")
                return pd.DataFrame(columns=["english", "amharic", "source"])

            z = zipfile.ZipFile(io.BytesIO(response.content))
            files = z.namelist()

            en_candidates = [f for f in files if f.endswith(".en") or ".en." in f]
            am_candidates = [f for f in files if f.endswith(".am") or ".am." in f]

            if not en_candidates or not am_candidates:
                logger.error(f"Could not locate aligned .en and .am in archive: {files}")
                return pd.DataFrame(columns=["english", "amharic", "source"])

            with z.open(en_candidates[0]) as f_en, z.open(am_candidates[0]) as f_am:
                en_lines = io.TextIOWrapper(f_en, encoding="utf-8", errors="replace").readlines()
                am_lines = io.TextIOWrapper(f_am, encoding="utf-8", errors="replace").readlines()

            total_lines = min(len(en_lines), len(am_lines))
            limit = max_samples if max_samples else total_lines
            pairs: list[tuple[str, str]] = []

            for i in range(min(total_lines, limit)):
                e = en_lines[i].strip()
                a = am_lines[i].strip()
                if e and a:
                    pairs.append((e, a))

            df = pd.DataFrame(pairs, columns=["english", "amharic"])
            df["source"] = f"opus_{name}"
            elapsed = time.time() - t0
            logger.info(f"Loaded {len(df):,} pairs from OPUS {name} in {elapsed:.1f}s")
            return df
        except Exception as e:
            logger.error(f"Failed to load OPUS archive '{name}': {e}")
            return pd.DataFrame(columns=["english", "amharic", "source"])

    def _load_hf_dataset(
        self,
        path: str,
        subset: str | None = None,
        source_name: str = "hf",
        max_samples: int | None = None,
    ) -> pd.DataFrame:
        """Loads parallel dataset from Hugging Face hub with script-based extraction.

        Args:
            path: Hugging Face dataset identifier.
            subset: Optional subset configuration.
            source_name: Identifier tag for source column.
            max_samples: Optional sample cap.

        Returns:
            DataFrame containing extracted sentence pairs.
        """
        from datasets import load_dataset

        t0 = time.time()
        logger.info(f"Loading HuggingFace dataset: {path} (subset: {subset})")
        try:
            ds = load_dataset(path, subset) if subset else load_dataset(path)
            pairs: list[dict[str, Any]] = []

            splits = list(ds.keys()) if hasattr(ds, "keys") else ["train"]
            for split in splits:
                split_data = ds[split]
                for row in split_data:
                    en, am = extract_pair(row)
                    if en and am:
                        pairs.append({
                            "english": en,
                            "amharic": am,
                            "source": source_name,
                            "split": split,
                        })
                    if max_samples and len(pairs) >= max_samples:
                        break
                if max_samples and len(pairs) >= max_samples:
                    break

            df = pd.DataFrame(pairs)
            if "split" not in df.columns:
                df["split"] = "train"
            elapsed = time.time() - t0
            logger.info(f"Loaded {len(df):,} pairs from HuggingFace {path} in {elapsed:.1f}s")
            return df
        except Exception as e:
            logger.error(f"Failed to load HuggingFace dataset {path}: {e}")
            return pd.DataFrame(columns=["english", "amharic", "source", "split"])

    def _load_local_parallel(
        self,
        am_path: str | Path | None,
        en_path: str | Path | None,
        source_name: str = "local_custom",
        max_samples: int | None = None,
    ) -> pd.DataFrame:
        """Ingests line-aligned text files from local or mounted cloud storage.

        Args:
            am_path: Path to Amharic text file.
            en_path: Path to English text file.
            source_name: Name identifier for source column.
            max_samples: Optional sample cap.

        Returns:
            DataFrame containing extracted sentence pairs.
        """
        if not am_path or not en_path:
            logger.info("Local parallel paths not provided. Skipping.")
            return pd.DataFrame(columns=["english", "amharic", "source"])

        target_am = Path(am_path)
        target_en = Path(en_path)

        if not (target_am.is_file() and target_en.is_file()):
            logger.info(f"Files not found at {target_am} and {target_en}. Skipping.")
            return pd.DataFrame(columns=["english", "amharic", "source"])

        t0 = time.time()
        logger.info(f"Reading parallel text files from: {target_am.parent}")
        try:
            with open(target_am, encoding="utf-8", errors="replace") as f_am:
                am_lines = f_am.readlines()
            with open(target_en, encoding="utf-8", errors="replace") as f_en:
                en_lines = f_en.readlines()

            total = min(len(am_lines), len(en_lines))
            limit = max_samples if max_samples else total
            pairs: list[tuple[str, str]] = []

            for i in range(min(total, limit)):
                a, e = am_lines[i].strip(), en_lines[i].strip()
                if a and e:
                    pairs.append((e, a))

            df = pd.DataFrame(pairs, columns=["english", "amharic"])
            df["source"] = source_name
            elapsed = time.time() - t0
            logger.info(f"Loaded {len(df):,} verified pairs from {target_am.name} in {elapsed:.1f}s")
            return df
        except Exception as e:
            logger.error(f"Error reading local parallel text: {e}")
            return pd.DataFrame(columns=["english", "amharic", "source"])

    def collect_all(
        self,
        output_dir: str | Path = "data/raw",
        max_samples_per_source: int | None = None,
        chunk_size: int = 500_000,
        skip_sources: list[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Runs the complete multi-source collection, validation, and chunked serialization.

        Args:
            output_dir: Target directory for Parquet artifacts and metadata.
            max_samples_per_source: Optional per-source sample cap.
            chunk_size: Streaming batch size for memory-safe Parquet writing.
            skip_sources: Optional list of source IDs to skip.

        Returns:
            Tuple of (df_training_corpus, df_gold_benchmark).
        """
        start_time = time.time()
        out_path = ensure_dir(output_dir)
        skip = set(skip_sources or [])

        training_dfs: list[pd.DataFrame] = []
        df_benchmark = pd.DataFrame(columns=["english", "amharic", "source", "split"])

        for source_id, item_cfg in self.config.sources.items():
            if not item_cfg.enabled or source_id in skip:
                logger.info(f"Skipping source [{source_id}] (enabled={item_cfg.enabled})")
                continue

            # Special handling for evaluation benchmark (FLORES-200)
            if source_id == "flores":
                df_bench = self.collect_source(source_id, item_cfg, max_samples=max_samples_per_source)
                if len(df_bench) > 0:
                    df_benchmark = df_bench
                continue

            df_src = self.collect_source(source_id, item_cfg, max_samples=max_samples_per_source)
            if len(df_src) > 0:
                training_dfs.append(df_src)

        # Consolidate training pool
        if training_dfs:
            df_train = pd.concat(training_dfs, ignore_index=True)
            df_train["source"] = df_train["source"].astype("category")
        else:
            df_train = pd.DataFrame(columns=["english", "amharic", "source"])

        total_elapsed = time.time() - start_time
        logger.info(f"Acquisition complete in {total_elapsed/60:.2f} minutes")
        logger.info(f"Total training pairs collected: {len(df_train):,}")
        logger.info(f"Total gold evaluation pairs  : {len(df_benchmark):,}")

        # Vectorized script integrity checks
        self._verify_integrity(df_train)

        # High-performance chunked serialization
        parquet_path = out_path / "raw_parallel_corpus.parquet"
        self.save_to_parquet_chunked(df_train, parquet_path, chunk_size=chunk_size)

        # Export sample TSV
        sample_tsv = out_path / "raw_parallel_corpus_sample_100k.tsv"
        df_train.head(100_000).to_csv(sample_tsv, sep="\t", index=False)

        # Export benchmark Parquet
        if len(df_benchmark) > 0:
            bench_parquet = out_path / "flores200_benchmark.parquet"
            df_benchmark.to_parquet(bench_parquet, engine="pyarrow", compression="snappy", index=False)

        # Export manifest
        self.export_manifest(
            output_dir=out_path,
            df_train=df_train,
            df_benchmark=df_benchmark,
            elapsed_seconds=total_elapsed,
        )

        return df_train, df_benchmark

    def _verify_integrity(self, df: pd.DataFrame) -> dict[str, Any]:
        """Runs vectorized script compliance and length statistics."""
        total = len(df)
        if total == 0:
            return {}

        en_valid = df["english"].notna() & (df["english"].astype(str).str.strip() != "")
        am_valid = df["amharic"].notna() & (df["amharic"].astype(str).str.strip() != "")
        valid_count = int((en_valid & am_valid).sum())

        sample_n = min(100_000, total)
        sample = df.sample(sample_n, random_state=42)
        fidel_pct = float(sample["amharic"].apply(lambda x: bool(ETHIOPIC_REGEX.search(str(x)))).mean())
        latin_pct = float(sample["english"].apply(lambda x: bool(LATIN_REGEX.search(str(x)))).mean())

        logger.info(f"Parallel non-empty pairs: {valid_count:,} / {total:,} ({(valid_count/total)*100:.2f}%)")
        logger.info(f"Script compliance (sample n={sample_n:,}): Amharic={fidel_pct*100:.2f}%, English={latin_pct*100:.2f}%")

        return {
            "total_pairs": total,
            "non_empty_pairs": valid_count,
            "fidel_compliance_pct": round(fidel_pct * 100, 2),
            "latin_compliance_pct": round(latin_pct * 100, 2),
        }

    def save_to_parquet_chunked(
        self,
        df: pd.DataFrame,
        output_path: str | Path,
        chunk_size: int = 500_000,
    ) -> Path:
        """Serializes large DataFrame to Parquet in streaming chunks to eliminate RAM spikes.

        Args:
            df: DataFrame to serialize.
            output_path: Destination path for Parquet file.
            chunk_size: Number of rows per PyArrow chunk.

        Returns:
            Path to saved Parquet file.
        """
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        schema = pa.schema([
            ("english", pa.string()),
            ("amharic", pa.string()),
            ("source", pa.string()),
        ])

        total_rows = len(df)
        t0 = time.time()
        logger.info(f"Serializing {total_rows:,} pairs to {p.name} (chunk_size={chunk_size:,})...")

        with pq.ParquetWriter(p, schema, compression="snappy") as writer:
            for start_idx in range(0, total_rows, chunk_size):
                end_idx = min(start_idx + chunk_size, total_rows)
                chunk_df = df.iloc[start_idx:end_idx]
                table = pa.Table.from_pandas(chunk_df, schema=schema)
                writer.write_table(table)
                del table, chunk_df
                gc.collect()

        size_mb = p.stat().st_size / (1024**2)
        elapsed = time.time() - t0
        rate = total_rows / max(elapsed, 0.01)
        logger.info(f"Saved {p.name} ({size_mb:.2f} MB in {elapsed:.1f}s, {rate:,.0f} rows/sec)")
        return p

    def export_manifest(
        self,
        output_dir: str | Path,
        df_train: pd.DataFrame,
        df_benchmark: pd.DataFrame,
        elapsed_seconds: float,
    ) -> Path:
        """Generates provenance manifest documenting corpus metrics and source distribution.

        Args:
            output_dir: Directory to store collection_manifest.json.
            df_train: Training DataFrame.
            df_benchmark: Evaluation benchmark DataFrame.
            elapsed_seconds: Elapsed collection time.

        Returns:
            Path to manifest file.
        """
        p = Path(output_dir) / "collection_manifest.json"
        manifest = {
            "task": "English-Amharic Parallel Corpus Acquisition",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_minutes": round(elapsed_seconds / 60, 2),
            "total_parallel_pairs": int(len(df_train)),
            "sources": df_train["source"].value_counts().to_dict() if len(df_train) > 0 else {},
            "benchmark": {
                "flores200_pairs": int(len(df_benchmark)),
            },
        }
        save_json(manifest, p)
        logger.info(f"Collection manifest exported: {p}")
        return p
