"""Unit tests for parallel data collector and script-based extraction."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.nmt_engine.data.collector import (
    ETHIOPIC_REGEX,
    LATIN_REGEX,
    DataCollector,
    extract_pair,
)


def test_script_regex_matching():
    """Verifies Unicode regex accurately separates Ethiopic and Latin scripts."""
    assert ETHIOPIC_REGEX.search("ሰላም ዓለም") is not None
    assert ETHIOPIC_REGEX.search("12345 Hello") is None

    assert LATIN_REGEX.search("Hello world") is not None
    assert LATIN_REGEX.search("ሰላም ዓለም") is None


def test_extract_pair_standard_keys():
    """Extracts pairs with standard 'english' and 'amharic' keys."""
    row = {"english": "Hello world", "amharic": "ሰላም ዓለም"}
    en, am = extract_pair(row)
    assert en == "Hello world"
    assert am == "ሰላም ዓለም"


def test_extract_pair_nested_translation():
    """Extracts pairs from nested translation dictionaries (OPUS style)."""
    row = {"translation": {"en": "Good morning", "am": "እንደምን አደሩ"}}
    en, am = extract_pair(row)
    assert en == "Good morning"
    assert am == "እንደምን አደሩ"


def test_extract_pair_script_fallback():
    """Extracts pairs using linguistic script fallback on arbitrary/non-standard column names."""
    row = {"col_x": "Peace to all people.", "col_y": "ሰላም ለሰው ልጆች ሁሉ።"}
    en, am = extract_pair(row)
    assert en == "Peace to all people."
    assert am == "ሰላም ለሰው ልጆች ሁሉ።"


def test_extract_pair_empty_invalid():
    """Returns (None, None) for empty, whitespace, or non-parallel entries."""
    assert extract_pair({}) == (None, None)
    assert extract_pair({"english": "", "amharic": ""}) == (None, None)
    assert extract_pair({"col": "123456"}) == (None, None)


def test_load_local_parallel(temp_dir: Path):
    """Verifies reading local line-aligned text files."""
    am_file = temp_dir / "test_am.txt"
    en_file = temp_dir / "test_en.txt"

    am_file.write_text("ሰላም\nእንደምን አደሩ\n", encoding="utf-8")
    en_file.write_text("Hello\nGood morning\n", encoding="utf-8")

    collector = DataCollector()
    df = collector._load_local_parallel(
        am_path=am_file,
        en_path=en_file,
        source_name="test_local",
    )

    assert len(df) == 2
    assert list(df.columns) == ["english", "amharic", "source"]
    assert df.iloc[0]["english"] == "Hello"
    assert df.iloc[0]["amharic"] == "ሰላም"
    assert df.iloc[0]["source"] == "test_local"


def test_save_to_parquet_chunked(temp_dir: Path):
    """Verifies chunked PyArrow serialization and readback."""
    df = pd.DataFrame({
        "english": [f"Sentence {i}" for i in range(100)],
        "amharic": [f"ዓረፍተ ነገር {i}" for i in range(100)],
        "source": ["mock"] * 100,
    })

    parquet_path = temp_dir / "chunked_output.parquet"
    collector = DataCollector()
    saved_path = collector.save_to_parquet_chunked(df, parquet_path, chunk_size=25)

    assert saved_path.exists()
    df_loaded = pd.read_parquet(saved_path)
    assert len(df_loaded) == 100
    assert list(df_loaded.columns) == ["english", "amharic", "source"]
    assert df_loaded.iloc[0]["english"] == "Sentence 0"


def test_export_manifest(temp_dir: Path):
    """Verifies metadata manifest generation."""
    collector = DataCollector()
    df_train = pd.DataFrame({
        "english": ["Hi", "Bye"],
        "amharic": ["ሰላም", "ደህና ሁን"],
        "source": ["src_a", "src_b"],
    })
    df_bench = pd.DataFrame({
        "english": ["Test"],
        "amharic": ["ሙከራ"],
        "source": ["flores"],
        "split": ["dev"],
    })

    manifest_path = collector.export_manifest(
        output_dir=temp_dir,
        df_train=df_train,
        df_benchmark=df_bench,
        elapsed_seconds=12.5,
    )

    assert manifest_path.exists()
    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["total_parallel_pairs"] == 2
    assert data["benchmark"]["flores200_pairs"] == 1
    assert "src_a" in data["sources"]
