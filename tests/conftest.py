"""Shared fixtures: a tiny but realistic workspace (parquet splits + trained tokenizer) and a
training config small enough to run in seconds on CPU."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from amnmt.core.config import Config
from amnmt.tokenization.train import train_from_texts

EN = [
    "The quick brown fox jumps over the lazy dog.",
    "Good morning, how are you today?",
    "Water boils at one hundred degrees.",
    "She reads a book every evening.",
    "Addis Ababa is the capital of Ethiopia.",
    "My brother works at the hospital.",
    "The market opens early on Saturday.",
    "Please close the door quietly.",
]
AM = [
    "ፈጣኑ ቡናማ ቀበሮ ሰነፉን ውሻ ዘሎ አለፈ።",
    "እንደምን አደርክ፣ ዛሬ እንዴት ነህ?",
    "ውሃ በመቶ ዲግሪ ይፈላል።",
    "እሷ በየምሽቱ መጽሐፍ ታነባለች።",
    "አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት።",
    "ወንድሜ በሆስፒታል ውስጥ ይሰራል።",
    "ገበያው ቅዳሜ ማለዳ ይከፈታል።",
    "እባክህ በሩን በጸጥታ ዝጋ።",
]


def make_workspace(root: Path) -> Path:
    """Processed parquet splits + a small trained tokenizer, laid out like a real run."""
    processed = root / "data" / "processed"
    processed.mkdir(parents=True)
    schema = pa.schema(
        [("en", pa.string()), ("am", pa.string()), ("source", pa.string()), ("score", pa.float32())]
    )
    rows = {"en": EN * 10, "am": AM * 10, "source": ["x"] * 80, "score": [None] * 80}
    pq.write_table(pa.table(rows, schema=schema), processed / "train.parquet")
    pq.write_table(pa.table({"en": EN[:4], "am": AM[:4]}), processed / "train_holdout.parquet")
    pq.write_table(
        pa.table({"en": EN[4:], "am": AM[4:], "en_raw": EN[4:], "am_raw": AM[4:]}),
        processed / "valid.parquet",
    )
    tok_dir = root / "artifacts" / "tokenizer"
    tok_dir.mkdir(parents=True)
    train_from_texts(iter((EN + AM) * 5), vocab_size=500, min_frequency=1).save(
        str(tok_dir / "tokenizer.json")
    )
    return root


def training_config(root: Path, **training: object) -> Config:
    return Config.model_validate(
        {
            "project": {"name": "t", "seed": 11, "device": "cpu"},
            "paths": {"root": str(root)},
            "model": {
                "d_model": 32,
                "n_heads": 4,
                "d_ff": 64,
                "n_encoder_layers": 1,
                "n_decoder_layers": 1,
                "dropout": 0.1,
                "max_len": 64,
            },
            "training": {
                "max_len": 40,
                "batch_tokens": 400,
                "peak_lr": 2e-3,
                "warmup_steps": 5,
                "max_steps": 24,
                "log_every": 4,
                "eval_every": 12,
                "save_every": 12,
                "n_samples": 1,
                "num_workers": 0,
                **training,
            },
        }
    )


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_workspace(tmp_path_factory.mktemp("ws"))
