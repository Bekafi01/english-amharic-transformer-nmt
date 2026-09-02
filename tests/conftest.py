"""Shared pytest fixtures for NMT Engine testing."""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
import yaml

from src.nmt_engine.config import AppConfig, load_config


@pytest.fixture
def sample_parallel_pairs() -> list[tuple[str, str]]:
    """Sample English-Amharic parallel sentence pairs for unit testing."""
    return [
        ("Hello world.", "ሰላም ዓለም።"),
        ("How are you today?", "ዛሬ እንዴት ነህ?"),
        ("This is a translation test.", "ይህ የትርጉም ሙከራ ነው።"),
        ("Neural machine translation with Transformers.", "የትራንስፎርመር የነርቭ ማሽን ትርጉም።"),
        ("Good morning, have a great day.", "እንደምን አደሩ፣ መልካም ቀን ይሁንልዎ።"),
    ]


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provides an isolated temporary directory cleaned up after test completion."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def mock_config_dir(temp_dir: Path) -> Path:
    """Creates a temporary config directory populated with valid YAML configs."""
    cfg_dir = temp_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    base_yaml = {
        "project_name": "test-nmt",
        "seed": 42,
        "device": "cpu",
        "num_workers": 0,
        "paths": {
            "raw_data_dir": str(temp_dir / "data" / "raw"),
            "processed_data_dir": str(temp_dir / "data" / "processed"),
            "tokenizer_dir": str(temp_dir / "artifacts" / "tokenizers"),
            "checkpoint_dir": str(temp_dir / "artifacts" / "checkpoints"),
            "onnx_dir": str(temp_dir / "artifacts" / "onnx"),
            "benchmarks_dir": str(temp_dir / "artifacts" / "benchmarks"),
            "tensorboard_dir": str(temp_dir / "artifacts" / "tensorboard"),
        },
        "logging": {
            "level": "DEBUG",
            "log_to_file": False,
            "rich_formatting": False,
        },
    }

    data_yaml = {
        "sources": {
            "mock_corpus": {
                "type": "local",
                "path": str(temp_dir / "mock_data.tsv"),
                "license": "MIT",
                "description": "Mock testing corpus",
                "enabled": True,
            }
        },
        "sampling": {
            "max_samples_per_source": 1000,
            "total_max_samples": 2000,
        },
        "splits": {
            "train": 0.8,
            "val": 0.1,
            "test": 0.1,
        },
        "preprocessing": {
            "normalize_homophones": True,
            "normalize_punctuation": True,
            "normalize_numerals": True,
            "remove_extra_whitespace": True,
            "min_seq_len": 1,
            "max_seq_len": 64,
            "min_len_ratio": 0.2,
            "max_len_ratio": 3.0,
            "deduplicate": True,
        },
        "tokenizer": {
            "type": "bpe",
            "vocab_size": 1000,
            "shared_vocab": True,
            "min_frequency": 1,
            "special_tokens": {
                "pad": "<pad>",
                "unk": "<unk>",
                "bos": "<bos>",
                "eos": "<eos>",
                "mask": "<mask_src>",
            },
        },
    }

    model_yaml = {
        "model": {
            "architecture": "Seq2SeqTransformer",
            "d_model": 128,
            "nhead": 4,
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "dim_feedforward": 256,
            "dropout": 0.1,
            "activation": "relu",
            "layer_norm_eps": 1e-5,
            "norm_type": "pre_ln",
            "positional_encoding": "sinusoidal",
            "max_position_embeddings": 128,
            "tie_weights": True,
        },
        "training": {
            "batch_size": 4,
            "bucket_batching": True,
            "num_buckets": 2,
            "gradient_accumulation_steps": 1,
            "max_epochs": 2,
            "early_stopping_patience": 2,
            "label_smoothing": 0.1,
            "mixed_precision": "no",
            "clip_grad_norm": 1.0,
        },
        "optimizer": {
            "type": "adamw",
            "lr": 0.001,
            "betas": [0.9, 0.98],
            "eps": 1e-9,
            "weight_decay": 0.0001,
        },
        "scheduler": {
            "type": "noam",
            "warmup_steps": 100,
        },
        "inference": {
            "beam_size": 2,
            "length_penalty": 0.6,
            "max_decode_len": 32,
            "repetition_penalty": 1.1,
        },
    }

    with open(cfg_dir / "base_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(base_yaml, f)
    with open(cfg_dir / "data_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f)
    with open(cfg_dir / "model_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(model_yaml, f)

    return cfg_dir


@pytest.fixture
def mock_app_config(mock_config_dir: Path) -> AppConfig:
    """Returns an AppConfig instance populated from temporary test YAML files."""
    return load_config(mock_config_dir)
