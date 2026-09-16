"""Unit tests for evaluation metrics (BLEU, chrF++, TER) and BenchmarkEvaluator."""

from pathlib import Path

import pandas as pd
import pytest

from src.nmt_engine.data.tokenizer import JointBpeTokenizer
from src.nmt_engine.evaluation.evaluator import BenchmarkEvaluator
from src.nmt_engine.evaluation.metrics import (
    compute_all_metrics,
    compute_bleu,
    compute_chrf,
    compute_ter,
)
from src.nmt_engine.inference.translator import Translator
from src.nmt_engine.models.transformer import Seq2SeqTransformer


def test_metrics_identical_strings() -> None:
    hyps = ["The Prime Minister arrived in Addis Ababa.", "Peace and cooperation."]
    refs = ["The Prime Minister arrived in Addis Ababa.", "Peace and cooperation."]

    bleu = compute_bleu(hyps, refs)
    chrf = compute_chrf(hyps, refs)
    ter = compute_ter(hyps, refs)

    assert bleu >= 99.0
    assert chrf >= 99.0
    assert ter <= 1.0


def test_metrics_disjoint_strings() -> None:
    hyps = ["apple orange banana"]
    refs = ["rocket spaceship satellite"]

    bleu = compute_bleu(hyps, refs)
    ter = compute_ter(hyps, refs)

    assert bleu == 0.0
    assert ter >= 90.0


def test_compute_all_metrics() -> None:
    hyps = ["Hello world"]
    refs = ["Hello earth"]

    res = compute_all_metrics(hyps, refs)
    assert "bleu" in res
    assert "chrf2" in res
    assert "ter" in res
    assert isinstance(res["bleu"], float)
    assert isinstance(res["chrf2"], float)
    assert isinstance(res["ter"], float)


@pytest.fixture
def mock_evaluator(tmp_path: Path):
    texts = ["hello", "world", "ሰላም", "ዓለም"]
    tok = JointBpeTokenizer()
    tok.train_from_iterator(iter([[t] for t in texts]), vocab_size=60, min_frequency=1)

    model = Seq2SeqTransformer(
        vocab_size=tok.vocab_size,
        d_model=32,
        n_heads=2,
        n_layers=1,
        d_ff=64,
        pad_id=tok.pad_id,
        tie_weights=True,
    )
    model.eval()

    translator = Translator(model=model, tokenizer=tok, device="cpu", max_len=10)
    return BenchmarkEvaluator(translator=translator)


def test_evaluator_evaluate_dataset(mock_evaluator) -> None:
    df = pd.DataFrame(
        {
            "english": ["hello", "world"],
            "amharic": ["ሰላም", "ዓለም"],
        }
    )

    res = mock_evaluator.evaluate_dataset(
        df=df,
        source_col="english",
        target_col="amharic",
        direction="en2am",
        method="greedy",
    )

    assert res["direction"] == "en2am"
    assert res["num_samples"] == 2
    assert "metrics" in res
    assert len(res["samples"]) == 2


def test_evaluator_evaluate_flores(mock_evaluator, tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "english": ["hello", "world"],
            "amharic": ["ሰላም", "ዓለም"],
        }
    )
    parquet_path = tmp_path / "flores200_mock.parquet"
    df.to_parquet(parquet_path, index=False)

    out_dir = tmp_path / "reports"
    res = mock_evaluator.evaluate_flores(
        benchmark_parquet=parquet_path,
        method="greedy",
        output_dir=out_dir,
    )

    assert "en2am" in res
    assert "am2en" in res
    assert (out_dir / "flores200_metrics.json").exists()
    assert (out_dir / "flores200_benchmark_report.md").exists()
