"""Benchmark evaluation and metrics calculation suite for English-Amharic NMT."""

from src.nmt_engine.evaluation.evaluator import BenchmarkEvaluator
from src.nmt_engine.evaluation.metrics import (
    compute_all_metrics,
    compute_bleu,
    compute_chrf,
    compute_ter,
)

__all__ = [
    "BenchmarkEvaluator",
    "compute_bleu",
    "compute_chrf",
    "compute_ter",
    "compute_all_metrics",
]
