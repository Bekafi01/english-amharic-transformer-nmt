"""Data collection, cleaning, tokenization, and dataset loading pipelines for English-Amharic NMT."""

from src.nmt_engine.data.collector import DataCollector, extract_pair

__all__ = ["DataCollector", "extract_pair"]
