#!/usr/bin/env python3
"""Unified Command-Line Interface for English-Amharic Neural Machine Translation Engine."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.nmt_engine.config import load_data_config  # noqa: E402
from src.nmt_engine.data.collector import DataCollector  # noqa: E402
from src.nmt_engine.utils.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger("nmt_engine.cli")


def run_collect(args: argparse.Namespace) -> int:
    """Executes multi-source parallel corpus collection and serialization."""
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    setup_logging(level=args.log_level, rich_formatting=True)
    logger.info("=" * 70)
    logger.info("PARALLEL CORPUS COLLECTION PIPELINE")
    logger.info("=" * 70)

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path.resolve()}")
        return 1

    data_cfg = load_data_config(config_path)
    collector = DataCollector(config=data_cfg)

    skip_list = [s.strip() for s in args.skip.split(",") if s.strip()] if args.skip else None

    logger.info(f"Target Output Directory: {Path(args.output_dir).resolve()}")
    if args.max_samples:
        logger.info(f"Per-Source Sample Cap  : {args.max_samples:,} pairs")

    df_train, df_bench = collector.collect_all(
        output_dir=args.output_dir,
        max_samples_per_source=args.max_samples,
        chunk_size=args.chunk_size,
        skip_sources=skip_list,
    )

    logger.info("=" * 70)
    logger.info(f"Collection complete. Total Training Pairs: {len(df_train):,}")
    logger.info(f"Evaluation Benchmark Pairs: {len(df_bench):,}")
    logger.info("=" * 70)
    return 0


def run_preprocess(args: argparse.Namespace) -> int:
    """Placeholder for data preprocessing and length filtering."""
    logger.info("Data preprocessing stage will clean and normalize Ethiopic/Latin text.")
    return 0


def run_train_tokenizer(args: argparse.Namespace) -> int:
    """Placeholder for Byte-Pair Encoding subword tokenizer training."""
    logger.info("Tokenizer training stage will train shared 32k BPE vocabulary.")
    return 0


def run_train(args: argparse.Namespace) -> int:
    """Placeholder for Transformer sequence-to-sequence model training."""
    logger.info("Model training stage will initialize and train custom Seq2Seq Transformer.")
    return 0


def run_evaluate(args: argparse.Namespace) -> int:
    """Placeholder for BLEU, chrF++, and TER benchmark evaluation."""
    logger.info("Evaluation stage will compute BLEU, chrF++, and TER on FLORES-200 benchmark.")
    return 0


def run_serve(args: argparse.Namespace) -> int:
    """Placeholder for FastAPI REST serving."""
    logger.info("Serving stage will launch containerized FastAPI REST server.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Constructs top-level CLI argument parser with modular subcommands."""
    parser = argparse.ArgumentParser(
        prog="nmt-cli",
        description="End-to-End English-Amharic Neural Machine Translation Engine CLI",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level (default: INFO)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True, help="Workflow stage to execute")

    # 1. Collect
    p_collect = subparsers.add_parser("collect", help="Acquire and serialize multi-source parallel corpora")
    p_collect.add_argument(
        "--config",
        "-c",
        default="configs/data_config.yaml",
        help="Path to data configuration YAML (default: configs/data_config.yaml)",
    )
    p_collect.add_argument(
        "--output-dir",
        "-o",
        default="data/raw",
        help="Directory to store Parquet datasets and manifest (default: data/raw)",
    )
    p_collect.add_argument(
        "--max-samples",
        "-n",
        type=int,
        default=None,
        help="Optional maximum samples to ingest per source (useful for fast validation)",
    )
    p_collect.add_argument(
        "--chunk-size",
        type=int,
        default=500_000,
        help="Number of rows per streaming Parquet chunk to bound memory (default: 500,000)",
    )
    p_collect.add_argument(
        "--skip",
        type=str,
        default=None,
        help="Comma-separated source identifiers to skip (e.g., 'ccaligned,opus_tanzil')",
    )
    p_collect.set_defaults(func=run_collect)

    # 2. Preprocess
    p_prep = subparsers.add_parser("preprocess", help="Clean and normalize raw parallel corpora")
    p_prep.set_defaults(func=run_preprocess)

    # 3. Train Tokenizer
    p_tok = subparsers.add_parser("train-tokenizer", help="Train shared 32k Byte-Pair Encoding subword vocabulary")
    p_tok.set_defaults(func=run_train_tokenizer)

    # 4. Train Model
    p_train = subparsers.add_parser("train", help="Train custom Transformer Seq2Seq model")
    p_train.set_defaults(func=run_train)

    # 5. Evaluate
    p_eval = subparsers.add_parser("evaluate", help="Compute BLEU, chrF++, TER metrics on gold benchmarks")
    p_eval.set_defaults(func=run_evaluate)

    # 6. Serve
    p_serve = subparsers.add_parser("serve", help="Launch FastAPI translation inference service")
    p_serve.set_defaults(func=run_serve)

    return parser


def main() -> None:
    """CLI entrypoint dispatcher."""
    parser = build_parser()
    args = parser.parse_args()
    exit_code = args.func(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
