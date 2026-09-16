#!/usr/bin/env python3
"""Unified Command-Line Interface for English-Amharic Neural Machine Translation Engine."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from functools import partial  # noqa: E402

import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.optim as optim  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.nmt_engine.config import load_data_config  # noqa: E402
from src.nmt_engine.data.collector import DataCollector  # noqa: E402
from src.nmt_engine.data.dataset import (  # noqa: E402
    BucketBatchSampler,
    TranslationDataset,
    pad_collate_fn,
)
from src.nmt_engine.data.preprocessor import DataPreprocessor  # noqa: E402
from src.nmt_engine.data.tokenizer import JointBpeTokenizer  # noqa: E402
from src.nmt_engine.evaluation.evaluator import BenchmarkEvaluator  # noqa: E402
from src.nmt_engine.inference.translator import Translator  # noqa: E402
from src.nmt_engine.models.transformer import Seq2SeqTransformer  # noqa: E402
from src.nmt_engine.training.loss import LabelSmoothingLoss  # noqa: E402
from src.nmt_engine.training.scheduler import NoamScheduler  # noqa: E402
from src.nmt_engine.training.trainer import TransformerTrainer  # noqa: E402
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
    """Executes end-to-end linguistic normalization, quality filtering, and deduplication."""
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    setup_logging(level=args.log_level, rich_formatting=True)
    logger.info("=" * 70)
    logger.info("LINGUISTIC PREPROCESSING & DEDUPLICATION PIPELINE")
    logger.info("=" * 70)

    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path.resolve()}")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading raw parallel corpus from: {input_path}")
    if input_path.suffix == ".parquet":
        df_raw = pd.read_parquet(input_path)
    elif input_path.suffix in [".tsv", ".csv"]:
        sep = "\t" if input_path.suffix == ".tsv" else ","
        df_raw = pd.read_csv(input_path, sep=sep)
    else:
        logger.error(f"Unsupported file format: {input_path.suffix}")
        return 1

    logger.info(f"Loaded {len(df_raw):,} raw sentence pairs.")

    preprocessor = DataPreprocessor(
        min_seq_len=args.min_len,
        max_seq_len=args.max_len,
        min_len_ratio=args.min_ratio,
        max_len_ratio=args.max_ratio,
        minhash_threshold=args.minhash_threshold,
        seed=args.seed,
    )

    df_train, df_val, df_test = preprocessor.process(
        df_raw=df_raw,
        run_minhash=not args.no_minhash,
    )

    # Save partitioned parquet splits
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    test_path = output_dir / "test.parquet"

    df_train.to_parquet(train_path, index=False)
    df_val.to_parquet(val_path, index=False)
    df_test.to_parquet(test_path, index=False)

    logger.info("=" * 70)
    logger.info("Preprocessing complete. Cleaned partitions saved:")
    logger.info(f"  • Train: {train_path} ({len(df_train):,} pairs)")
    logger.info(f"  • Val  : {val_path} ({len(df_val):,} pairs)")
    logger.info(f"  • Test : {test_path} ({len(df_test):,} pairs)")
    logger.info("=" * 70)
    return 0


def run_train_tokenizer(args: argparse.Namespace) -> int:
    """Trains shared 32k Byte-Pair Encoding subword tokenizer and serializes artifacts."""
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    setup_logging(level=args.log_level, rich_formatting=True)
    logger.info("=" * 70)
    logger.info("JOINT BYTE-PAIR ENCODING (BPE) TOKENIZER TRAINING")
    logger.info("=" * 70)

    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"Input training data not found: {input_path.resolve()}")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading training data from: {input_path}")
    if input_path.suffix == ".parquet":
        df_train = pd.read_parquet(input_path)
    else:
        df_train = pd.read_csv(input_path)

    # In-memory batch iterator over both languages
    def text_iterator(batch_size: int = 5000):
        for i in range(0, len(df_train), batch_size):
            am_batch = df_train["amharic"].iloc[i : i + batch_size].astype(str).tolist()
            en_batch = df_train["english"].iloc[i : i + batch_size].astype(str).tolist()
            yield am_batch + en_batch

    tokenizer = JointBpeTokenizer()
    tokenizer.train_from_iterator(
        iterator=text_iterator(),
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )

    model_path = output_dir / "joint_bpe_32k.json"
    tokenizer.save(model_path)
    logger.info(f"Saved trained tokenizer model to: {model_path}")

    # Optionally tokenize datasets and export bidirectional sequences
    if args.tokenize_datasets:
        tokenized_dir = Path(args.tokenized_output_dir)
        tokenized_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Generating bidirectional tokenized datasets in: {tokenized_dir}")

        train_bidir = tokenizer.build_bidirectional_dataset(df_train, max_seq_len=args.max_seq_len)
        train_out = tokenized_dir / "train_ids.parquet"
        train_bidir.to_parquet(train_out, index=False)
        logger.info(
            f"Exported {len(train_bidir):,} bidirectional training sequences to {train_out}"
        )

        # If val and test splits exist in parent directory, tokenize them too
        cleaned_dir = input_path.parent
        val_in = cleaned_dir / "val.parquet"
        test_in = cleaned_dir / "test.parquet"

        if val_in.exists():
            df_val = pd.read_parquet(val_in)
            val_bidir = tokenizer.build_bidirectional_dataset(df_val, max_seq_len=args.max_seq_len)
            val_out = tokenized_dir / "val_ids.parquet"
            val_bidir.to_parquet(val_out, index=False)
            logger.info(
                f"Exported {len(val_bidir):,} bidirectional validation sequences to {val_out}"
            )

        if test_in.exists():
            df_test = pd.read_parquet(test_in)
            test_bidir = tokenizer.build_bidirectional_dataset(
                df_test, max_seq_len=args.max_seq_len
            )
            test_out = tokenized_dir / "test_ids.parquet"
            test_bidir.to_parquet(test_out, index=False)
            logger.info(f"Exported {len(test_bidir):,} bidirectional test sequences to {test_out}")

    logger.info("=" * 70)
    logger.info("Tokenizer training and dataset serialization complete.")
    logger.info("=" * 70)
    return 0


def run_train(args: argparse.Namespace) -> int:
    """Executes full Sequence-to-Sequence Transformer model training."""
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    setup_logging(level=args.log_level, rich_formatting=True)
    logger.info("=" * 70)
    logger.info("CUSTOM SEQ2SEQ TRANSFORMER TRAINING PIPELINE")
    logger.info("=" * 70)

    train_path = Path(args.train_file)
    val_path = Path(args.val_file)

    if not train_path.exists():
        logger.error(f"Training dataset not found: {train_path.resolve()}")
        return 1
    if not val_path.exists():
        logger.error(f"Validation dataset not found: {val_path.resolve()}")
        return 1

    # Device selection
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif (
        args.device == "mps"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        device = torch.device("mps")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Target Compute Device: {device}")

    logger.info(f"Loading training data from: {train_path}")
    ds_train = TranslationDataset(train_path)
    logger.info(f"Loading validation data from: {val_path}")
    ds_val = TranslationDataset(val_path)
    logger.info(f"Dataset summary: Train = {len(ds_train):,} pairs, Val = {len(ds_val):,} pairs")

    collate_fn = partial(pad_collate_fn, pad_id=args.pad_id)

    if args.bucket_batching:
        logger.info(
            f"Configuring BucketBatchSampler ({args.num_buckets} buckets, batch_size={args.batch_size})"
        )
        train_sampler = BucketBatchSampler(
            lengths=ds_train.lengths,
            batch_size=args.batch_size,
            num_buckets=args.num_buckets,
            shuffle=True,
            seed=args.seed,
        )
        val_sampler = BucketBatchSampler(
            lengths=ds_val.lengths,
            batch_size=args.batch_size,
            num_buckets=args.num_buckets,
            shuffle=False,
            seed=args.seed,
        )
        train_loader = DataLoader(ds_train, batch_sampler=train_sampler, collate_fn=collate_fn)
        val_loader = DataLoader(ds_val, batch_sampler=val_sampler, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(
            ds_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
        )
        val_loader = DataLoader(
            ds_val, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
        )

    # Initialize model
    logger.info("Instantiating Seq2SeqTransformer...")
    model = Seq2SeqTransformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=args.max_len,
        pad_id=args.pad_id,
        tie_weights=not args.no_tie_weights,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model Total Parameters: {total_params:,}")

    # Loss, Optimizer, Scheduler
    criterion = LabelSmoothingLoss(
        vocab_size=args.vocab_size, pad_id=args.pad_id, smoothing=args.label_smoothing
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )
    scheduler = NoamScheduler(optimizer, d_model=args.d_model, warmup_steps=args.warmup_steps)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    trainer = TransformerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        grad_accum_steps=args.grad_accum_steps,
        clip_grad_norm=args.clip_grad_norm,
        checkpoint_dir=ckpt_dir,
        mixed_precision=args.mixed_precision,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train(max_epochs=args.epochs, patience=args.patience)
    logger.info("=" * 70)
    logger.info("Model training pipeline completed successfully.")
    logger.info("=" * 70)
    return 0


def run_evaluate(args: argparse.Namespace) -> int:
    """Executes gold-standard benchmark evaluation (SacreBLEU, chrF++, TER)."""
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    setup_logging(level=args.log_level, rich_formatting=True)
    logger.info("=" * 70)
    logger.info("BENCHMARK EVALUATION PIPELINE (FLORES-200 / TEST SPLIT)")
    logger.info("=" * 70)

    model_path = Path(args.model_path)
    tok_path = Path(args.tokenizer_path)
    bench_path = Path(args.benchmark_file)

    if not model_path.exists():
        logger.error(f"Model checkpoint not found: {model_path.resolve()}")
        return 1
    if not tok_path.exists():
        logger.error(f"Tokenizer model artifact not found: {tok_path.resolve()}")
        return 1
    if not bench_path.exists():
        logger.error(f"Benchmark dataset not found: {bench_path.resolve()}")
        return 1

    translator = Translator(
        model_path=model_path,
        tokenizer_path=tok_path,
        device=args.device,
        beam_size=args.beam_size,
        length_penalty=args.length_penalty,
        repetition_penalty=args.repetition_penalty,
        max_len=args.max_len,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
    )

    evaluator = BenchmarkEvaluator(translator=translator)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = evaluator.evaluate_flores(
        benchmark_parquet=bench_path,
        max_samples=args.max_samples,
        method=args.method,
        output_dir=out_dir,
    )

    en2am = summary["en2am"]["metrics"]
    am2en = summary["am2en"]["metrics"]

    logger.info("=" * 70)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 70)
    logger.info(
        f"• EN -> AM: BLEU = {en2am['bleu']:.2f} | chrF++ = {en2am['chrf2']:.2f} | TER = {en2am['ter']:.2f}"
    )
    logger.info(
        f"• AM -> EN: BLEU = {am2en['bleu']:.2f} | chrF++ = {am2en['chrf2']:.2f} | TER = {am2en['ter']:.2f}"
    )
    logger.info(f"Full markdown and JSON reports saved to: {out_dir.resolve()}")
    logger.info("=" * 70)
    return 0


def run_serve(args: argparse.Namespace) -> int:
    """Launches the production Uvicorn ASGI server hosting the FastAPI NMT translation application."""
    import uvicorn

    setup_logging(level=args.log_level, rich_formatting=True)
    logger.info("=" * 70)
    logger.info("ENGLISH-AMHARIC NMT INFERENCE REST SERVICE")
    logger.info("=" * 70)
    logger.info(f"• Host Interface : {args.host}")
    logger.info(f"• Port Number    : {args.port}")
    logger.info(f"• Worker Procs   : {args.workers}")
    logger.info(f"• Model Checkpoint: {Path(args.model_path).resolve()}")
    logger.info(f"• Tokenizer Model : {Path(args.tokenizer_path).resolve()}")
    logger.info(f"• Compute Device  : {args.device}")
    logger.info("=" * 70)

    # Export environment variables for the FastAPI lifespan loader
    os.environ["NMT_MODEL_PATH"] = str(args.model_path)
    os.environ["NMT_TOKENIZER_PATH"] = str(args.tokenizer_path)
    os.environ["NMT_DEVICE"] = str(args.device)

    uvicorn.run(
        "src.nmt_engine.serving.app:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level=args.log_level.lower(),
    )
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

    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Workflow stage to execute"
    )

    # 1. Collect
    p_collect = subparsers.add_parser(
        "collect", help="Acquire and serialize multi-source parallel corpora"
    )
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
    p_prep = subparsers.add_parser(
        "preprocess", help="Clean, normalize, and partition raw parallel corpora"
    )
    p_prep.add_argument(
        "--input-file",
        "-i",
        default="data/raw/raw_parallel_corpus.parquet",
        help="Path to raw parallel corpus Parquet (default: data/raw/raw_parallel_corpus.parquet)",
    )
    p_prep.add_argument(
        "--output-dir",
        "-o",
        default="data/processed/cleaned",
        help="Directory to store cleaned Parquet splits (default: data/processed/cleaned)",
    )
    p_prep.add_argument(
        "--min-len",
        type=int,
        default=2,
        help="Minimum sequence length in words (default: 2)",
    )
    p_prep.add_argument(
        "--max-len",
        type=int,
        default=128,
        help="Maximum sequence length in words (default: 128)",
    )
    p_prep.add_argument(
        "--min-ratio",
        type=float,
        default=0.30,
        help="Minimum length ratio AM/EN (default: 0.30)",
    )
    p_prep.add_argument(
        "--max-ratio",
        type=float,
        default=2.50,
        help="Maximum length ratio AM/EN (default: 2.50)",
    )
    p_prep.add_argument(
        "--no-minhash",
        action="store_true",
        help="Disable MinHash LSH near-duplicate clustering for faster execution",
    )
    p_prep.add_argument(
        "--minhash-threshold",
        type=float,
        default=0.90,
        help="Jaccard similarity threshold for MinHash LSH clustering (default: 0.90)",
    )
    p_prep.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic dataset partitioning (default: 42)",
    )
    p_prep.set_defaults(func=run_preprocess)

    # 3. Train Tokenizer
    p_tok = subparsers.add_parser(
        "train-tokenizer", help="Train shared 32k Byte-Pair Encoding subword vocabulary"
    )
    p_tok.add_argument(
        "--input-file",
        "-i",
        default="data/processed/cleaned/train.parquet",
        help="Path to training data Parquet file (default: data/processed/cleaned/train.parquet)",
    )
    p_tok.add_argument(
        "--output-dir",
        "-o",
        default="artifacts/tokenizers",
        help="Directory to save tokenizer JSON model (default: artifacts/tokenizers)",
    )
    p_tok.add_argument(
        "--vocab-size",
        "-v",
        type=int,
        default=32000,
        help="Target subword vocabulary size (default: 32,000)",
    )
    p_tok.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum subword frequency cutoff (default: 2)",
    )
    p_tok.add_argument(
        "--max-seq-len",
        type=int,
        default=128,
        help="Maximum sequence length constraint in tokens (default: 128)",
    )
    p_tok.add_argument(
        "--tokenize-datasets",
        action="store_true",
        default=True,
        help="Export bidirectional tokenized datasets to Parquet (default: True)",
    )
    p_tok.add_argument(
        "--tokenized-output-dir",
        default="data/processed/tokenized",
        help="Directory to store tokenized Parquet splits (default: data/processed/tokenized)",
    )
    p_tok.set_defaults(func=run_train_tokenizer)

    # 4. Train Model
    p_train = subparsers.add_parser("train", help="Train custom Transformer Seq2Seq model")
    p_train.add_argument(
        "--train-file",
        default="data/processed/tokenized/train_ids.parquet",
        help="Path to tokenized training Parquet",
    )
    p_train.add_argument(
        "--val-file",
        default="data/processed/tokenized/val_ids.parquet",
        help="Path to tokenized validation Parquet",
    )
    p_train.add_argument(
        "--checkpoint-dir", default="checkpoints", help="Directory to save model checkpoints"
    )
    p_train.add_argument(
        "--vocab-size", type=int, default=32000, help="Vocabulary size (default: 32000)"
    )
    p_train.add_argument("--d-model", type=int, default=512, help="Hidden dimension (default: 512)")
    p_train.add_argument(
        "--n-heads", type=int, default=8, help="Number of attention heads (default: 8)"
    )
    p_train.add_argument(
        "--n-layers", type=int, default=6, help="Encoder/decoder layers (default: 6)"
    )
    p_train.add_argument(
        "--d-ff", type=int, default=2048, help="Feed-forward dimension (default: 2048)"
    )
    p_train.add_argument("--dropout", type=float, default=0.1, help="Dropout rate (default: 0.1)")
    p_train.add_argument(
        "--max-len", type=int, default=512, help="Max sequence length (default: 512)"
    )
    p_train.add_argument("--pad-id", type=int, default=0, help="Padding token ID (default: 0)")
    p_train.add_argument("--no-tie-weights", action="store_true", help="Disable 3-way weight tying")
    p_train.add_argument(
        "--batch-size", type=int, default=32, help="Batch size per device (default: 32)"
    )
    p_train.add_argument(
        "--bucket-batching", action="store_true", default=True, help="Use dynamic length bucketing"
    )
    p_train.add_argument(
        "--num-buckets", type=int, default=10, help="Number of length buckets (default: 10)"
    )
    p_train.add_argument(
        "--epochs", type=int, default=30, help="Maximum epochs to train (default: 30)"
    )
    p_train.add_argument(
        "--patience", type=int, default=5, help="Early stopping patience (default: 5)"
    )
    p_train.add_argument(
        "--grad-accum-steps", type=int, default=2, help="Gradient accumulation steps (default: 2)"
    )
    p_train.add_argument(
        "--clip-grad-norm", type=float, default=1.0, help="Max gradient norm (default: 1.0)"
    )
    p_train.add_argument(
        "--lr", type=float, default=0.0005, help="Base learning rate (default: 0.0005)"
    )
    p_train.add_argument(
        "--warmup-steps", type=int, default=4000, help="Linear warmup steps (default: 4000)"
    )
    p_train.add_argument(
        "--label-smoothing", type=float, default=0.1, help="Label smoothing epsilon (default: 0.1)"
    )
    p_train.add_argument(
        "--mixed-precision",
        default="fp16",
        choices=["fp16", "bf16", "no"],
        help="AMP mode (default: fp16)",
    )
    p_train.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "mps", "cpu"], help="Compute device"
    )
    p_train.add_argument("--beta1", type=float, default=0.9, help="Adam beta1 (default: 0.9)")
    p_train.add_argument("--beta2", type=float, default=0.98, help="Adam beta2 (default: 0.98)")
    p_train.add_argument("--eps", type=float, default=1e-9, help="Adam epsilon (default: 1e-9)")
    p_train.add_argument(
        "--weight-decay", type=float, default=0.0001, help="Weight decay (default: 0.0001)"
    )
    p_train.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p_train.add_argument(
        "--resume", default=None, help="Path to checkpoint to resume training from"
    )
    p_train.set_defaults(func=run_train)

    # 5. Evaluate
    p_eval = subparsers.add_parser(
        "evaluate", help="Compute BLEU, chrF++, TER metrics on gold benchmarks"
    )
    p_eval.add_argument(
        "--model-path",
        default="checkpoints/best_model.pt",
        help="Path to trained model checkpoint (.pt) (default: checkpoints/best_model.pt)",
    )
    p_eval.add_argument(
        "--tokenizer-path",
        default="artifacts/tokenizers/joint_bpe_32k.json",
        help="Path to trained tokenizer JSON model",
    )
    p_eval.add_argument(
        "--benchmark-file",
        default="data/raw/flores200_benchmark.parquet",
        help="Path to FLORES-200 or test benchmark parquet file",
    )
    p_eval.add_argument(
        "--output-dir",
        default="artifacts/evaluation",
        help="Directory to save evaluation markdown and JSON reports",
    )
    p_eval.add_argument(
        "--method",
        default="beam",
        choices=["beam", "greedy"],
        help="Decoding method (default: beam)",
    )
    p_eval.add_argument("--beam-size", type=int, default=4, help="Beam size (default: 4)")
    p_eval.add_argument(
        "--length-penalty", type=float, default=0.6, help="Length penalty alpha (default: 0.6)"
    )
    p_eval.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.2,
        help="Repetition penalty theta (default: 1.2)",
    )
    p_eval.add_argument(
        "--max-len", type=int, default=128, help="Max decoded length (default: 128)"
    )
    p_eval.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional max samples to evaluate (for fast benchmarking)",
    )
    p_eval.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "mps", "cpu"], help="Compute device"
    )
    p_eval.add_argument("--vocab-size", type=int, default=32000, help="Vocab size (default: 32000)")
    p_eval.add_argument("--d-model", type=int, default=512, help="Hidden dimension (default: 512)")
    p_eval.add_argument("--n-heads", type=int, default=8, help="Number of heads (default: 8)")
    p_eval.add_argument("--n-layers", type=int, default=6, help="Number of layers (default: 6)")
    p_eval.add_argument(
        "--d-ff", type=int, default=2048, help="Feedforward dimension (default: 2048)"
    )
    p_eval.set_defaults(func=run_evaluate)

    # 6. Serve
    p_serve = subparsers.add_parser("serve", help="Launch FastAPI translation inference service")
    p_serve.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host network interface to bind REST service (default: 0.0.0.0)",
    )
    p_serve.add_argument(
        "--port",
        type=int,
        default=8000,
        help="TCP port to bind REST service (default: 8000)",
    )
    p_serve.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of Uvicorn worker processes (default: 1)",
    )
    p_serve.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for local development",
    )
    p_serve.add_argument(
        "--model-path",
        default="checkpoints/best_model.pt",
        help="Path to trained model checkpoint (.pt) (default: checkpoints/best_model.pt)",
    )
    p_serve.add_argument(
        "--tokenizer-path",
        default="artifacts/tokenizers/joint_bpe_32k.json",
        help="Path to trained tokenizer JSON model",
    )
    p_serve.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Compute device (default: auto)",
    )
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
