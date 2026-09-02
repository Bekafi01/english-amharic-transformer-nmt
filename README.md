# English-Amharic Transformer Neural Machine Translation (NMT)

[![CI Pipeline](https://github.com/beka/english-amharic-transformer-nmt/actions/workflows/ci.yml/badge.svg)](https://github.com/beka/english-amharic-transformer-nmt/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, production-ready Neural Machine Translation (NMT) system specifically engineered for **English $\leftrightarrow$ Amharic** translation, featuring a custom **Sequence-to-Sequence Transformer** implemented from scratch in PyTorch, an Ethiopic script normalization engine, Byte-Pair Encoding (BPE) tokenization, mixed-precision training, beam search decoding, ONNX Runtime acceleration, and full FastAPI / Streamlit serving.

---

## Key Highlights

- **From-Scratch Transformer Architecture**: Pre-LayerNorm (Pre-LN) multi-head self-attention and cross-attention blocks built modularly with Sinusoidal Positional Encoding and 3-way weight tying.
- **Ethiopic Script Normalization**: Specialized preprocessor handling Amharic homophones (ሀ/ሐ/ኀ/ሃ, ዐ/አ, ጸ/ፀ, ሠ/ሰ), Ethiopic punctuation (`፡` `።` `፤` `፥`), and numeral standardizations.
- **Shared BPE Vocabulary (32k)**: A unified subword tokenizer covering Latin and Ethiopic Unicode scripts to enable joint representation and weight tying.
- **Optimized Training Loop**: Mixed-precision training (`torch.cuda.amp`), Noam learning rate schedule with warmup, dynamic length-bucketed batch sampling, and label-smoothed cross-entropy loss.
- **High-Performance Inference**: Configurable Beam Search decoding with length penalty and repetition penalty, plus automated export to ONNX Runtime.
- **Production Serving**: Containerized FastAPI REST API (`/translate`, `/health`) and interactive Streamlit web dashboard.

---

## Project Structure

```
english-amharic-transformer-nmt/
├── .github/
│   └── workflows/
│       └── ci.yml                     # GitHub Actions CI for linting & automated tests
├── configs/
│   ├── base_config.yaml               # Global seed, paths, logging, hardware config
│   ├── data_config.yaml               # Data sources, URLs, licensing, Ethiopic normalizers
│   └── model_config.yaml              # Transformer dimensions, optimizer, scheduler params
├── data/
│   ├── raw/                           # Downloaded raw corpus archives
│   └── processed/                     # Cleaned, tokenized train/val/test splits
├── artifacts/
│   ├── tokenizers/                    # Trained vocabularies and tokenizer configs
│   ├── checkpoints/                   # Model weights, best checkpoints, optimizer state
│   ├── onnx/                          # Exported ONNX models and quantized weights
│   └── benchmarks/                    # Quality & ablation reports (BLEU, chrF++, TER)
├── notebooks/
│   ├── 01_corpus_collection_and_cleaning.ipynb   # Exploratory corpus analysis & Ethiopic EDA
│   ├── 02_tokenizer_training.ipynb               # BPE/WordPiece tokenization experiments
│   ├── 03_transformer_from_scratch.ipynb         # Transformer modules & attention map viz
│   └── 04_evaluation_and_benchmarks.ipynb        # Test set evaluation, error analysis & ONNX
├── src/
│   └── nmt_engine/
│       ├── config.py                  # Strongly-typed Pydantic YAML config loader
│       ├── data/
│       │   ├── collector.py           # Multi-source dataset downloader & dispatcher
│       │   ├── preprocessor.py        # Amharic & English Unicode cleaner and length filter
│       │   ├── tokenizer.py          # BPE tokenizer wrapper (encode/decode/save/load)
│       │   ├── dataset.py            # PyTorch TranslationDataset with dynamic padding
│       │   └── samplers.py           # Length-bucketed dynamic batch sampler
│       ├── models/
│       │   ├── transformer.py        # Full Encoder-Decoder Transformer model
│       │   ├── encoder.py            # Pre-LN TransformerEncoder & EncoderLayer
│       │   ├── decoder.py            # Pre-LN TransformerDecoder & DecoderLayer
│       │   ├── attention.py          # MultiHeadAttention with scaled dot-product
│       │   ├── embeddings.py         # Token & Positional embeddings (Sinusoidal/Learned)
│       │   └── layers.py             # FeedForward, LayerNorm/RMSNorm, Residuals
│       ├── training/
│       │   ├── trainer.py            # Epoch loop, AMP, gradient accumulation & clipping
│       │   ├── loss.py               # Label smoothed cross-entropy loss
│       │   ├── lr_schedule.py        # Noam / Inverse-Square-Root warmup scheduler
│       │   └── callbacks.py          # Checkpointing, EarlyStopping, Metric tracking
│       ├── inference/
│       │   ├── translator.py         # Greedy & Beam Search decoding with length penalty
│       │   └── exporter.py           # PyTorch to ONNX export & dynamic batching
│       ├── evaluation/
│       │   ├── metrics.py            # BLEU (sacrebleu), chrF++, TER computation
│       │   └── benchmarks.py         # Runtime hardware profiler (latency, throughput, memory)
│       ├── serving/
│       │   ├── api.py                # FastAPI endpoints (/translate, /health, /metrics)
│       │   └── schemas.py            # Pydantic request/response schemas
│       └── utils/
│           ├── logging.py            # Structured rich logging setup
│           ├── seed.py               # Deterministic seed initialization
│           └── io.py                 # File read/write utilities
├── scripts/
│   └── main.py                        # Unified CLI entry point for all stages
├── tests/
│   ├── conftest.py                   # Pytest fixtures and mock datasets
│   ├── test_collector.py             # Mock-based unit tests per source dispatch
│   ├── test_preprocessor.py          # Ethiopic Unicode normalization & filtering tests
│   ├── test_tokenizer.py             # Tokenizer roundtrip and special token tests
│   ├── test_transformer.py           # Shape, mask, and gradient flow tests for Transformer
│   ├── test_inference.py             # Beam search vs greedy search tests
│   └── test_api.py                   # FastAPI test client integration tests
├── app.py                             # Interactive Web UI (Streamlit / Gradio)
├── Dockerfile                         # Production container image
├── docker-compose.yml                 # Service orchestrator (API + UI)
├── Makefile                           # Development task runner (lint, test, run, docker)
├── .env.example                       # Environment variables template
├── pyproject.toml                     # Modern Python build configuration & dependencies
└── README.md                          # Comprehensive project documentation
```

---

## Architectural & Design Decisions

### 1. Ethiopic Script Normalization & Low-Resource Sparing
Amharic is written in the Ethiopic syllabary (Fidel) and features several character pairs that produce identical phonetic values in modern spoken Amharic (e.g. `ሀ`, `ሐ`, `ኀ`, `ሃ` all map to /h/; `ሰ`, `ሠ` to /s/; `አ`, `ዐ` to /ʔ/; `ጸ`, `ፀ` to /tsʼ/).
- **Normalization ON by default**: Mapping these homophones reduces vocabulary sparsity significantly during training on low-resource parallel corpora.
- **Ablation Tracking**: The data pipeline preserves the raw corpus in `data/raw/` so datasets can be reproduced with normalization disabled. Quality comparisons (BLEU and chrF++) are recorded in `artifacts/benchmarks/`.

### 2. Shared BPE Vocabulary (32k) & Weight Tying
A unified 32,000 Byte-Pair Encoding subword vocabulary is used for both English and Amharic.
- Because Latin and Ethiopic scripts are disjoint in Unicode, the vocabulary naturally partitions merges between the two scripts.
- Shared embeddings allow **3-way weight tying** ($E_{\text{src}} = E_{\text{tgt}} = W_{\text{out}}^T$), reducing parameter count and regularizing numbers, punctuation, and shared international tokens.

### 3. Benchmarks Disambiguation
- **Runtime Performance Profiler (`src/nmt_engine/evaluation/benchmarks.py`)**: Profiles hardware latency (p50/p95/p99 ms), generation throughput (tokens/sec), and memory footprint comparing PyTorch vs ONNX Runtime.
- **Quality Evaluation Reports (`artifacts/benchmarks/`)**: Stores translation metric logs (SacreBLEU, chrF++, TER) across test sets and ablation configurations.

---

## Quick Start

### 1. Installation
```bash
# Clone repository
git clone https://github.com/beka/english-amharic-transformer-nmt.git
cd english-amharic-transformer-nmt

# Install in editable mode with development dependencies
pip install -e ".[dev]"
```

### 2. Pipeline Execution via CLI
```bash
# 1. Download parallel corpus
python scripts/main.py collect

# 2. Preprocess & normalize Ethiopic / English text
python scripts/main.py preprocess

# 3. Train shared BPE tokenizer
python scripts/main.py train-tokenizer

# 4. Train Transformer model from scratch
python scripts/main.py train

# 5. Evaluate on test set (SacreBLEU / chrF++)
python scripts/main.py evaluate

# 6. Export to optimized ONNX format
python scripts/main.py export
```

### 3. Interactive Web App & API Serving
```bash
# Start FastAPI backend
uvicorn src.nmt_engine.serving.api:app --host 0.0.0.0 --port 8000 --reload

# Start Streamlit translation UI
streamlit run app.py
```

### 4. Running with Docker Compose
```bash
docker compose up --build
```

---

## Testing & Quality Assurance
```bash
# Run complete test suite
pytest tests/ -v

# Run linting and code formatting checks
ruff check src/ tests/ configs/
black --check src/ tests/
```

---

## Data Sources & Licensing

| Dataset | Source | License | Description |
| :--- | :--- | :--- | :--- |
| **OPUS-100** | [Helsinki-NLP/opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100) | CC-BY-4.0 / Various | Multi-domain collection from OPUS |
| **OPUS Books** | [opus_books](https://huggingface.co/datasets/opus_books) | Public Domain / CC0 | Parallel literary texts |
| **CCAligned** | [Statmt](https://data.statmt.org/cc-aligned/) | CC-BY-SA | Web-crawled aligned sentences |
| **Tanzil** | [OPUS Tanzil](https://object.pouta.csc.fi/OPUS-Tanzil/) | Tanzil License | Parallel religious/classical texts |

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
