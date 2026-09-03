# English-Amharic Transformer Neural Machine Translation (NMT)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/package%20manager-uv-de5fe9.svg)](https://github.com/astral-sh/uv)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, production-grade Neural Machine Translation (NMT) system specifically engineered for bidirectional **English $\leftrightarrow$ Amharic** translation. The repository encompasses large-scale multi-source parallel corpus acquisition (**17.5+ Million sentence pairs**), an Ethiopic script normalization engine, shared Byte-Pair Encoding (BPE) subword tokenization, a custom **Sequence-to-Sequence Transformer** built from scratch in PyTorch, mixed-precision training, beam search decoding, ONNX Runtime hardware acceleration, and containerized FastAPI / Streamlit serving.

---

## Key Highlights

- **17.5M+ Parallel Corpus Acquisition**: High-speed, streaming data acquisition pipeline mining Meta NLLB LASER-3 bitext, OPUS archives (CCAligned, Tanzil, OPUS-100), curated MT560 slices, and verified human discourse corpora.
- **Constant-Memory Chunked Serialization**: Memory-bounded streaming PyArrow Parquet writer ($500\text{k}$ chunk size) maintaining a flat $\le 250\text{ MB}$ RAM footprint throughout multi-gigabyte exports.
- **Zero-Contamination Benchmark Isolation**: Strict quarantine of the 6,027 gold-standard FLORES-200 professional human translations exclusively for evaluation and ablation.
- **Ethiopic Script Normalization**: Specialized preprocessor addressing Amharic orthographic homophones (ሀ/ሐ/ኀ/ሃ, ዐ/አ, ጸ/ፀ, ሠ/ሰ), Ethiopic punctuation (`፡` `።` `፤` `፥`), and numeral conversions.
- **Shared BPE Vocabulary (32k)**: Unified subword tokenizer across Latin and Ethiopic Unicode scripts to enable joint cross-lingual representation and 3-way embedding weight tying.
- **From-Scratch Transformer Architecture**: Pre-LayerNorm (Pre-LN) multi-head self-attention and cross-attention blocks with Sinusoidal Positional Embeddings and residual connections.
- **Optimized Training & Inference**: Mixed-precision (`torch.cuda.amp`), Noam learning rate schedule with linear warmup, dynamic length-bucketed batch sampling, label-smoothed cross-entropy, beam search decoding, and ONNX Runtime deployment.
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
│   ├── raw/                           # Downloaded raw corpus archives & Parquet tables
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

## Parallel Corpus Engineering & Empirical Benchmarks

The data acquisition engine consolidates parallel corpora across multiple domains, combining semantically mined bitext with high-register human translations:

| Source ID | Origin / Repository | Pairs Collected | % Share | Mining & Alignment Methodology |
|:---|:---|:---:|:---:|:---|
| **`nllb`** | Meta AI NLLB Bitext (`amh_Ethi-eng_Latn`) | **16,137,053** | 91.86% | **LASER-3 Semantic Vector Mining**: High-margin cross-lingual cosine similarity streamed directly from GCS storage. |
| **`mt560`** | `michsethowusu/english-amharic_sentence-pairs_mt560` | **669,145** | 3.81% | Multi-domain OPUS benchmark slice providing broad lexical coverage. |
| **`opus_ccaligned`** | OPUS CCAligned (`am-en`) | **346,511** | 1.97% | Common Crawl web-mined parallel sentences for technical & modern terms. |
| **`drive_custom`** | Curated Human Translation Corpus | **228,000** | 1.30% | Verified news, discourse, and conversational parallel sentences. |
| **`opus_tanzil`** | OPUS Tanzil (`am-en`) | **93,526** | 0.53% | Classical and literary parallel texts with strict sentence-level alignment. |
| **`opus100`** | `Helsinki-NLP/opus-100` (`am-en`) | **93,027** | 0.53% | Curated multi-domain parallel corpus across news, subtitles, and documentation. |
| **Total Training Pool** | **Aggregated Parallel Corpus** | **17,567,262** | **100.0%** | **Compressed Parquet Size: 1.78 GB (Snappy)** |
| **`flores200`** | `rasyosef/flores_english_amharic_mt` | **6,027** | *Isolated* | **FLORES-200 Gold Standard**: Professionally translated benchmark quarantined for evaluation. |

### Data Quality & Script Verification Metrics
- **Parallel Completeness**: **100.00%** ($17,567,262$ / $17,567,262$ non-empty, non-null pairs).
- **Distinct Target Pairs**: **99.99%** unique translation pairs.
- **Script Compliance ($n=200,000$)**:
  - Amharic text contains Ge'ez Fidel (`\u1200`–`\u137F`): **99.95%**
  - English text contains Latin characters (`[a-zA-Z]`): **99.98%**
- **Token Profile**:
  - English sentence length: Mean = 12.1 words, Median = 10.0 words, $p_{95} = 27$ words.
  - Amharic sentence length: Mean = 8.6 words, Median = 7.0 words, $p_{95} = 19$ words.
- **Streaming Throughput**: **310,925 pairs/sec** during chunked serialization into Snappy Parquet.

---

## Architectural & Design Decisions

### 1. Ethiopic Script Normalization
Amharic uses the Ge'ez syllabary (Fidel) and features several character sets that share identical phonetic pronunciations in modern spoken Amharic (homophones):
- `ሀ`, `ሐ`, `ኀ`, `ሃ` $\rightarrow$ unified to `ሀ` (/h/)
- `አ`, `ዐ` $\rightarrow$ unified to `አ` (/ʔ/)
- `ጸ`, `ፀ` $\rightarrow$ unified to `ጸ` (/tsʼ/)
- `ሰ`, `ሠ` $\rightarrow$ unified to `ሰ` (/s/)

Normalizing these characters collapses vocabulary fragmentation and reduces out-of-vocabulary (OOV) rates without altering semantic fidelity. Ethiopic word separators (`፡`) are standardized to spaces, and terminal punctuation (`።` `፤` `፥` `፦`) is preserved.

### 2. Shared 32k BPE Vocabulary & Weight Tying
A unified 32,000 subword vocabulary is trained jointly across English and Amharic text:
- Because the Latin and Ethiopic Unicode blocks are completely disjoint, subword merges naturally partition across language boundaries.
- A shared vocabulary enables **three-way weight tying**:
  $$E_{\text{src}} = E_{\text{tgt}} = W_{\text{projection}}^T$$
  This drastically decreases parameter count, regularizes shared numerals and punctuation, and improves gradient flow.

### 3. Pre-LN Transformer Architecture
Following modern Transformer best practices, Layer Normalization is placed on the input paths of each sub-layer (**Pre-LN**) rather than on the residual addition path (**Post-LN**). This stabilizes gradients at initialization and enables training without delicate learning-rate warmup warm-starts.

---

## Quick Start

### 1. Environment Setup

This project uses [uv](https://github.com/astral-sh/uv) for fast, deterministic Python environment and dependency management.

```bash
# Clone the repository
git clone https://github.com/Bekafi01/english-amharic-transformer-nmt.git
cd english-amharic-transformer-nmt

# Synchronize dependencies with uv
uv sync --extra dev
```

Alternatively, standard `pip` can be used:
```bash
pip install -e ".[dev]"
```

### 2. Parallel Corpus Acquisition via CLI

To run the multi-source parallel data acquisition pipeline:

```bash
# Ingest all configured sources and serialize to Parquet
uv run python scripts/main.py collect --config configs/data_config.yaml --output-dir data/raw

# Run rapid test collection capped at 10,000 pairs per source
uv run python scripts/main.py collect --max-samples 10000 --output-dir data/raw_sample
```

CLI options:
- `--config`, `-c`: Path to data configuration YAML (default: `configs/data_config.yaml`).
- `--output-dir`, `-o`: Output folder for raw Parquet and manifest (default: `data/raw`).
- `--max-samples`, `-n`: Cap on samples per source for quick iteration.
- `--chunk-size`: Number of rows per PyArrow chunk for constant RAM usage (default: `500,000`).
- `--skip`: Comma-separated source IDs to skip (e.g. `--skip ccaligned,opus_tanzil`).

### 3. Interactive Web Application & Serving

```bash
# Launch FastAPI inference server
uv run uvicorn src.nmt_engine.serving.api:app --host 0.0.0.0 --port 8000 --reload

# Launch interactive Streamlit UI
uv run streamlit run app.py
```

### 4. Containerized Deployment

```bash
docker compose up --build
```

---

## Testing & Quality Assurance

The test suite validates data ingestion, Unicode normalization, script-based extraction, and model dimensions:

```bash
# Run complete test suite with coverage
uv run pytest tests/ -v

# Run collector-specific unit tests
uv run pytest tests/test_collector.py -v

# Run linting and code style checks
uv run ruff check src/ tests/ configs/
uv run ruff format --check src/ tests/
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
