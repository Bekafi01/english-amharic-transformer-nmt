# Rebuild Plan — English ↔ Amharic Transformer NMT (v2)

Status legend: `[ ]` not started · `[~]` in progress · `[x]` done (milestone verified)

The v1 code is preserved at git tag `v1-legacy`. v2 is a clean rewrite; nothing is copied
from v1 without being re-read, simplified, and re-tested.

## Why v1 failed (and the rule that fixes each)

| Problem in v1                                                                          | Rule in v2                                                                                                                                                                                   |
| -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Messy code: notebooks and `src/` both held logic, `scratch/` scripts patched notebooks | **`src/` is the only source of truth.** Notebooks are thin runners that `pip install` the package and call its CLI. No logic in notebooks.                                                   |
| Wrong architecture: modules imported each other freely                                 | **Layered package with enforced boundaries** (`import-linter` runs in CI). Lower layers never import higher ones.                                                                            |
| Poor results: trained at full scale before validating anything                         | **Scale ladder.** Every stage is first proven on a tiny slice (minutes on CPU), then a small slice (≤ 1 h on Colab), then full scale. No phase advances without its milestone check passing. |
| Unclear pipeline: 6 CLI stages with hidden coupling                                    | **One CLI, explicit artifacts.** Each stage reads named inputs and writes named outputs under `artifacts/<stage>/`. A stage never reaches into another stage's internals.                    |

## Non-negotiables

1. Every phase ends with a **milestone check** — an automated test or a reproducible script whose output is recorded in `docs/milestones/`.
2. `uv run pytest`, `uv run ruff check`, `uv run lint-imports` pass on `main` at all times.
3. FLORES-200 dev/devtest are **never** used for training, tokenizer training, or dedup reference. A test asserts zero overlap.
4. Every experiment config is a YAML file committed to `configs/`. No untracked hyperparameters.
5. Small before big: a config named `*_tiny.yaml` must exist and run end-to-end before any `*_full.yaml`.

## Package layout and layer rules

Package name: `amnmt`. Layers, lowest first. A module may import only from its own layer or lower.

```
src/amnmt/
├── core/          L0  config (pydantic), paths, logging, seeding, io. No ML deps.
├── data/          L1  acquisition, cleaning, Ethiopic normalization, dedup, splits → parquet
├── tokenization/  L2  joint BPE training + encode/decode wrapper
├── model/         L3  Transformer (pure torch.nn, no I/O, no config file reading)
├── training/      L4  dataset/batching, loss, scheduler, trainer, checkpointing
├── inference/     L5  greedy/beam decoding, Translator facade (loads checkpoint + tokenizer)
├── evaluation/    L6  FLORES-200 loading, BLEU/chrF++, report writer
├── serving/       L7  FastAPI app, schemas; Streamlit UI lives in top-level app.py
└── cli.py         Typer app; the only module allowed to import every layer
```

Contracts are declared in `pyproject.toml` under `[tool.importlinter]`.

## Phases

### Phase 0 — Foundation `[x]`

Deliverables

- `pyproject.toml` (uv, dependency groups per layer), `uv.lock`
- `src/amnmt/core/`: `config.py` (typed YAML config incl. paths), `logging.py`
- Empty layer packages with a docstring stating responsibility and allowed imports
- `cli.py` with `version` and `config validate`
- `configs/tiny.yaml` baseline config
- `tests/` with config round-trip tests
- `.github/workflows/ci.yml`: ruff, import-linter, pytest
- `Makefile`, `.gitignore`, `README.md` (short; points to this plan)

Milestone check

- `uv sync && uv run pytest && uv run ruff check . && uv run lint-imports` all green locally and in CI.

### Phase 1 — Data `[~]`

Deliverables

- `data/sources.py`: three loaders — `opus_moses` (OPUS zip), `hf` (HF datasets, streaming), `local_pair` (two aligned text files) — each yielding `(en, am)`
- `data/normalize.py`: NFC, Ethiopic wordspace/punctuation, homophone folding (flag), Ge'ez numerals → digits, light detokenization — pure functions
- `data/filters.py`: empty/length/ratio/script-share/copy/URL filters — pure functions returning a reject reason or `None`
- `data/flores.py`: FLORES-200 dev/devtest from the official tarball (eval-only)
- `data/pipeline.py`: per source `stream → normalize → filter → parquet shard` (chunked, bounded memory); then DuckDB exact-dedup on a normalized key, FLORES exclusion, holdout split, `data_card.json`
- CLI: `amnmt data build --config configs/<name>.yaml [--root DIR] [--raw-dir DIR]`
- Outputs under `paths.data_processed`: `shards/<source>.parquet`, `train.parquet`, `train_holdout.parquet`, `valid.parquet` (FLORES dev), `test.parquet` (FLORES devtest), `data_card.json`

Milestone check

- `tests/test_normalize.py`, `tests/test_filters.py` cover every rule with Amharic examples.
- `tests/test_pipeline.py` runs the full build on a synthetic `local_pair` source + fake FLORES in `tmp_path` and asserts 0 FLORES sentences in train.
- `tiny` config (5k pairs/source) builds on Colab; `data_card.json` pasted into `docs/milestones/phase1_data_tiny.md`.
- `full` config builds on Colab; card committed to `docs/milestones/phase1_data_full.md`.

Scale ladder: tiny (5k/source) → full. (No `small` for data: filtering is linear; `small` is a training-time subsample.)

### Phase 2 — Tokenization `[ ]`

Deliverables

- `tokenization/train.py`: joint BPE (HF `tokenizers`), byte-fallback, specials `<pad> <s> </s> <unk> <2am> <2en>`
- `tokenization/tokenizer.py`: thin wrapper with `encode(text, lang_tag) -> list[int]`, `decode`, `pad_id`, etc.
- CLI: `amnmt tokenizer train`
- Stats script: vocab coverage, avg tokens/sentence per language, `<unk>` rate

Milestone check

- Round-trip test: `decode(encode(x)) == x` for a fixture of Amharic + English strings incl. punctuation `።፡፤` and Ge'ez numerals.
- `<unk>` rate on val < 0.01 %; tokens/sentence ratio am:en within 0.7–1.5.

### Phase 3 — Model `[ ]`

Deliverables

- `model/layers.py`: MHA, FFN, sinusoidal PE (from scratch, `torch.nn` primitives only)
- `model/transformer.py`: Pre-LN encoder–decoder, tied embeddings/output projection, `forward` and `encode`/`decode_step` API
- `model/masks.py`: padding + causal masks

Milestone check

- Shape/mask unit tests.
- **Parity test**: outputs match `torch.nn.Transformer` (with matching weights) to 1e-5 in eval mode.
- **Overfit test**: 32 synthetic pairs → loss < 0.1 within 300 steps on CPU.

### Phase 4 — Training `[ ]`

Deliverables

- `training/dataset.py`: parquet-backed dataset, token-budget bucketing sampler, collate
- `training/loss.py`: label-smoothed CE (ignores pad)
- `training/scheduler.py`: inverse-sqrt warmup
- `training/trainer.py`: AMP, grad accumulation, clipping, checkpoint/resume, val loss, periodic greedy-decode sample logging
- CLI: `amnmt train --config`, `amnmt train --resume`
- `notebooks/train_colab.ipynb`: ≤ 10 cells — mount drive, `pip install git+…`, run CLI, copy checkpoints to Drive

Milestone check

- `tiny` config trains end-to-end on CPU in < 5 min; val loss decreases monotonically over 3 evals.
- Kill/resume test: resumed run continues from identical step and loss.
- `small` config on Colab T4 reaches val loss < 4.0 within 1 h (baseline recorded in `docs/milestones/`).

### Phase 5 — Inference `[ ]`

Deliverables

- `inference/greedy.py`, `inference/beam.py` (batched, length penalty, early stop on EOS)
- `inference/translator.py`: `Translator.from_checkpoint(path).translate(texts, direction)`
- CLI: `amnmt translate "text" --direction en-am`

Milestone check

- Beam=1 equals greedy exactly.
- Beam=4 BLEU ≥ greedy BLEU on `small` val set (numbers recorded).

### Phase 6 — Evaluation `[ ]`

Deliverables

- `evaluation/metrics.py`: sacreBLEU, chrF++ wrappers
- `evaluation/flores.py`: run both directions on FLORES-200 devtest, write JSON + Markdown
- CLI: `amnmt eval --checkpoint`

Milestone check

- Report committed under `docs/milestones/eval_<tag>.md` for the `small` model.
- Metric wrappers tested against known sacreBLEU reference values.

### Phase 7 — Full-scale training and iteration `[ ]`

- `configs/full.yaml`; run on Colab/Kaggle with Drive checkpointing; log to `docs/milestones/`
- Iterate only one variable at a time (data filters, model size, LR); each run gets a config file and a row in `docs/experiments.md`

Milestone check

- FLORES-200 devtest BLEU/chrF++ recorded for both directions; beats the `small` baseline.

### Phase 8 — Serving `[ ]`

Deliverables

- `serving/app.py`: FastAPI `/translate`, `/health`; `serving/schemas.py`
- `app.py`: Streamlit UI calling `Translator`
- `Dockerfile` (multi-stage, CPU), `docker-compose.yml`

Milestone check

- `tests/test_serving.py` via `TestClient` with a tiny checkpoint fixture.
- `docker compose up` → `curl /translate` returns a translation.

## Decisions (change here, not ad hoc)

| #   | Decision                                                                                                                                              | Rationale                                                                                                |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| D1  | Single bidirectional model with target-language tag (`<2am>`, `<2en>`)                                                                                | Halves training/serving cost; joint vocab already shared. Revisit if one direction lags badly.           |
| D2  | Joint 32k BPE with byte fallback                                                                                                                      | Proven in v1; no `<unk>` for rare Ethiopic glyphs.                                                       |
| D3  | Pre-LN Transformer, base size (d=512, 6+6, h=8, ff=2048) for `full`; d=256, 3+3 for `small`; d=64, 2+2 for `tiny`                                     | Pre-LN is stable without careful warmup on Colab-length runs.                                            |
| D4  | Token-budget batching (e.g. 8k tokens/batch on T4 with AMP) rather than fixed sentence count                                                          | Stable memory, less padding.                                                                             |
| D5  | Training runs on Colab/Kaggle; local machine is CPU-only and used for `tiny` checks and tests                                                         | Matches available hardware.                                                                              |
| D6  | uv for env/lock; Python 3.12                                                                                                                          | Already in place.                                                                                        |
| D7  | Exact dedup on a normalized key (lowercase, punctuation/whitespace stripped) via DuckDB; **no MinHash**                                               | MinHash over 17M pairs is hours of Python on Colab; the normalized key catches most near-dups.           |
| D8  | No language-ID model; script-share filters only                                                                                                       | For en–am the script test does the job; avoids a 130 MB fastText download. Revisit if Tigrinya leaks in. |
| D9  | `valid` = FLORES-200 dev, `test` = FLORES-200 devtest; a 2k random `train_holdout` for loss monitoring                                                | Gold references; no leakage risk from carving noisy train data.                                          |
| D10 | Homophone folding ON for both sides (config flag). Evaluation reports scores against folded **and** raw references                                    | The model can only emit folded forms; scoring only against raw refs would understate quality.            |
| D11 | Heavy compute (data build, training) runs on Colab/Kaggle via `git clone` + `pip install -e .[data]` + `amnmt …`; local machine writes code and tests | User's workflow. Raw downloads go to local Colab disk (`--raw-dir`), outputs to Drive (`--root`).        |

## Definition of done for the whole project

All phases `[x]`, CI green, README documents how to reproduce every milestone number from a fresh clone.
