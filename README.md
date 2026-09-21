# English ↔ Amharic Transformer NMT

A from-scratch PyTorch Transformer for bidirectional English–Amharic translation, built as a
layered Python package with one CLI. Heavy stages (corpus build, training) run on Colab/Kaggle;
everything is reproducible from the YAML configs in `configs/`.

## Setup

```bash
uv sync --all-extras --group dev
uv run amnmt --help
make check      # ruff + import-linter + mypy + pytest
```

## Pipeline

| Stage        | Command                                                                               | Output                                                                           |
| ------------ | ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| Corpus build | `amnmt data build -c configs/full.yaml [--root DIR] [--raw-dir DIR]`                  | `data/processed/full/{train,train_holdout,valid,test}.parquet`, `data_card.json` |
| Tokenizer    | `amnmt tokenizer train -c configs/full.yaml [--root DIR]`                             | `artifacts/full/tokenizer/{tokenizer.json,stats.json}`                           |
| Training     | `amnmt train -c configs/full.yaml [--root DIR] --run NAME --resume --time-limit MIN`  | `artifacts/full/runs/NAME/{last.pt,best.pt,metrics.jsonl}`                       |
| Translate    | `amnmt translate -m best.pt -d en-am "text" [--beam 4]`                               | stdout                                                                           |
| Evaluation   | `amnmt eval -m best.pt -c configs/full.yaml [--root DIR] --split test [--spbleu]`     | `runs/NAME/eval_test_beam4.{json,md}` (BLEU, chrF++, spBLEU, both directions)    |
| Serving      | `amnmt serve -m best.pt [--port 8000]` · `streamlit run app.py` · `docker compose up` | REST `POST /translate`, `GET /health`, OpenAPI at `/docs`; Streamlit UI on :8501 |

`configs/tiny.yaml` runs the same pipeline on a few thousand pairs and must always work.

### Reproducing the full model on Colab / Kaggle

Every stage is idempotent and resumable, so the sequence below can be spread over as many
sessions as needed. `ROOT` is a persistent folder (Drive); `RAW` is fast local disk for downloads.

```bash
git clone https://github.com/Bekafi01/english-amharic-transformer-nmt.git repo && cd repo
pip install -q -e ".[data,tokenization,evaluation]"
ROOT=/content/drive/MyDrive/amnmt; RAW=/content/raw

# 1. Corpus: ~1 GB download, 16M pairs streamed → 15M clean pairs           (CPU, ~50 min)
amnmt data build -c configs/full.yaml --root $ROOT --raw-dir $RAW
# 2. Tokenizer: 32k joint BPE from a 2M-sentence/language sample              (CPU, ~4 min)
amnmt tokenizer train -c configs/full.yaml --root $ROOT
# 3. Training: first call tokenizes the subset into a cache (~8 min), then trains.
#    Re-run the same line in each new session; it resumes from last.pt.      (T4, ~15 h total)
amnmt train -c configs/full.yaml --root $ROOT --run base_medium --resume --time-limit 690
# 4. Evaluation on FLORES-200 devtest, both directions                        (T4, ~5 min)
amnmt eval -m $ROOT/artifacts/full/runs/base_medium/best.pt -c configs/full.yaml --root $ROOT --spbleu
# 5. Use it
amnmt translate -m $ROOT/artifacts/full/runs/base_medium/best.pt -d en-am "Coffee originated in Ethiopia."
```

Outputs land under `$ROOT/{data/processed/full, artifacts/full/{tokenizer,cache,runs/base_medium}}`.
The data build skips sources whose shard already exists; training checkpoints every 2,000 steps
and restores model, optimizer, LR schedule, sampler position and RNG state exactly.

## Data

All training sources are public. `valid`/`test` are FLORES-200 dev/devtest (official tarball) and
are excluded from training by a normalized-key match on both sides.

| Source                 | Pairs (raw) | License        | Origin                                                                                                                                       |
| ---------------------- | ----------- | -------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| MT560 (en–am slice)    | 669k        | CC-BY-4.0      | [HF: michsethowusu/english-amharic_sentence-pairs_mt560](https://huggingface.co/datasets/michsethowusu/english-amharic_sentence-pairs_mt560) |
| OPUS-100 `am-en`       | 89k         | mixed (OPUS)   | [HF: Helsinki-NLP/opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100)                                                           |
| Tanzil                 | ~100k       | Tanzil         | [OPUS](https://opus.nlpl.eu/Tanzil.php)                                                                                                      |
| CCAligned              | ~350k       | CC (web crawl) | [OPUS](https://opus.nlpl.eu/CCAligned.php)                                                                                                   |
| NLLB (LASER-mined)     | ~16M        | CC-BY-NC-4.0   | [OPUS](https://opus.nlpl.eu/NLLB.php)                                                                                                        |
| FLORES-200 (eval only) | 997 + 1012  | CC-BY-SA-4.0   | [Meta](https://github.com/facebookresearch/flores/tree/main/flores200)                                                                       |

Preprocessing (`amnmt.data`): NFC, Ethiopic homophone folding, Ge'ez numerals → digits, wordspace
`፡` → space, Moses detokenization, then length/ratio/script-share/URL/markup filters and exact
dedup on a normalized key. Per-source keep/reject counts are written to `data_card.json`.

Tokenizer (`amnmt.tokenization`): one joint 32k BPE for both languages (HF `tokenizers`), Metaspace
word split, isolated punctuation, individual digits, byte fallback (no `<unk>` ever). Specials:
`<pad>=0 <s>=1 </s>=2 <unk>=3 <2am>=4 <2en>=5`. Trained on a seeded reservoir sample of the
score-filtered training subset.

## Results

`configs/full.yaml` — Transformer-base (60.5M params, 6+6 layers, d=512, tied 32k BPE), one
bidirectional model trained on the _medium_ subset (4.43M pairs: all curated sources + NLLB pairs
with LASER score ≥ 1.068) for 54k steps (7 epochs, ≈ 15 GPU-hours on a Colab T4, fp16, 8k tokens × 3
accumulation) until FLORES dev loss plateaued. Scores on **FLORES-200 devtest** (1,012 sentences),
beam 4, sacreBLEU 2.6:

| direction | BLEU  | chrF++ | spBLEU |
| --------- | ----- | ------ | ------ |
| en → am   | 14.51 | 37.74  | 28.36  |
| am → en   | 24.39 | 49.27  | 24.95  |

BLEU is `13a`-tokenized; spBLEU uses the `flores200` SentencePiece tokenizer. Hypotheses are
scored against references normalized the same way as the training data (homophone folding etc.);
against the raw references en→am BLEU is 12.24 — the cost of folding, which a production model
should avoid (`normalize.fold_homophones: false`). Greedy decoding is about 1 BLEU lower in each
direction.

## Layout

```
src/amnmt/
├── core/          config (pydantic), logging
├── data/          sources, normalize, filters, flores, pipeline
├── tokenization/  joint BPE
├── model/         Transformer (pure torch.nn)
├── training/      dataset, loss, scheduler, trainer
├── inference/     greedy / beam search, Translator
├── evaluation/    FLORES-200 BLEU / chrF++
├── serving/       FastAPI
└── cli.py         `amnmt` entrypoint
```

Layers may only import downward; `import-linter` enforces this in CI.
