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

### Running on Colab / Kaggle

```bash
git clone https://github.com/Bekafi01/english-amharic-transformer-nmt.git repo && cd repo
pip install -q -e ".[data,tokenization,evaluation]"
# --root: persistent output dir (Drive / Kaggle working); --raw-dir: fast local disk for downloads
amnmt data build -c configs/tiny.yaml --root /content/drive/MyDrive/amnmt --raw-dir /content/raw
amnmt data build -c configs/full.yaml --root /content/drive/MyDrive/amnmt --raw-dir /content/raw
```

Re-running skips sources whose shard already exists under `--root`, so a session timeout only
costs the source that was in progress.

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
bidirectional model trained on the *medium* subset (4.43M pairs: all curated sources + NLLB pairs
with LASER score ≥ 1.068) for 44k steps ≈ 12 GPU-hours on a Colab T4 (fp16, 8k tokens × 3
accumulation). Scores on **FLORES-200 devtest** (1,012 sentences), beam 4, sacreBLEU 2.6:

| direction | BLEU  | chrF++ | spBLEU |
| --------- | ----- | ------ | ------ |
| en → am   | 14.06 | 36.97  | 27.40  |
| am → en   | 24.02 | 48.90  | 24.62  |

BLEU is `13a`-tokenized; spBLEU uses the `flores200` SentencePiece tokenizer. Hypotheses are
scored against references normalized the same way as the training data (homophone folding etc.);
against the raw references en→am BLEU is 11.80. Still improving slowly at 44k — numbers will be
updated as training continues.

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
