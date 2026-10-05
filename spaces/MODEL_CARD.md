---
language:
  - en
  - am
license: cc-by-nc-4.0
library_name: pytorch
pipeline_tag: translation
tags:
  - translation
  - amharic
  - english
  - transformer
  - from-scratch
datasets:
  - michsethowusu/english-amharic_sentence-pairs_mt560
  - Helsinki-NLP/opus-100
metrics:
  - bleu
  - chrf
model-index:
  - name: amnmt-en-am-base
    results:
      - task: { type: translation, name: English → Amharic }
        dataset:
          {
            type: flores200,
            name: FLORES-200 devtest,
            config: eng_Latn-amh_Ethi,
          }
        metrics:
          - { type: bleu, value: 14.51, name: BLEU (13a) }
          - { type: chrf, value: 37.74, name: chrF++ }
          - { type: bleu, value: 28.36, name: spBLEU (flores200) }
      - task: { type: translation, name: Amharic → English }
        dataset:
          {
            type: flores200,
            name: FLORES-200 devtest,
            config: amh_Ethi-eng_Latn,
          }
        metrics:
          - { type: bleu, value: 24.39, name: BLEU (13a) }
          - { type: chrf, value: 49.27, name: chrF++ }
          - { type: bleu, value: 24.95, name: spBLEU (flores200) }
---

# amnmt-en-am-base

A from-scratch PyTorch Transformer for **bidirectional English ↔ Amharic** translation. One
60.5M-parameter model serves both directions via a target-language tag (`<2am>` / `<2en>`).
Code: [english-amharic-transformer-nmt](https://github.com/Bekafi01/english-amharic-transformer-nmt) ·
Demo: [Space](https://huggingface.co/spaces/Bekafi/amnmt).

## Usage

```bash
pip install "amnmt[tokenization,hub] @ git+https://github.com/Bekafi01/english-amharic-transformer-nmt.git"
amnmt translate -m hf://Bekafi/amnmt-en-am-base -d en-am "Coffee originated in Ethiopia."
```

```python
from amnmt.inference.translator import Translator

tr = Translator.from_checkpoint("hf://Bekafi/amnmt-en-am-base")
tr.translate(["The children are playing in the garden."], "en-am")  # ['ልጆች በአትክልቱ ውስጥ እየተጫወቱ ነው።']
tr.translate(["ቡና መነሻው ኢትዮጵያ ነው።"], "am-en", beam_size=4)
```

Inputs are normalized the same way as the training data (NFC, Ethiopic homophone folding, Ge'ez
numerals → digits), so raw Amharic spelling variants are accepted.

## Files

| file             |                                                                                                              |
| ---------------- | ------------------------------------------------------------------------------------------------------------ |
| `model.pt`       | fp16 weights + model config, ~120 MB. `torch.load` → `{"model", "config", "vocab_size", "pad_id", "export"}` |
| `tokenizer.json` | joint 32k BPE (HF `tokenizers`), byte fallback; specials `<pad>=0 <s>=1 </s>=2 <unk>=3 <2am>=4 <2en>=5`      |

## Model

|              |                                                                                                        |
| ------------ | ------------------------------------------------------------------------------------------------------ |
| Architecture | Pre-LN encoder–decoder, 6+6 layers, d_model 512, 8 heads, d_ff 2048, dropout 0.1, sinusoidal positions |
| Embeddings   | one 32k × 512 matrix tied across source, target and output projection                                  |
| Parameters   | 60,524,544                                                                                             |
| Max length   | 256 positions (trained on ≤ 128 per side)                                                              |
| Decoding     | beam search, GNMT length penalty 1.0                                                                   |

## Training data

4,429,739 sentence pairs after cleaning, both directions (8.86M training samples):

| source                                        | license        | pairs after dedup |
| --------------------------------------------- | -------------- | ----------------- |
| NLLB mined bitext (OPUS), LASER score ≥ 1.068 | CC-BY-NC-4.0   | 3.5M              |
| MT560 en–am slice                             | CC-BY-4.0      | 628k              |
| CCAligned (OPUS)                              | CC (web crawl) | 259k              |
| Tanzil (OPUS)                                 | Tanzil         | 35k               |
| OPUS-100 am–en                                | mixed          | 3k                |

Preprocessing: NFC, Ethiopic homophone folding (ሐ/ኀ→ሀ, ሠ→ሰ, ዐ→አ, ፀ→ጸ), Ge'ez numerals → digits,
`፡` → space, Moses detokenization; length/ratio/script/URL/markup filters; exact dedup on a
normalized key. **FLORES-200 dev and devtest were removed from training by normalized-key match.**

## Training

54,000 steps (7.1 epochs) on a single Colab T4, fp16 + GradScaler, 8,000 padded tokens × 3
accumulation per step, AdamW (β 0.9/0.98), peak LR 7e-4 with 4k warmup and inverse-sqrt decay,
label smoothing 0.1, grad clip 1.0. ≈ 15 GPU-hours over four resumed sessions. Stopped when
FLORES dev loss plateaued (perplexity 7.9).

## Evaluation — FLORES-200 devtest (1,012 sentences), sacreBLEU 2.6

| direction | beam | BLEU      | chrF++    | spBLEU    | BLEU vs raw refs |
| --------- | ---- | --------- | --------- | --------- | ---------------- |
| en → am   | 4    | **14.51** | **37.74** | **28.36** | 12.24            |
| en → am   | 1    | 13.42     | 36.90     | –         | 11.32            |
| am → en   | 4    | **24.39** | **49.27** | **24.95** | 24.29            |
| am → en   | 1    | 23.53     | 48.39     | –         | 23.45            |

"vs raw refs" scores the folded output against the unfolded FLORES references — the −2.3 BLEU
en→am gap is the cost of homophone folding, not translation errors.

## Limitations

- Amharic output uses folded spellings; words written with ሐ, ሠ, ዐ, ፀ in standard orthography
  come out with ሀ, ሰ, አ, ጸ.
- Trained mostly on religious, web-crawled and mined text; news/encyclopedic input (FLORES) works
  well, conversational and technical text less so.
- Sentences longer than ~120 tokens are truncated. No document context.
- 60M parameters: expect occasional dropped clauses and tense errors on long sentences.

## License

Weights: **CC-BY-NC-4.0** (inherited from the NLLB bitext). Code: MIT.
