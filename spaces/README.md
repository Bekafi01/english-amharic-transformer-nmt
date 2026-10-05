---
title: English ↔ Amharic Translation
emoji: 🇪🇹
colorFrom: green
colorTo: yellow
sdk: streamlit
sdk_version: "1.39.0"
app_file: app.py
pinned: false
license: mit
short_description: From-scratch 60M Transformer, en↔am, trained on a Colab T4
models:
  - Bekafi01/amnmt-en-am-base
---

# English ↔ Amharic Translation

Live demo of [english-amharic-transformer-nmt](https://github.com/Bekafi01/english-amharic-transformer-nmt):
a from-scratch PyTorch Transformer (60M parameters, one model for both directions) trained on 4.4M
public sentence pairs in 15 GPU-hours on a free Colab T4.

FLORES-200 devtest, beam 4: **en→am 14.5 BLEU / 37.7 chrF++ · am→en 24.4 BLEU / 49.3 chrF++**.

Weights: [Bekafi01/amnmt-en-am-base](https://huggingface.co/Bekafi01/amnmt-en-am-base).
Research/non-commercial use (the NLLB training bitext is CC-BY-NC-4.0).
