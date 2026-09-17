# Phase 1 — Data (tiny, local run)

- Date: 2026-09-17
- Command: `uv run amnmt data build -c configs/tiny.yaml` (Windows, CPU, real network)
- Sources: first 5,000 pairs of mt560, opus100, tanzil, ccaligned; FLORES-200 from official tarball

## Result

| source    | read  | kept  | main reject reasons                                   |
| --------- | ----- | ----- | ----------------------------------------------------- |
| mt560     | 5,000 | 4,728 | ratio 227, am_script 40                               |
| opus100   | 5,000 | 4,050 | ratio 322, am_script 312, identical 150, markup 130   |
| tanzil    | 5,000 | 4,977 | ratio 23                                              |
| ccaligned | 5,000 | 4,216 | am_script 682, en_script 43, ratio 41, url 15         |

- Dedup: 17,971 → 17,216 (−755 exact/near-exact on normalized key)
- Splits: train 16,716 · train_holdout 500 · valid 997 (FLORES dev) · test 1,012 (FLORES devtest)
- Wall time: ~1 min including downloads; re-run with cached shards: 1 s

## Observations

- opus100 (GNOME/KDE strings) is the noisiest: 19% rejected. Expected; low weight in the full mix.
- ccaligned rejects 14% on `am_script` — mostly Latin/other-script text mislabelled as Amharic.
- Sample rows look correctly detokenized (`word .` → `word.`, `Jehovah 's` → `Jehovah's`,
  `" text "` → `"text"`, `55 : 22` → `55:22`). Known leftover: `Qur ' an` stays as is.
- FLORES normalization removes the Ethiopic wordspace `፡` as designed; raw text is kept in `*_raw` columns.

Result: PASS (pipeline). Full-scale build pending on Colab → `phase1_data_full.md`.
