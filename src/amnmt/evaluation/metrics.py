"""Corpus-level MT metrics via sacreBLEU. Higher is better for all three.

- BLEU (13a tokenizer): the classic number; under-tokenizes Ethiopic punctuation.
- spBLEU (`flores200` SentencePiece tokenizer): the NLLB/FLORES standard, fair across scripts.
  Downloads a small SPM model on first use, so it is opt-in.
- chrF++ (chrF with word bigrams): most reliable for morphologically rich targets like Amharic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from sacrebleu.metrics.bleu import BLEU
from sacrebleu.metrics.chrf import CHRF


@dataclass(frozen=True)
class Scores:
    bleu: float
    chrf: float
    spbleu: float | None = None

    def as_dict(self) -> dict[str, float | None]:
        return asdict(self)


def bleu(hyps: list[str], refs: list[str], tokenize: str = "13a") -> float:
    return float(BLEU(tokenize=tokenize).corpus_score(hyps, [refs]).score)


def chrf(hyps: list[str], refs: list[str]) -> float:
    return float(CHRF(word_order=2).corpus_score(hyps, [refs]).score)  # chrF++


def score(hyps: list[str], refs: list[str], spbleu: bool = False) -> Scores:
    if len(hyps) != len(refs):
        raise ValueError(f"{len(hyps)} hypotheses vs {len(refs)} references")
    return Scores(
        bleu=round(bleu(hyps, refs), 2),
        chrf=round(chrf(hyps, refs), 2),
        spbleu=round(bleu(hyps, refs, tokenize="flores200"), 2) if spbleu else None,
    )
