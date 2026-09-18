"""FLORES-200 benchmark: translate both directions, score against normalized and raw references,
write JSON + Markdown next to the checkpoint.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from amnmt.core.logging import get_logger
from amnmt.evaluation.metrics import Scores, score
from amnmt.inference.translator import Direction, Translator

log = get_logger(__name__)

DIRECTIONS: tuple[Direction, ...] = ("en-am", "am-en")


def load_split(parquet: Path) -> dict[str, list[str]]:
    table = pq.read_table(parquet, columns=["en", "am", "en_raw", "am_raw"])
    return {name: table.column(name).to_pylist() for name in table.column_names}


def evaluate(
    translator: Translator,
    split_parquet: Path,
    beam_size: int = 4,
    length_penalty: float = 1.0,
    spbleu: bool = False,
    limit: int | None = None,
    n_examples: int = 5,
) -> dict[str, Any]:
    data = load_split(split_parquet)
    if limit:
        data = {k: v[:limit] for k, v in data.items()}
    results: dict[str, Any] = {
        "split": split_parquet.stem,
        "n": len(data["en"]),
        "beam_size": beam_size,
        "length_penalty": length_penalty,
        "directions": {},
    }
    for direction in DIRECTIONS:
        src_lang, tgt_lang = direction.split("-")
        t0 = time.perf_counter()
        hyps = translator.translate(
            data[src_lang], direction, beam_size=beam_size, length_penalty=length_penalty
        )
        seconds = time.perf_counter() - t0
        vs_norm: Scores = score(hyps, data[tgt_lang], spbleu=spbleu)
        vs_raw: Scores = score(hyps, data[f"{tgt_lang}_raw"], spbleu=spbleu)
        results["directions"][direction] = {
            "vs_normalized_ref": vs_norm.as_dict(),
            "vs_raw_ref": vs_raw.as_dict(),
            "seconds": round(seconds, 1),
            "examples": [
                {"src": s, "ref": r, "hyp": h}
                for s, r, h in zip(
                    data[src_lang][:n_examples],
                    data[tgt_lang][:n_examples],
                    hyps[:n_examples],
                    strict=True,
                )
            ],
        }
        log.info(
            "%s  BLEU %.2f  chrF++ %.2f%s  (raw ref: BLEU %.2f chrF++ %.2f)  %.0fs",
            direction,
            vs_norm.bleu,
            vs_norm.chrf,
            f"  spBLEU {vs_norm.spbleu:.2f}" if vs_norm.spbleu is not None else "",
            vs_raw.bleu,
            vs_raw.chrf,
            seconds,
        )
    return results


def to_markdown(results: dict[str, Any], title: str) -> str:
    lines = [
        f"# {title}",
        "",
        f"FLORES-200 `{results['split']}` · {results['n']} sentences · beam {results['beam_size']} "
        f"· length penalty {results['length_penalty']}",
        "",
        "| direction | BLEU | chrF++ | spBLEU | BLEU (raw ref) | chrF++ (raw ref) | time |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for d, r in results["directions"].items():
        n, w = r["vs_normalized_ref"], r["vs_raw_ref"]
        sp = f"{n['spbleu']:.2f}" if n["spbleu"] is not None else "–"
        lines.append(
            f"| {d} | {n['bleu']:.2f} | {n['chrf']:.2f} | {sp} | {w['bleu']:.2f} | {w['chrf']:.2f} "
            f"| {r['seconds']:.0f}s |"
        )
    for d, r in results["directions"].items():
        lines += ["", f"## Examples — {d}", ""]
        for ex in r["examples"]:
            lines += [f"- **src** {ex['src']}", f"  **ref** {ex['ref']}", f"  **hyp** {ex['hyp']}"]
    return "\n".join(lines) + "\n"


def run_and_save(
    checkpoint: Path,
    split_parquet: Path,
    out_dir: Path,
    tokenizer: Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    translator = Translator.from_checkpoint(checkpoint, tokenizer)
    results = evaluate(translator, split_parquet, **kwargs)
    results["checkpoint"] = str(checkpoint)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"eval_{split_parquet.stem}_beam{results['beam_size']}"
    (out_dir / f"{stem}.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / f"{stem}.md").write_text(
        to_markdown(results, f"{checkpoint.parent.name} / {checkpoint.name}"), encoding="utf-8"
    )
    log.info("wrote %s", out_dir / f"{stem}.md")
    return results
