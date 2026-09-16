"""Benchmark evaluation engine for English <-> Amharic machine translation."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import sacrebleu

from src.nmt_engine.evaluation.metrics import compute_all_metrics
from src.nmt_engine.inference.translator import Translator
from src.nmt_engine.utils.logging import get_logger

logger = get_logger(__name__)


class BenchmarkEvaluator:
    """Evaluates NMT models on test sets and gold-standard benchmarks (e.g. FLORES-200).

    Computes corpus-level SacreBLEU, chrF++, and TER, and exports formatted markdown
    and JSON evaluation reports with qualitative sample inspections.

    Args:
        translator: Instantiated Translator engine.
    """

    def __init__(self, translator: Translator) -> None:
        self.translator = translator

    def evaluate_dataset(
        self,
        df: pd.DataFrame,
        source_col: str = "english",
        target_col: str = "amharic",
        direction: str = "en2am",
        method: str = "beam",
        max_samples: int | None = None,
        num_sample_examples: int = 5,
    ) -> dict[str, Any]:
        """Runs evaluation over a bitext DataFrame.

        Args:
            df: DataFrame containing parallel source and target text columns.
            source_col: Name of source text column.
            target_col: Name of target gold reference column.
            direction: Translation direction identifier ('en2am' or 'am2en').
            method: Decoding strategy ('beam' or 'greedy').
            max_samples: Optional cap on the number of evaluation pairs to evaluate.
            num_sample_examples: Number of qualitative sample pairs to include in report.

        Returns:
            Dictionary containing metrics, sample size, and qualitative comparison examples.
        """
        if max_samples and len(df) > max_samples:
            df = df.iloc[:max_samples].copy()

        sources = df[source_col].astype(str).tolist()
        references = df[target_col].astype(str).tolist()

        logger.info(
            f"Evaluating {len(sources):,} sentences | Direction: {direction} | Method: {method}..."
        )

        source_lang, target_lang = ("en", "am") if direction == "en2am" else ("am", "en")
        hypotheses = self.translator.translate_batch(
            texts=sources,
            source_lang=source_lang,
            target_lang=target_lang,
            method=method,
        )

        metrics = compute_all_metrics(hypotheses=hypotheses, references=references)
        logger.info(
            f"[{direction.upper()} Results] BLEU: {metrics['bleu']:.2f} | "
            f"chrF++: {metrics['chrf2']:.2f} | TER: {metrics['ter']:.2f}"
        )

        # Extract qualitative examples with sentence-level chrF scores
        samples = []
        for i in range(min(num_sample_examples, len(sources))):
            src_text = sources[i]
            ref_text = references[i]
            hyp_text = hypotheses[i]
            sent_chrf = float(sacrebleu.sentence_chrf(hyp_text, [ref_text], word_order=2).score)
            samples.append(
                {
                    "index": i + 1,
                    "source": src_text,
                    "reference": ref_text,
                    "hypothesis": hyp_text,
                    "chrf2": round(sent_chrf, 2),
                }
            )

        return {
            "direction": direction,
            "method": method,
            "num_samples": len(sources),
            "metrics": metrics,
            "samples": samples,
        }

    def evaluate_flores(
        self,
        benchmark_parquet: str | Path,
        max_samples: int | None = None,
        method: str = "beam",
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Evaluates bidirectional translation against held-out FLORES-200 benchmark.

        Args:
            benchmark_parquet: Path to FLORES-200 parquet file.
            max_samples: Optional cap on evaluated sentence pairs.
            method: Decoding strategy ('beam' or 'greedy').
            output_dir: Optional directory to serialize markdown and JSON reports.

        Returns:
            Dictionary containing bidirectional evaluation metrics.
        """
        p = Path(benchmark_parquet)
        if not p.exists():
            raise FileNotFoundError(f"FLORES-200 benchmark not found at: {p.resolve()}")

        df = pd.read_parquet(p)
        logger.info(f"Loaded {len(df):,} benchmark pairs from {p.resolve()}.")

        # Support both 'english'/'amharic' and 'src'/'tgt' schemas
        en_col = "english" if "english" in df.columns else "en"
        am_col = "amharic" if "amharic" in df.columns else "am"

        # 1. English -> Amharic Evaluation
        en2am_res = self.evaluate_dataset(
            df=df,
            source_col=en_col,
            target_col=am_col,
            direction="en2am",
            method=method,
            max_samples=max_samples,
        )

        # 2. Amharic -> English Evaluation
        am2en_res = self.evaluate_dataset(
            df=df,
            source_col=am_col,
            target_col=en_col,
            direction="am2en",
            method=method,
            max_samples=max_samples,
        )

        summary = {
            "benchmark_source": str(p),
            "method": method,
            "en2am": en2am_res,
            "am2en": am2en_res,
        }

        if output_dir is not None:
            out_p = Path(output_dir)
            out_p.mkdir(parents=True, exist_ok=True)
            self._save_reports(summary, out_p)

        return summary

    def _save_reports(self, summary: dict[str, Any], output_dir: Path) -> None:
        """Serializes markdown and JSON benchmark reports to disk."""
        json_path = output_dir / "flores200_metrics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Exported benchmark metrics JSON to: {json_path}")

        md_path = output_dir / "flores200_benchmark_report.md"
        en2am = summary["en2am"]
        am2en = summary["am2en"]

        md_content = f"""# 🌐 FLORES-200 Benchmark Evaluation Report

Evaluation results on held-out gold benchmark pairs using `{summary["method"]}` decoding.

---

## 📊 Summary Performance Metrics

| Translation Direction | BLEU (↑) | chrF++ (↑) | TER (↓) | Evaluated Samples |
| :--- | :---: | :---: | :---: | :---: |
| **English $\\to$ Amharic** (`en2am`) | **{en2am["metrics"]["bleu"]:.2f}** | **{en2am["metrics"]["chrf2"]:.2f}** | **{en2am["metrics"]["ter"]:.2f}** | {en2am["num_samples"]:,} |
| **Amharic $\\to$ English** (`am2en`) | **{am2en["metrics"]["bleu"]:.2f}** | **{am2en["metrics"]["chrf2"]:.2f}** | **{am2en["metrics"]["ter"]:.2f}** | {am2en["num_samples"]:,} |

---

## 🔍 Qualitative Inspection: English $\\to$ Amharic

| # | Source English | Gold Reference (Amharic) | Model Hypothesis (Amharic) | chrF++ |
| :--- | :--- | :--- | :--- | :---: |
"""
        for s in en2am["samples"]:
            md_content += f"| {s['index']} | {s['source']} | {s['reference']} | {s['hypothesis']} | {s['chrf2']} |\n"

        md_content += """
---

## 🔍 Qualitative Inspection: Amharic $\\to$ English

| # | Source Amharic | Gold Reference (English) | Model Hypothesis (English) | chrF++ |
| :--- | :--- | :--- | :--- | :---: |
"""
        for s in am2en["samples"]:
            md_content += f"| {s['index']} | {s['source']} | {s['reference']} | {s['hypothesis']} | {s['chrf2']} |\n"

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        logger.info(f"Exported benchmark report Markdown to: {md_path}")
