import sys
from pathlib import Path

# Verify that from notebooks/ CWD, repo root is found
for root_candidate in [Path("..").resolve(), Path(".").resolve()]:
    if (root_candidate / "src").exists() and str(root_candidate) not in sys.path:
        sys.path.insert(0, str(root_candidate))

from src.nmt_engine.models.transformer import Seq2SeqTransformer
from src.nmt_engine.inference.translator import Translator
from src.nmt_engine.evaluation.metrics import compute_all_metrics
from src.nmt_engine.evaluation.evaluator import BenchmarkEvaluator

print("Successfully imported all production modules from notebooks/ CWD!")
