import os
import sys
import math
import random
from pathlib import Path
import torch
import pandas as pd
import numpy as np
import sacrebleu

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPO_ROOT = Path("..").resolve() if Path("../src").exists() else Path(".").resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_DIR = Path("..") if Path("../checkpoints").exists() else Path(".")

CKPT_DIR = BASE_DIR / "checkpoints"
MODEL_PATH = CKPT_DIR / "best_model.pt" if (CKPT_DIR / "best_model.pt").exists() else CKPT_DIR / "latest.pt"
TOKENIZER_PATH = BASE_DIR / "artifacts" / "tokenizers" / "joint_bpe_32k.json"
FLORES_PATH = BASE_DIR / "data" / "raw" / "flores200_benchmark.parquet"

print("1. Artifacts check:")
print("Model:", MODEL_PATH.exists())
print("Tokenizer:", TOKENIZER_PATH.exists())
print("FLORES:", FLORES_PATH.exists())

from src.nmt_engine.inference.translator import Translator
from src.nmt_engine.evaluation.evaluator import BenchmarkEvaluator

translator = Translator(
    model_path=MODEL_PATH,
    tokenizer_path=TOKENIZER_PATH,
    device=DEVICE,
    beam_size=4,
    length_penalty=0.6,
    repetition_penalty=1.2,
)
print("2. Translator initialized!")

# Test translate
res = translator.translate("The students are learning in the classroom.", source_lang="en", target_lang="am")
print("3. Test translation EN->AM:", res)

# Test Evaluator with 5 samples
evaluator = BenchmarkEvaluator(translator=translator)
summary = evaluator.evaluate_flores(
    benchmark_parquet=FLORES_PATH,
    max_samples=5,
    method="beam",
    output_dir=BASE_DIR / "artifacts" / "evaluation",
)
print("4. Evaluation summary en2am:", summary["en2am"]["metrics"])
print("5. Evaluation summary am2en:", summary["am2en"]["metrics"])
print("ALL TESTS PASSED SUCCESSFULLY!")
