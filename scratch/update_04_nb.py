import json

nb_path = "notebooks/04_evaluation_and_benchmarks.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update Cell 1 (code cell index 1, which is nb['cells'][1])
cell_1_code = """import os
import sys
import math
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sacrebleu

# Verify CUDA / Apple Silicon / CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu")

print("=" * 70)
print("💻 COMPUTATIONAL & INFERENCE ENVIRONMENT")
print("=" * 70)
print(f"• PyTorch Version : {torch.__version__}")
print(f"• Active Device   : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"• GPU Model       : {torch.cuda.get_device_name(0)}")
    print(f"• Total VRAM      : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
print("=" * 70)

# Google Drive Mount if running in Google Colab, or resolve local repository root
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE_DIR = Path("/content/drive/MyDrive/Internship/MT/Data")
    REPO_ROOT = Path(".")
else:
    REPO_ROOT = Path("..").resolve() if Path("../src").exists() else Path(".").resolve()
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    BASE_DIR = REPO_ROOT

# Locate model and tokenizer checkpoints
CKPT_DIR = BASE_DIR / "checkpoints" if (BASE_DIR / "checkpoints" / "best_model.pt").exists() else BASE_DIR / "artifacts" / "checkpoints"
MODEL_PATH = CKPT_DIR / "best_model.pt" if (CKPT_DIR / "best_model.pt").exists() else CKPT_DIR / "latest.pt"
TOKENIZER_PATH = BASE_DIR / "artifacts" / "tokenizers" / "joint_bpe_32k.json"
FLORES_PATH = BASE_DIR / "data" / "raw" / "flores200_benchmark.parquet"

print(f"• Model Checkpoint Path : {MODEL_PATH} ({'Found' if MODEL_PATH.exists() else 'Missing'})")
print(f"• Tokenizer Model Path  : {TOKENIZER_PATH} ({'Found' if TOKENIZER_PATH.exists() else 'Missing'})")
print(f"• FLORES Benchmark Path : {FLORES_PATH} ({'Found' if FLORES_PATH.exists() else 'Missing'})")
print("=" * 70)
"""

nb["cells"][1]["source"] = [line + "\n" for line in cell_1_code.strip().split("\n")]

# Update Cell 4 (eval samples default)
cell_4_source = nb["cells"][4]["source"]
for idx, line in enumerate(cell_4_source):
    if "EVAL_SAMPLES = 200" in line:
        cell_4_source[idx] = "    EVAL_SAMPLES = 50  # Set to 50 for rapid local evaluation, or None for full 6,027 pairs\n"

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("04_evaluation_and_benchmarks.ipynb updated successfully!")
