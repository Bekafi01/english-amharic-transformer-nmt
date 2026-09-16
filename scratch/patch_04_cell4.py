import json

nb_path = "notebooks/04_evaluation_and_benchmarks.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find the cell that contains "from src.nmt_engine"
target_idx = None
for idx, cell in enumerate(nb["cells"]):
    content = "".join(cell.get("source", []))
    if "from src.nmt_engine.models.transformer" in content:
        target_idx = idx
        break

print("Target cell index:", target_idx)

updated_cell_source = """import sys
from pathlib import Path
from tokenizers import Tokenizer

# Ensure repository root is in sys.path regardless of whether kernel runs from /notebooks or repo root
for root_candidate in [Path("..").resolve(), Path(".").resolve()]:
    if (root_candidate / "src").exists() and str(root_candidate) not in sys.path:
        sys.path.insert(0, str(root_candidate))

# Load or initialize BPE Tokenizer
if TOKENIZER_PATH.exists():
    raw_tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    VOCAB_SIZE = raw_tok.get_vocab_size()
    PAD_ID = raw_tok.token_to_id("<pad>")
    BOS_ID = raw_tok.token_to_id("<bos>")
    EOS_ID = raw_tok.token_to_id("<eos>")
    TO_EN_ID = raw_tok.token_to_id("<2en>")
    TO_AM_ID = raw_tok.token_to_id("<2am>")
else:
    VOCAB_SIZE = 32000
    PAD_ID, BOS_ID, EOS_ID, TO_EN_ID, TO_AM_ID = 0, 2, 3, 5, 6

print(f"Subword Vocabulary Size: {VOCAB_SIZE:,} tokens")

# Import production classes from src
from src.nmt_engine.models.transformer import Seq2SeqTransformer
from src.nmt_engine.inference.translator import Translator
from src.nmt_engine.evaluation.metrics import compute_all_metrics
from src.nmt_engine.evaluation.evaluator import BenchmarkEvaluator

print("✅ Successfully imported production NMT engine modules.")

# Instantiate Translator engine
if MODEL_PATH.exists() and TOKENIZER_PATH.exists():
    translator = Translator(
        model_path=MODEL_PATH,
        tokenizer_path=TOKENIZER_PATH,
        device=DEVICE,
        beam_size=4,
        length_penalty=0.6,
        repetition_penalty=1.2,
    )
    print(f"Translator successfully initialized with trained weights from {MODEL_PATH}")
else:
    print("Initializing standalone Translator with Xavier weights for pipeline verification...")
    translator = Translator(
        device=DEVICE,
        beam_size=4,
        length_penalty=0.6,
        repetition_penalty=1.2,
    )
"""

nb["cells"][target_idx]["source"] = [line + "\n" for line in updated_cell_source.strip().split("\n")]

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Cell {target_idx} updated successfully!")
