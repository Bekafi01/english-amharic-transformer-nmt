import json
import sys

nb_path = "notebooks/04_evaluation_and_benchmarks.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Cell 3 is code cell 2
print("Cell 3 source snippet:")
for i, line in enumerate(nb["cells"][3]["source"]):
    if "from src." in line or "sys.path" in line:
        print(f"  {i}: {line.strip()}")
