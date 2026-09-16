import json

nb_path = "notebooks/04_evaluation_and_benchmarks.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, line in enumerate(nb["cells"][4]["source"]):
    print(f"{i:2d}: {line}", end="")
