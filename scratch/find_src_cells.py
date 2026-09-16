import json

nb_path = "notebooks/04_evaluation_and_benchmarks.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "from src." in src or "Seq2SeqTransformer" in src:
        print(f"Cell {i} [{cell['cell_type']}]:")
        for line_no, line in enumerate(cell.get("source", [])):
            if "src" in line or "sys.path" in line:
                print(f"  line {line_no}: {line.strip()}")
