import json
with open('notebooks/03_model_and_training.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
for line in nb['cells'][6]['source']:
    if any(k in line for k in ['TOKENIZED_DATA_DIR', 'Path', 'val', 'train', 'parquet']):
        print(line.strip())
