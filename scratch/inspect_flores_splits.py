import pandas as pd

df = pd.read_parquet('data/raw/flores200_benchmark.parquet')
print("Columns:", df.columns.tolist())
print("Split distribution:\n", df['split'].value_counts())
print("Source distribution:\n", df['source'].value_counts())
