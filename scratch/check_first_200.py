import pandas as pd

df = pd.read_parquet('data/raw/flores200_benchmark.parquet')
first_200 = df.iloc[:200]
num_contaminated = first_200['english'].str.contains("Give me the same text", regex=False).sum()
print(f"In the first 200 rows: {num_contaminated}/200 ({num_contaminated/200*100:.1f}%) have prompt contamination!")
