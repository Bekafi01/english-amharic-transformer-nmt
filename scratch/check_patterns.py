import pandas as pd
import re

df = pd.read_parquet('data/raw/flores200_benchmark.parquet')

patterns = [
    r"Give me the same text.*",
    r"Translate.*",
    r"In Amharic.*",
    r"\[.*\]",
]

for p in patterns:
    matches = df['english'].str.contains(p, regex=True).sum()
    print(f"Pattern '{p}': {matches} matches")
