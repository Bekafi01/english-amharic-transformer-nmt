import pandas as pd

df = pd.read_parquet('data/raw/flores200_benchmark.parquet')
print("Total rows:", len(df))
has_prompt = df['english'].str.contains("Give me the same text", regex=False).sum()
print("Rows with 'Give me the same text':", has_prompt)

for i in range(5):
    en = df.iloc[i]['english']
    am = df.iloc[i]['amharic']
    print(f"--- ROW {i} ---")
    print("EN:", en[:120])
    print("AM (len):", len(am))
