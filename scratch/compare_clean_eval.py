import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import pandas as pd
import re
from src.nmt_engine.inference.translator import Translator
from src.nmt_engine.evaluation.evaluator import BenchmarkEvaluator

df = pd.read_parquet('data/raw/flores200_benchmark.parquet')
clean_df = df.copy()
clean_df['english'] = clean_df['english'].str.replace(r"\s*Give me the same text in Amharic\.?", "", regex=True).str.strip()

print("Original row 0 EN:", repr(df.iloc[0]['english']))
print("Cleaned row 0 EN :", repr(clean_df.iloc[0]['english']))

translator = Translator(
    model_path='checkpoints/best_model.pt',
    tokenizer_path='artifacts/tokenizers/joint_bpe_32k.json',
    device='cpu',
    beam_size=4
)
evaluator = BenchmarkEvaluator(translator=translator)

# Test 10 pairs on dirty vs clean
dirty_en2am = evaluator.evaluate_dataset(df.iloc[:10], 'english', 'amharic', direction='en2am', method='beam')
clean_en2am = evaluator.evaluate_dataset(clean_df.iloc[:10], 'english', 'amharic', direction='en2am', method='beam')

dirty_am2en = evaluator.evaluate_dataset(df.iloc[:10], 'amharic', 'english', direction='am2en', method='beam')
clean_am2en = evaluator.evaluate_dataset(clean_df.iloc[:10], 'amharic', 'english', direction='am2en', method='beam')

print("DIRTY EN2AM BLEU:", dirty_en2am['metrics']['bleu'], "chrF++:", dirty_en2am['metrics']['chrf2'])
print("CLEAN EN2AM BLEU:", clean_en2am['metrics']['bleu'], "chrF++:", clean_en2am['metrics']['chrf2'])
print("DIRTY AM2EN BLEU:", dirty_am2en['metrics']['bleu'], "chrF++:", dirty_am2en['metrics']['chrf2'])
print("CLEAN AM2EN BLEU:", clean_am2en['metrics']['bleu'], "chrF++:", clean_am2en['metrics']['chrf2'])
