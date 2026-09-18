"""Print parameter counts for each config's model section. Usage: python scripts/model_size.py"""

from pathlib import Path

from amnmt.core.config import load_config
from amnmt.model.transformer import Seq2SeqTransformer

for path in sorted(Path("configs").glob("*.yaml")):
    cfg = load_config(path)
    if cfg.model is None or cfg.tokenizer is None:
        continue
    m = Seq2SeqTransformer(cfg.model, cfg.tokenizer.vocab_size, pad_id=0)
    emb = m.embed.weight.numel()
    vocab = cfg.tokenizer.vocab_size
    print(f"{path.stem:8} {m.num_parameters():>12,} params  (embedding {emb:,}, vocab {vocab:,})")
