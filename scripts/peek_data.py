"""Print per-source stats and sample rows from a built corpus.

Usage: python scripts/peek_data.py <processed_dir>
"""

import json
import sys
from pathlib import Path

import duckdb

out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/processed/tiny")
card = json.loads((out / "data_card.json").read_text(encoding="utf-8"))
for s in card["sources"]:
    print(f"{s['name']:14} read={s['read']:>9,} kept={s['kept']:>9,}  rejected={s['rejected']}")
print("dedup:", {k: v for k, v in card["dedup"].items() if k != "by_source"})
print("splits:", card["splits"])

con = duckdb.connect()
train = str(out / "train.parquet").replace("\\", "/")
for (src,) in con.execute(f"SELECT DISTINCT source FROM '{train}' ORDER BY 1").fetchall():
    n = con.execute(f"SELECT count(*) FROM '{train}' WHERE source = ?", [src]).fetchone()
    print(f"\n--- {src} ({n[0] if n else '?'} rows)")
    rows = con.execute(
        f"SELECT en, am FROM '{train}' WHERE source = ? ORDER BY hash(en) LIMIT 3", [src]
    ).fetchall()
    for en, am in rows:
        print("EN:", en[:120])
        print("AM:", am[:120])
valid = str(out / "valid.parquet").replace("\\", "/")
print("\n--- valid (FLORES dev)")
for en, am, am_raw in con.execute(f"SELECT en, am, am_raw FROM '{valid}' LIMIT 2").fetchall():
    print("EN:", en[:120])
    print("AM:", am[:120])
    print("RAW:", am_raw[:120])
