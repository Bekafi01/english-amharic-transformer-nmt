"""Show how the trained tokenizer segments a few sentences.

Usage: python scripts/peek_tokens.py <tokenizer.json>
"""

import sys

from amnmt.tokenization.tokenizer import Tokenizer

DEFAULT = "artifacts/tiny/tokenizer/tokenizer.json"
tok = Tokenizer.from_file(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
for s in [
    "The children are playing in the garden.",
    "Addis Ababa is the capital of Ethiopia.",
    "ልጆቹ በአትክልት ስፍራ ውስጥ እየተጫወቱ ነው።",
    "አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት።",
    "በ1992 ዓ.ም. እንደ ኢትዮጵያ አቆጣጠር",
]:
    ids = tok.encode(s)
    print(f"{len(ids):3d} | " + " ".join(tok.id_to_token(i) for i in ids))
