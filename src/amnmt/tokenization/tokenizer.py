"""Thin wrapper over a trained HF `tokenizers` model.

`encode` returns content ids only; the training/inference layers add `<s>`, `</s>` and the
target-language tag themselves so the tokenizer stays a pure text<->ids mapping.
"""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer as _HFTokenizer

PAD, BOS, EOS, UNK, TO_AM, TO_EN = "<pad>", "<s>", "</s>", "<unk>", "<2am>", "<2en>"
SPECIAL_TOKENS = [PAD, BOS, EOS, UNK, TO_AM, TO_EN]  # ids 0..5, in this order
BYTE_TOKENS = [f"<0x{i:02X}>" for i in range(256)]  # ids 6..261, used by byte fallback
LANG_TAG = {"am": TO_AM, "en": TO_EN}


class Tokenizer:
    def __init__(self, hf: _HFTokenizer) -> None:
        self._hf = hf
        maybe = {tok: hf.token_to_id(tok) for tok in SPECIAL_TOKENS}
        missing = [tok for tok, i in maybe.items() if i is None]
        if missing:
            raise ValueError(f"tokenizer lacks special tokens {missing}")
        ids: dict[str, int] = {tok: int(i) for tok, i in maybe.items() if i is not None}
        self.pad_id = ids[PAD]
        self.bos_id = ids[BOS]
        self.eos_id = ids[EOS]
        self.unk_id = ids[UNK]
        self._lang_ids = {lang: ids[tag] for lang, tag in LANG_TAG.items()}
        self._special_ids = frozenset(ids.values())
        byte_ids = [hf.token_to_id(t) for t in BYTE_TOKENS]
        self._byte_ids = frozenset(i for i in byte_ids if i is not None)

    @classmethod
    def from_file(cls, path: str | Path) -> Tokenizer:
        return cls(_HFTokenizer.from_file(str(path)))

    @property
    def vocab_size(self) -> int:
        return int(self._hf.get_vocab_size())

    def lang_id(self, lang: str) -> int:
        return self._lang_ids[lang]

    def encode(self, text: str) -> list[int]:
        return list(self._hf.encode(text, add_special_tokens=False).ids)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [list(e.ids) for e in self._hf.encode_batch(texts, add_special_tokens=False)]

    def decode(self, ids: list[int]) -> str:
        # Byte tokens are registered as "special" by the trainer, so we cannot use
        # skip_special_tokens=True; drop only our real specials instead.
        content = [i for i in ids if i not in self._special_ids]
        return str(self._hf.decode(content, skip_special_tokens=False))

    def is_byte_token(self, token_id: int) -> bool:
        return token_id in self._byte_ids

    def id_to_token(self, token_id: int) -> str:
        tok = self._hf.id_to_token(token_id)
        return tok if tok is not None else UNK
