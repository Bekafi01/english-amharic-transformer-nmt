"""User-facing translation: text in, text out. Loads a trainer checkpoint + tokenizer, applies the
same normalization the corpus went through (the tokenizer never saw unfolded Amharic), and runs
beam search in length-sorted batches.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch import Tensor

from amnmt.core.config import ModelConfig
from amnmt.data.normalize import normalize_amharic, normalize_english
from amnmt.inference.beam import BeamConfig, beam_search
from amnmt.model.transformer import Seq2SeqTransformer
from amnmt.tokenization.tokenizer import Tokenizer

Direction = Literal["en-am", "am-en"]
_LANGS: dict[str, tuple[str, str]] = {"en-am": ("en", "am"), "am-en": ("am", "en")}


def default_tokenizer_path(checkpoint: Path) -> Path:
    """<artifacts>/runs/<run>/<ckpt>.pt -> <artifacts>/tokenizer/tokenizer.json"""
    return checkpoint.resolve().parent.parent.parent / "tokenizer" / "tokenizer.json"


class Translator:
    def __init__(
        self,
        model: Seq2SeqTransformer,
        tok: Tokenizer,
        device: torch.device,
        fold_homophones: bool = True,
        numerals_to_digits: bool = True,
    ) -> None:
        self.model = model.to(device).eval()
        self.tok = tok
        self.device = device
        self.fold = fold_homophones
        self.digits = numerals_to_digits
        self.forbidden = (tok.pad_id, tok.bos_id, tok.unk_id, tok.lang_id("am"), tok.lang_id("en"))

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        tokenizer: str | Path | None = None,
        device: str | torch.device | None = None,
    ) -> Translator:
        ckpt_path = Path(checkpoint)
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
        cfg = ckpt["config"]
        model = Seq2SeqTransformer(
            ModelConfig.model_validate(cfg["model"]), ckpt["vocab_size"], ckpt["pad_id"]
        )
        model.load_state_dict(ckpt["model"])
        tok = Tokenizer.from_file(tokenizer or default_tokenizer_path(ckpt_path))
        norm = (cfg.get("data") or {}).get("normalize") or {}
        return cls(
            model,
            tok,
            dev,
            fold_homophones=norm.get("fold_homophones", True),
            numerals_to_digits=norm.get("numerals_to_digits", True),
        )

    # ------------------------------------------------------------------ pieces

    def normalize(self, text: str, lang: str) -> str:
        if lang == "am":
            return normalize_amharic(text, fold=self.fold, digits=self.digits)
        return normalize_english(text)

    def encode_source(self, texts: list[str], direction: Direction) -> Tensor:
        src_lang, tgt_lang = _LANGS[direction]
        tag, eos, pad = self.tok.lang_id(tgt_lang), self.tok.eos_id, self.tok.pad_id
        limit = self.model.cfg.max_len - 2
        ids = [[tag, *self.tok.encode(self.normalize(t, src_lang))[:limit], eos] for t in texts]
        width = max(len(x) for x in ids)
        return torch.tensor([x + [pad] * (width - len(x)) for x in ids], device=self.device)

    def decode_target(self, tokens: list[int]) -> str:
        return self.tok.decode(tokens)

    # ------------------------------------------------------------------ public

    def translate(
        self,
        texts: list[str],
        direction: Direction,
        beam_size: int = 4,
        length_penalty: float = 1.0,
        batch_size: int = 32,
    ) -> list[str]:
        if direction not in _LANGS:
            raise ValueError(f"direction must be one of {list(_LANGS)}, got {direction!r}")
        out = [""] * len(texts)
        nonempty = [i for i, t in enumerate(texts) if t.strip()]
        # Length-sorted batches waste far less padding than input order.
        nonempty.sort(key=lambda i: len(texts[i]))
        cfg = BeamConfig(
            beam_size=beam_size, length_penalty=length_penalty, forbidden_ids=self.forbidden
        )
        for start in range(0, len(nonempty), batch_size):
            idx = nonempty[start : start + batch_size]
            src = self.encode_source([texts[i] for i in idx], direction)
            hyps = beam_search(self.model, src, self.tok.bos_id, self.tok.eos_id, cfg)
            for i, h in zip(idx, hyps, strict=True):
                out[i] = self.decode_target(h.tokens)
        return out
