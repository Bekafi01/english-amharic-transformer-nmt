"""Joint Byte-Pair Encoding (BPE) subword tokenizer for bidirectional English-Amharic NMT."""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
)

from src.nmt_engine.utils.logging import get_logger

logger = get_logger(__name__)


class JointBpeTokenizer:
    """Production-grade Joint Byte-Pair Encoding (BPE) Tokenizer.

    Constructed over a shared 32,000 subword vocabulary across English (Latin) and
    Amharic (Ethiopic) scripts to enable 3-way embedding weight tying ($E_{src} = E_{tgt} = W_{proj}^T$).
    Employs Metaspace pre-tokenization with Byte Fallback to achieve 100% lossless reversibility.
    """

    DEFAULT_SPECIAL_TOKENS: list[str] = [
        "<pad>",  # 0: Padding token
        "<unk>",  # 1: Out-of-vocabulary unknown token
        "<bos>",  # 2: Beginning of sequence
        "<eos>",  # 3: End of sequence
        "<mask_src>",  # 4: Mask token for denoising / word dropout
        "<2en>",  # 5: Translate to English direction prefix
        "<2am>",  # 6: Translate to Amharic direction prefix
    ]

    def __init__(
        self,
        tokenizer: Tokenizer | None = None,
        special_tokens: list[str] | None = None,
    ) -> None:
        self.special_tokens = special_tokens or self.DEFAULT_SPECIAL_TOKENS
        if tokenizer is not None:
            self._tokenizer = tokenizer
        else:
            self._tokenizer = self._build_empty_tokenizer()

    def _build_empty_tokenizer(self) -> Tokenizer:
        """Instantiate an untrained Hugging Face Tokenizer with Metaspace and ByteFallback components."""
        tok = Tokenizer(models.BPE(unk_token="<unk>"))
        tok.normalizer = normalizers.Sequence([normalizers.NFC()])
        tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement=" ", prepend_scheme="always")
        tok.decoder = decoders.Sequence(
            [
                decoders.ByteFallback(),
                decoders.Metaspace(replacement=" ", prepend_scheme="always"),
            ]
        )
        return tok

    @property
    def hf_tokenizer(self) -> Tokenizer:
        """Return the underlying Hugging Face Tokenizer object."""
        return self._tokenizer

    @property
    def vocab_size(self) -> int:
        """Return total vocabulary size including special tokens."""
        return self._tokenizer.get_vocab_size()

    @property
    def pad_id(self) -> int:
        return self._tokenizer.token_to_id("<pad>")

    @property
    def unk_id(self) -> int:
        return self._tokenizer.token_to_id("<unk>")

    @property
    def bos_id(self) -> int:
        return self._tokenizer.token_to_id("<bos>")

    @property
    def eos_id(self) -> int:
        return self._tokenizer.token_to_id("<eos>")

    @property
    def mask_id(self) -> int:
        return self._tokenizer.token_to_id("<mask_src>")

    @property
    def to_en_id(self) -> int:
        return self._tokenizer.token_to_id("<2en>")

    @property
    def to_am_id(self) -> int:
        return self._tokenizer.token_to_id("<2am>")

    def token_to_id(self, token: str) -> int | None:
        """Map a token string to its vocabulary ID."""
        return self._tokenizer.token_to_id(token)

    def id_to_token(self, token_id: int) -> str | None:
        """Map a vocabulary ID to its token string."""
        return self._tokenizer.id_to_token(token_id)

    def get_vocab(self) -> dict[str, int]:
        """Return the complete token-to-id vocabulary mapping."""
        return self._tokenizer.get_vocab()

    @classmethod
    def get_initial_alphabet(cls) -> list[str]:
        """Generate base alphabet covering printable ASCII and the complete Ethiopic Unicode block."""
        ascii_chars = [chr(i) for i in range(32, 127)]
        ethiopic_chars = [chr(i) for i in range(0x1200, 0x1380)]
        extra_geez = [chr(i) for i in range(0x2D80, 0x2DE0)]
        return sorted(set(ascii_chars + ethiopic_chars + extra_geez))

    def train_from_iterator(
        self,
        iterator: Iterable[str | list[str]],
        vocab_size: int = 32000,
        min_frequency: int = 2,
    ) -> None:
        """Train the joint BPE subword model on an in-memory text iterator."""
        initial_alphabet = self.get_initial_alphabet()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=self.special_tokens,
            initial_alphabet=initial_alphabet,
            show_progress=True,
        )

        logger.info(
            f"Training joint BPE tokenizer (vocab_size={vocab_size:,}, min_frequency={min_frequency})..."
        )
        self._tokenizer.train_from_iterator(iterator, trainer=trainer)
        logger.info(f"Tokenizer trained successfully: vocab_size={self.vocab_size:,}.")

    def train_from_files(
        self,
        files: list[str],
        vocab_size: int = 32000,
        min_frequency: int = 2,
    ) -> None:
        """Train the joint BPE subword model from text files on disk."""
        initial_alphabet = self.get_initial_alphabet()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=self.special_tokens,
            initial_alphabet=initial_alphabet,
            show_progress=True,
        )

        logger.info(
            f"Training joint BPE tokenizer from {len(files)} files (vocab_size={vocab_size:,})..."
        )
        self._tokenizer.train(files, trainer=trainer)
        logger.info(f"Tokenizer trained successfully: vocab_size={self.vocab_size:,}.")

    def encode(
        self,
        text: str,
        add_direction: str | None = None,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Tokenize a single string into a sequence of integer subword token IDs.

        Args:
            text: Input string to tokenize.
            add_direction: Optional direction token ('to_en' for <2en>, 'to_am' for <2am>).
            add_bos: Prepend beginning-of-sequence token <bos>.
            add_eos: Append end-of-sequence token <eos>.
        """
        encoding = self._tokenizer.encode(text)
        token_ids = list(encoding.ids)

        if add_direction == "to_en":
            token_ids = [self.to_en_id] + token_ids
        elif add_direction == "to_am":
            token_ids = [self.to_am_id] + token_ids

        if add_bos:
            token_ids = [self.bos_id] + token_ids
        if add_eos:
            token_ids = token_ids + [self.eos_id]

        return token_ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode a list of subword IDs back into a reconstructed string."""
        if skip_special_tokens:
            special_ids = {
                self.pad_id,
                self.unk_id,
                self.bos_id,
                self.eos_id,
                self.mask_id,
                self.to_en_id,
                self.to_am_id,
            }
            token_ids = [tid for tid in token_ids if tid not in special_ids]

        return self._tokenizer.decode(token_ids)

    def encode_batch(
        self,
        texts: list[str],
        add_direction: str | None = None,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[list[int]]:
        """Tokenize a batch of strings into token IDs."""
        return [
            self.encode(t, add_direction=add_direction, add_bos=add_bos, add_eos=add_eos)
            for t in texts
        ]

    def decode_batch(
        self, sequences: list[list[int]], skip_special_tokens: bool = True
    ) -> list[str]:
        """Decode a batch of token ID sequences into reconstructed strings."""
        return [self.decode(s, skip_special_tokens=skip_special_tokens) for s in sequences]

    def save(self, path: str | Path) -> None:
        """Persist tokenizer model and special token metadata to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(path))
        logger.info(f"Tokenizer model saved to: {path}")

    @classmethod
    def load(cls, path: str | Path) -> "JointBpeTokenizer":
        """Load a trained tokenizer from a serialized JSON artifact."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer model file not found: {path}")

        tok = Tokenizer.from_file(str(path))
        instance = cls(tokenizer=tok)
        logger.info(f"Loaded tokenizer from {path} (vocab_size={instance.vocab_size:,}).")
        return instance

    def compute_fertility(self, texts: list[str]) -> tuple[float, float]:
        """Compute subword fertility (subwords per whitespace-separated word).

        Returns:
            Tuple of (mean_fertility, std_fertility).
        """
        fertilities = []
        for t in texts:
            words = len(t.split())
            if words == 0:
                continue
            subwords = len(self._tokenizer.encode(t).ids)
            fertilities.append(subwords / words)

        if not fertilities:
            return 0.0, 0.0
        return float(np.mean(fertilities)), float(np.std(fertilities))

    def compute_oov_rate(self, texts: list[str]) -> tuple[int, int, float]:
        """Evaluate out-of-vocabulary (<unk>) rate across an evaluation set.

        Returns:
            Tuple of (total_tokens, unk_tokens, unk_percentage).
        """
        total_tokens = 0
        unk_tokens = 0
        unk_id = self.unk_id

        for t in texts:
            ids = self._tokenizer.encode(t).ids
            total_tokens += len(ids)
            unk_tokens += ids.count(unk_id)

        unk_rate = (unk_tokens / total_tokens * 100.0) if total_tokens > 0 else 0.0
        return total_tokens, unk_tokens, unk_rate

    def build_bidirectional_dataset(
        self,
        df: pd.DataFrame,
        max_seq_len: int = 128,
    ) -> pd.DataFrame:
        """Convert parallel text bitext into bidirectional tokenized pairs with direction tokens.

        For each row (amharic, english):
        1. am2en: src = [<2en>] + am_ids + [<eos>], tgt = [<bos>] + en_ids + [<eos>]
        2. en2am: src = [<2am>] + en_ids + [<eos>], tgt = [<bos>] + am_ids + [<eos>]
        """
        records = []
        for _, row in df.iterrows():
            am_ids = self._tokenizer.encode(str(row["amharic"])).ids
            en_ids = self._tokenizer.encode(str(row["english"])).ids

            # Enforce max sequence length (reserve 2 slots for prefix & suffix tokens)
            if len(am_ids) > (max_seq_len - 2) or len(en_ids) > (max_seq_len - 2):
                continue

            # Direction 1: Amharic -> English (<2en>)
            records.append(
                {
                    "direction": "am2en",
                    "src_ids": [self.to_en_id] + am_ids + [self.eos_id],
                    "tgt_ids": [self.bos_id] + en_ids + [self.eos_id],
                    "src_len": len(am_ids) + 2,
                    "tgt_len": len(en_ids) + 2,
                }
            )

            # Direction 2: English -> Amharic (<2am>)
            records.append(
                {
                    "direction": "en2am",
                    "src_ids": [self.to_am_id] + en_ids + [self.eos_id],
                    "tgt_ids": [self.bos_id] + am_ids + [self.eos_id],
                    "src_len": len(en_ids) + 2,
                    "tgt_len": len(am_ids) + 2,
                }
            )

        return pd.DataFrame(records)
