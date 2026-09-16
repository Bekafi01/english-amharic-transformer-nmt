"""Production translation inference engine for bidirectional English-Amharic NMT."""

from pathlib import Path
from typing import Any

import torch

from src.nmt_engine.data.preprocessor import EthiopicNormalizer
from src.nmt_engine.data.tokenizer import JointBpeTokenizer
from src.nmt_engine.inference.decoding import beam_search_decode, greedy_decode
from src.nmt_engine.models.transformer import Seq2SeqTransformer
from src.nmt_engine.utils.logging import get_logger

logger = get_logger(__name__)


class Translator:
    """Production translation interface wrapping Seq2SeqTransformer and JointBpeTokenizer.

    Supports:
      - Bidirectional translation (English <-> Amharic)
      - Greedy Search and Length-Normalized Beam Search decoding
      - Repetition penalties to prevent degenerate token loops
      - Linguistic Ethiopic homophone and punctuation normalization
      - Batch translation for evaluation pipelines

    Args:
        model: Pre-instantiated Seq2SeqTransformer instance or None.
        model_path: Path to serialized PyTorch model checkpoint (.pt).
        tokenizer: Pre-instantiated JointBpeTokenizer or None.
        tokenizer_path: Path to serialized BPE tokenizer JSON artifact.
        device: Target compute device ('auto', 'cuda', 'mps', 'cpu', or torch.device).
        beam_size: Default beam width for beam search decoding (default: 4).
        length_penalty: Alpha parameter for length normalization (default: 0.6).
        repetition_penalty: Multiplier penalizing repeated subwords (default: 1.2).
        max_len: Maximum decoded sequence length limit (default: 128).
    """

    def __init__(
        self,
        model: Seq2SeqTransformer | None = None,
        model_path: str | Path | None = None,
        tokenizer: JointBpeTokenizer | None = None,
        tokenizer_path: str | Path | None = None,
        device: str | torch.device = "auto",
        beam_size: int = 4,
        length_penalty: float = 0.6,
        repetition_penalty: float = 1.2,
        max_len: int = 128,
        vocab_size: int = 32000,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
    ) -> None:
        self.device = self._resolve_device(device)
        self.beam_size = beam_size
        self.length_penalty = length_penalty
        self.repetition_penalty = repetition_penalty
        self.max_len = max_len
        self.normalizer = EthiopicNormalizer()

        # 1. Initialize or load Tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif tokenizer_path is not None:
            self.tokenizer = JointBpeTokenizer.load(tokenizer_path)
        else:
            self.tokenizer = JointBpeTokenizer()

        # 2. Initialize or load Model
        if model is not None:
            self.model = model.to(self.device)
        elif model_path is not None:
            self.model = self._load_model_from_checkpoint(
                model_path=model_path,
                vocab_size=vocab_size,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                d_ff=d_ff,
                pad_id=self.tokenizer.pad_id,
                max_len=max_len,
            )
        else:
            self.model = Seq2SeqTransformer(
                vocab_size=vocab_size,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                d_ff=d_ff,
                max_len=max_len,
                pad_id=self.tokenizer.pad_id,
                tie_weights=True,
            ).to(self.device)

        self.model.eval()

    def _resolve_device(self, device: str | torch.device) -> torch.device:
        """Determines target PyTorch device based on user preference and hardware."""
        if isinstance(device, torch.device):
            return device
        dev_str = str(device).lower()
        if dev_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(dev_str)

    def _load_model_from_checkpoint(
        self,
        model_path: str | Path,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        pad_id: int,
        max_len: int = 128,
    ) -> Seq2SeqTransformer:
        """Instantiates Seq2SeqTransformer and restores weights from checkpoint."""
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"Model checkpoint not found at: {p.resolve()}")

        checkpoint: dict[str, Any] = torch.load(p, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("model_state", checkpoint)

        # Auto-detect max_len from checkpoint positional encoding buffer if available
        if "pos_encoding.pe" in state_dict and state_dict["pos_encoding.pe"].ndim == 3:
            max_len = state_dict["pos_encoding.pe"].shape[1]

        model = Seq2SeqTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_len=max_len,
            pad_id=pad_id,
            tie_weights=True,
        ).to(self.device)

        model.load_state_dict(state_dict)
        logger.info(f"Loaded Seq2SeqTransformer checkpoint from {p.resolve()} to {self.device} (max_len={max_len}).")
        return model

    def _standardize_language_code(self, lang: str) -> str:
        """Normalizes language identifier to standard ISO 639-1 code ('en' or 'am')."""
        l_norm = lang.strip().lower()
        if l_norm in ("en", "eng", "english"):
            return "en"
        if l_norm in ("am", "amh", "amharic"):
            return "am"
        raise ValueError(f"Unsupported language code: '{lang}'. Expected 'en' or 'am'.")

    def translate(
        self,
        text: str,
        source_lang: str = "en",
        target_lang: str = "am",
        method: str = "beam",
        beam_size: int | None = None,
        length_penalty: float | None = None,
        repetition_penalty: float | None = None,
        max_len: int | None = None,
    ) -> str:
        """Translates a single sentence between English and Amharic.

        Args:
            text: Input text string to translate.
            source_lang: Source language ('en' or 'am').
            target_lang: Target language ('en' or 'am').
            method: Decoding strategy ('beam' or 'greedy').
            beam_size: Optional override for beam search width.
            length_penalty: Optional override for length normalization penalty alpha.
            repetition_penalty: Optional override for repetition penalty theta.
            max_len: Optional override for maximum sequence length.

        Returns:
            Translated UTF-8 string with special tokens stripped.
        """
        if not text or not text.strip():
            return ""

        src_l = self._standardize_language_code(source_lang)
        tgt_l = self._standardize_language_code(target_lang)
        if src_l == tgt_l:
            return text.strip()

        # Linguistic normalization for Amharic source text
        clean_text = text.strip()
        if src_l == "am":
            clean_text = self.normalizer.normalize(clean_text)

        # Direction prefix tag
        add_direction = "to_am" if tgt_l == "am" else "to_en"

        # Encode: [direction_id, token_1, ..., token_L, eos_id]
        src_ids = self.tokenizer.encode(
            clean_text,
            add_direction=add_direction,
            add_bos=False,
            add_eos=True,
        )

        b_size = beam_size or self.beam_size
        l_pen = length_penalty if length_penalty is not None else self.length_penalty
        r_pen = repetition_penalty if repetition_penalty is not None else self.repetition_penalty
        m_len = max_len or self.max_len

        if method.lower() == "greedy":
            out_ids = greedy_decode(
                model=self.model,
                src_ids=src_ids,
                max_len=m_len,
                pad_id=self.tokenizer.pad_id,
                bos_id=self.tokenizer.bos_id,
                eos_id=self.tokenizer.eos_id,
                device=self.device,
            )
        else:
            out_ids = beam_search_decode(
                model=self.model,
                src_ids=src_ids,
                beam_size=b_size,
                max_len=m_len,
                length_penalty=l_pen,
                repetition_penalty=r_pen,
                pad_id=self.tokenizer.pad_id,
                bos_id=self.tokenizer.bos_id,
                eos_id=self.tokenizer.eos_id,
                device=self.device,
            )

        # Detokenize and filter special tokens
        return self.tokenizer.decode(out_ids, skip_special_tokens=True).strip()

    def translate_batch(
        self,
        texts: list[str],
        source_lang: str = "en",
        target_lang: str = "am",
        method: str = "beam",
        beam_size: int | None = None,
        length_penalty: float | None = None,
        repetition_penalty: float | None = None,
        max_len: int | None = None,
    ) -> list[str]:
        """Translates a batch of sentences sequentially using the configured decoding policy.

        Args:
            texts: List of text strings to translate.
            source_lang: Source language code.
            target_lang: Target language code.
            method: Decoding strategy ('beam' or 'greedy').
            beam_size: Optional beam size override.
            length_penalty: Optional length penalty override.
            repetition_penalty: Optional repetition penalty override.
            max_len: Optional max length cutoff.

        Returns:
            List of translated UTF-8 strings.
        """
        translations = []
        for text in texts:
            trans = self.translate(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                method=method,
                beam_size=beam_size,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                max_len=max_len,
            )
            translations.append(trans)
        return translations
