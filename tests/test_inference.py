"""Unit tests for inference: greedy_decode, beam_search_decode, and Translator."""

from pathlib import Path

import pytest
import torch

from src.nmt_engine.data.tokenizer import JointBpeTokenizer
from src.nmt_engine.inference.decoding import beam_search_decode, greedy_decode
from src.nmt_engine.inference.translator import Translator
from src.nmt_engine.models.transformer import Seq2SeqTransformer


@pytest.fixture
def mock_components(tmp_path: Path):
    """Fixture providing a lightweight Seq2SeqTransformer and trained JointBpeTokenizer."""
    texts = [
        "Hello world",
        "ሰላም ዓለም",
        "Good morning",
        "እንደምን አደሩ",
        "Thank you very much",
        "በጣም አመሰግናለሁ",
    ]

    tok = JointBpeTokenizer()
    tok.train_from_iterator(iter([[t] for t in texts]), vocab_size=100, min_frequency=1)
    tok_path = tmp_path / "tokenizer.json"
    tok.save(tok_path)

    model = Seq2SeqTransformer(
        vocab_size=tok.vocab_size,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        pad_id=tok.pad_id,
        tie_weights=True,
    )
    model.eval()

    ckpt_path = tmp_path / "best_model.pt"
    torch.save({"model_state": model.state_dict()}, ckpt_path)

    return model, tok, ckpt_path, tok_path


def test_greedy_decode(mock_components) -> None:
    model, tok, _, _ = mock_components
    src_ids = [tok.to_am_id, 10, 11, tok.eos_id]

    out_ids = greedy_decode(
        model=model,
        src_ids=src_ids,
        max_len=16,
        pad_id=tok.pad_id,
        bos_id=tok.bos_id,
        eos_id=tok.eos_id,
    )

    assert isinstance(out_ids, list)
    assert len(out_ids) >= 1
    assert out_ids[0] == tok.bos_id
    assert len(out_ids) <= 16


def test_beam_search_decode(mock_components) -> None:
    model, tok, _, _ = mock_components
    src_ids = [tok.to_en_id, 12, 13, tok.eos_id]

    out_ids = beam_search_decode(
        model=model,
        src_ids=src_ids,
        beam_size=3,
        max_len=16,
        length_penalty=0.6,
        repetition_penalty=1.2,
        pad_id=tok.pad_id,
        bos_id=tok.bos_id,
        eos_id=tok.eos_id,
    )

    assert isinstance(out_ids, list)
    assert len(out_ids) >= 1
    assert out_ids[0] == tok.bos_id
    assert len(out_ids) <= 16


def test_translator_basic_and_batch(mock_components) -> None:
    model, tok, _, _ = mock_components
    translator = Translator(
        model=model,
        tokenizer=tok,
        device="cpu",
        beam_size=2,
        max_len=16,
    )

    # Empty string
    assert translator.translate("", source_lang="en", target_lang="am") == ""
    assert translator.translate("   ", source_lang="en", target_lang="am") == ""

    # Identical language
    assert translator.translate("Hello", source_lang="en", target_lang="en") == "Hello"

    # Single translation
    out_am = translator.translate(
        "Hello world", source_lang="en", target_lang="am", method="greedy"
    )
    assert isinstance(out_am, str)

    out_en = translator.translate("ሰላም ዓለም", source_lang="am", target_lang="en", method="beam")
    assert isinstance(out_en, str)

    # Batch translation
    batch_out = translator.translate_batch(
        ["Hello", "Good morning"],
        source_lang="en",
        target_lang="am",
        method="greedy",
    )
    assert len(batch_out) == 2
    assert all(isinstance(t, str) for t in batch_out)


def test_translator_from_checkpoint_paths(mock_components) -> None:
    _, tok, ckpt_path, tok_path = mock_components
    translator = Translator(
        model_path=ckpt_path,
        tokenizer_path=tok_path,
        device="cpu",
        vocab_size=tok.vocab_size,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
    )

    res = translator.translate("Thank you", source_lang="english", target_lang="amharic")
    assert isinstance(res, str)


def test_translator_unsupported_language(mock_components) -> None:
    model, tok, _, _ = mock_components
    translator = Translator(model=model, tokenizer=tok, device="cpu")

    with pytest.raises(ValueError, match="Unsupported language code"):
        translator.translate("Hello", source_lang="french", target_lang="am")
