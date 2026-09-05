"""Unit tests for Joint Byte-Pair Encoding (BPE) subword tokenizer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.nmt_engine.data.tokenizer import JointBpeTokenizer


@pytest.fixture
def sample_bilingual_corpus() -> list[str]:
    """Sample bilingual text collection covering both scripts."""
    return [
        "ጠቅላይ ሚኒስትሩ በአዲስ አበባ አዳዲስ የኢኮኖሚ ማሻሻያዎችን አስታወቁ።",
        "The Prime Minister announced economic reforms in Addis Ababa.",
        "ሰላምና ኢኮኖሚያዊ እድገት ለቀጣናው ውህደት አስፈላጊ ናቸው።",
        "Peace and economic prosperity are essential for regional integration.",
        "ዶክተር አብይ አህመድ ታላቁን የኢትዮጵያ ህዳሴ ግድብ ጎበኙ።",
        "Dr. Abiy Ahmed visited the Grand Ethiopian Renaissance Dam.",
        "ዋጋው 3.14 ዶላር ሲሆን ሰዓቱ 10:30 ነበር፤ ስብሰባውም ተጠናቀቀ።",
        "The price was 3.14 dollars and the time was 10:30; the meeting concluded.",
    ]


def test_special_tokens_canonical_order():
    """Verifies that special token IDs are locked in canonical positions 0 through 6."""
    tok = JointBpeTokenizer()

    # Pre-training: IDs in empty tokenizer
    assert tok.pad_id is not None or tok.special_tokens[0] == "<pad>"
    assert tok.special_tokens == [
        "<pad>",
        "<unk>",
        "<bos>",
        "<eos>",
        "<mask_src>",
        "<2en>",
        "<2am>",
    ]


def test_tokenizer_training_and_special_tokens(sample_bilingual_corpus: list[str]):
    """Verifies BPE training assigns exact canonical special token IDs."""
    tok = JointBpeTokenizer()
    tok.train_from_iterator(sample_bilingual_corpus, vocab_size=500, min_frequency=1)

    assert tok.vocab_size >= 7
    assert tok.pad_id == 0
    assert tok.unk_id == 1
    assert tok.bos_id == 2
    assert tok.eos_id == 3
    assert tok.mask_id == 4
    assert tok.to_en_id == 5
    assert tok.to_am_id == 6


def test_metaspace_lossless_roundtrip(sample_bilingual_corpus: list[str]):
    """Verifies 100% reversible lossless roundtrip reconstruction across both scripts."""
    tok = JointBpeTokenizer()
    tok.train_from_iterator(sample_bilingual_corpus, vocab_size=500, min_frequency=1)

    for text in sample_bilingual_corpus:
        encoded = tok.encode(text)
        decoded = tok.decode(encoded, skip_special_tokens=True)
        assert decoded.strip() == text.strip()


def test_directional_and_sequence_delimiters(sample_bilingual_corpus: list[str]):
    """Verifies correct injection of direction prefix (<2en>, <2am>) and delimiters (<bos>, <eos>)."""
    tok = JointBpeTokenizer()
    tok.train_from_iterator(sample_bilingual_corpus, vocab_size=500, min_frequency=1)

    # Direction: to_en with eos
    ids_to_en = tok.encode("ሰላም ዓለም", add_direction="to_en", add_eos=True)
    assert ids_to_en[0] == tok.to_en_id
    assert ids_to_en[-1] == tok.eos_id

    # Target: bos and eos
    ids_tgt = tok.encode("Hello world", add_bos=True, add_eos=True)
    assert ids_tgt[0] == tok.bos_id
    assert ids_tgt[-1] == tok.eos_id

    # Direction: to_am
    ids_to_am = tok.encode("Hello world", add_direction="to_am")
    assert ids_to_am[0] == tok.to_am_id


def test_fertility_and_oov_computation(sample_bilingual_corpus: list[str]):
    """Verifies subword fertility and OOV rate calculations."""
    tok = JointBpeTokenizer()
    tok.train_from_iterator(sample_bilingual_corpus, vocab_size=500, min_frequency=1)

    mean_fert, std_fert = tok.compute_fertility(sample_bilingual_corpus)
    assert mean_fert >= 1.0

    total_tokens, unks, unk_rate = tok.compute_oov_rate(sample_bilingual_corpus)
    assert total_tokens > 0
    assert unks == 0  # In-corpus text should have 0 UNKs
    assert unk_rate == 0.0


def test_save_and_load(tmp_path: Path, sample_bilingual_corpus: list[str]):
    """Verifies saving tokenizer to JSON and reloading preserves vocabulary and special token IDs."""
    tok = JointBpeTokenizer()
    tok.train_from_iterator(sample_bilingual_corpus, vocab_size=500, min_frequency=1)

    model_path = tmp_path / "tokenizer.json"
    tok.save(model_path)
    assert model_path.exists()

    reloaded = JointBpeTokenizer.load(model_path)
    assert reloaded.vocab_size == tok.vocab_size
    assert reloaded.pad_id == tok.pad_id
    assert reloaded.to_en_id == tok.to_en_id
    assert reloaded.to_am_id == tok.to_am_id


def test_build_bidirectional_dataset(sample_bilingual_corpus: list[str]):
    """Verifies transformation of parallel DataFrame into bidirectional seq2seq records."""
    tok = JointBpeTokenizer()
    tok.train_from_iterator(sample_bilingual_corpus, vocab_size=500, min_frequency=1)

    df_bitext = pd.DataFrame(
        {
            "amharic": ["ሰላም ዓለም", "ደህና ሁን"],
            "english": ["Hello world", "Goodbye"],
        }
    )

    df_bidir = tok.build_bidirectional_dataset(df_bitext, max_seq_len=64)

    # 2 rows x 2 directions = 4 rows
    assert len(df_bidir) == 4
    assert set(df_bidir["direction"].unique()) == {"am2en", "en2am"}

    am2en_row = df_bidir[df_bidir["direction"] == "am2en"].iloc[0]
    assert am2en_row["src_ids"][0] == tok.to_en_id
    assert am2en_row["tgt_ids"][0] == tok.bos_id
    assert am2en_row["tgt_ids"][-1] == tok.eos_id
