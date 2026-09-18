import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from amnmt.core.config import Config, SubsetConfig
from amnmt.tokenization.tokenizer import BYTE_TOKENS, SPECIAL_TOKENS, Tokenizer
from amnmt.tokenization.train import train, train_from_texts

EN = [
    "The quick brown fox jumps over the lazy dog.",
    "Good morning, how are you today?",
    "Water boils at 100 degrees; ice melts at 0.",
    'She said: "I\'m fine, thanks!"',
    "Addis Ababa is the capital of Ethiopia.",
]
AM = [
    "ፈጣኑ ቡናማ ቀበሮ ሰነፉን ውሻ ዘሎ አለፈ።",
    "እንደምን አደርክ፣ ዛሬ እንዴት ነህ?",
    "ውሃ በ100 ዲግሪ ይፈላል፤ በረዶ በ0 ይቀልጣል።",
    'እሷ አለች፦ "ደህና ነኝ፣ አመሰግናለሁ!"',
    "አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት።",
]


@pytest.fixture(scope="module")
def tok(tmp_path_factory: pytest.TempPathFactory) -> Tokenizer:
    hf = train_from_texts(iter((EN + AM) * 20), vocab_size=600, min_frequency=1)
    path = tmp_path_factory.mktemp("tok") / "tokenizer.json"
    hf.save(str(path))
    return Tokenizer.from_file(path)


def test_special_and_byte_token_ids_are_fixed(tok: Tokenizer) -> None:
    assert (tok.pad_id, tok.bos_id, tok.eos_id, tok.unk_id) == (0, 1, 2, 3)
    assert tok.lang_id("am") == 4 and tok.lang_id("en") == 5
    assert [tok.id_to_token(i) for i in range(6)] == SPECIAL_TOKENS
    assert tok.id_to_token(6) == BYTE_TOKENS[0] and tok.id_to_token(261) == BYTE_TOKENS[-1]
    assert tok.is_byte_token(6) and tok.is_byte_token(261) and not tok.is_byte_token(5)


@pytest.mark.parametrize("text", EN + AM)
def test_roundtrip_in_vocab(tok: Tokenizer, text: str) -> None:
    ids = tok.encode(text)
    assert ids and all(i >= len(SPECIAL_TOKENS) for i in ids)  # no specials in content ids
    assert tok.decode(ids) == text


@pytest.mark.parametrize(
    "text",
    [
        "Unseen glyphs: 你好 🌍 ñandú",
        "ሰላም 🙂 ለዓለም",
        "Ge'ez numerals ፲፱፻፺፪ survive untouched",
        "tabs\tand  double  spaces are lossless too",
    ],
)
def test_roundtrip_out_of_vocab_via_byte_fallback(tok: Tokenizer, text: str) -> None:
    ids = tok.encode(text)
    assert tok.unk_id not in ids
    assert any(tok.is_byte_token(i) for i in ids)
    assert tok.decode(ids) == text


def test_digits_are_split_individually(tok: Tokenizer) -> None:
    toks = [tok.id_to_token(i) for i in tok.encode("at 100")]
    assert toks[-3:] == ["1", "0", "0"]


def test_decode_skips_specials(tok: Tokenizer) -> None:
    ids = [tok.lang_id("am"), *tok.encode("ሰላም"), tok.eos_id, tok.pad_id]
    assert tok.decode(ids) == "ሰላም"


def test_encode_batch_matches_encode(tok: Tokenizer) -> None:
    assert tok.encode_batch(EN) == [tok.encode(t) for t in EN]


# ----------------------------------------------------------------------------- subset SQL


def test_subset_sql_where() -> None:
    assert SubsetConfig().sql_where() == "TRUE"
    assert SubsetConfig(sources=["mt560", "nllb"]).sql_where() == "source IN ('mt560', 'nllb')"
    assert (
        SubsetConfig(min_score={"nllb": 1.068}).sql_where()
        == "(source <> 'nllb' OR score >= 1.068)"
    )
    with pytest.raises(ValueError, match="must match"):
        SubsetConfig(sources=["bad name'"])


# ----------------------------------------------------------------------------- end to end


def test_train_end_to_end(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    rows = {
        "en": EN * 40,
        "am": AM * 40,
        "source": ["a", "nllb", "a", "nllb", "a"] * 40,
        "score": [None, 1.05, None, 1.2, None] * 40,
    }
    schema = pa.schema(
        [("en", pa.string()), ("am", pa.string()), ("source", pa.string()), ("score", pa.float32())]
    )
    pq.write_table(pa.table(rows, schema=schema), processed / "train.parquet")
    pq.write_table(pa.table({"en": EN, "am": AM}), processed / "train_holdout.parquet")

    cfg = Config.model_validate(
        {
            "project": {"name": "t", "seed": 3},
            "paths": {"root": str(tmp_path), "data_processed": "processed", "artifacts": "art"},
            "tokenizer": {
                "vocab_size": 600,
                "min_frequency": 1,
                "sample_per_lang": 50,
                "subset": {"min_score": {"nllb": 1.1}},  # drops the score-1.05 nllb rows
            },
        }
    )
    stats = train(cfg)
    out = tmp_path / "art" / "tokenizer"
    assert (out / "tokenizer.json").exists()
    saved = json.loads((out / "stats.json").read_text(encoding="utf-8"))
    assert saved["vocab_size"] == stats["vocab_size"] <= 600  # tiny fixture cannot fill 600
    assert stats["trained_on_sentences"] == 100  # 50 per language
    assert stats["roundtrip_exact"] == 1.0
    assert stats["en"]["unk_rate"] == 0.0 and stats["am"]["unk_rate"] == 0.0
    assert 0.5 < stats["am_en_length_ratio"] < 2.0
    assert stats["subset_where"] == "(source <> 'nllb' OR score >= 1.1)"

    tok = Tokenizer.from_file(out / "tokenizer.json")
    assert tok.decode(tok.encode(AM[0])) == AM[0]
