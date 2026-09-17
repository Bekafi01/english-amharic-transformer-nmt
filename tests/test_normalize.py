import pytest

from amnmt.data.normalize import (
    dedup_key,
    ethiopic_number_to_int,
    fold_homophones,
    normalize_amharic,
    normalize_english,
    numerals_to_digits,
)

# ----------------------------------------------------------------------------- homophones


@pytest.mark.parametrize(
    ("raw", "folded"),
    [
        ("ሐ", "ሀ"),
        ("ኀ", "ሀ"),
        ("ሠላም", "ሰላም"),
        ("ዓመት", "ኣመት"),
        ("ፀሐይ", "ጸሀይ"),
        ("ሕይወት", "ህይወት"),
        ("ሰላም", "ሰላም"),  # canonical forms untouched
    ],
)
def test_fold_homophones(raw: str, folded: str) -> None:
    assert fold_homophones(raw) == folded


def test_fold_can_be_disabled() -> None:
    assert normalize_amharic("ሐሰት", fold=False) == "ሐሰት"
    assert normalize_amharic("ሐሰት", fold=True) == "ሀሰት"


# ----------------------------------------------------------------------------- numerals


@pytest.mark.parametrize(
    ("numeral", "value"),
    [
        ("፩", 1),
        ("፲", 10),
        ("፲፪", 12),
        ("፻", 100),
        ("፪፻", 200),
        ("፪፻፲", 210),
        ("፲፱፻፺፪", 1992),
        ("፼", 10_000),
        ("፪፼", 20_000),
        ("፲፼", 100_000),
        ("፻፼", 1_000_000),
        ("፩፼፭፻", 10_500),
    ],
)
def test_ethiopic_number_to_int(numeral: str, value: int) -> None:
    assert ethiopic_number_to_int(numeral) == value


def test_numerals_to_digits_in_context() -> None:
    assert numerals_to_digits("በ፲፱፻፺፪ ዓ.ም") == "በ1992 ዓ.ም"
    assert normalize_amharic("በ፲፱፻፺፪ ዓ.ም", digits=False) == "በ፲፱፻፺፪ ኣ.ም"


# ----------------------------------------------------------------------------- amharic


def test_wordspace_becomes_space_and_double_wordspace_becomes_full_stop() -> None:
    assert normalize_amharic("ሰላም፡ለዓለም፡ ፡") == "ሰላም ለኣለም።"
    assert normalize_amharic("ሰላም ፡ ፡ እንዴት") == "ሰላም። እንዴት"


def test_amharic_detokenize() -> None:
    assert normalize_amharic("ይናገር ነበር ፤ ደግሞም ።") == "ይናገር ነበር፤ ደግሞም።"
    assert normalize_amharic("( ሰሎሞን ) ስለ") == "(ሰሎሞን) ስለ"
    assert normalize_amharic("ሰላም ።።።") == "ሰላም።"


def test_amharic_whitespace_and_invisibles() -> None:
    assert normalize_amharic("  ሰላም\u200b  \tለዓለም \n") == "ሰላም ለኣለም"


def test_amharic_is_idempotent() -> None:
    s = "ሐሰት ፡ ፡ በ፲፱፻፺፪ ( ዓ.ም ) ።"
    once = normalize_amharic(s)
    assert normalize_amharic(once) == once


# ----------------------------------------------------------------------------- english


def test_english_detokenize_clitics_and_punct() -> None:
    assert normalize_english("Jehovah 's creation , he said .") == "Jehovah's creation, he said."
    assert normalize_english("don 't do it !") == "don't do it!"
    assert normalize_english("I 'm here ; you 're not") == "I'm here; you're not"
    assert normalize_english("[ Solomon ] spoke") == "[Solomon] spoke"


def test_detokenize_quotes_and_numeric_colons() -> None:
    assert normalize_english('She said : " I was well . "') == 'She said: "I was well."'
    assert normalize_english("- Psalm 55 : 22 .") == "- Psalm 55:22."
    assert normalize_english("at 10 : 30 and 11 : 45") == "at 10:30 and 11:45"
    assert normalize_amharic('" ሰላም ለዓለም " አለ ።') == '"ሰላም ለኣለም" አለ።'
    # A lone quote is left alone; prose colons keep their space
    assert normalize_english('He said " hi') == 'He said " hi'
    assert normalize_english("Note: see below") == "Note: see below"


def test_english_moses_escapes_and_entities() -> None:
    assert normalize_english("wrong @-@ doers &amp; others") == "wrong-doers & others"
    assert normalize_english("&quot;hi&quot;") == '"hi"'


def test_english_curly_quotes_straightened() -> None:
    assert normalize_english("“Hello” it’s") == '"Hello" it\'s'


def test_english_is_idempotent() -> None:
    s = "Jehovah 's creation , &quot; he said @-@ so . &quot;"
    once = normalize_english(s)
    assert normalize_english(once) == once


# ----------------------------------------------------------------------------- dedup key


def test_dedup_key_ignores_case_punct_and_space() -> None:
    assert dedup_key("Hello, World!") == dedup_key("hello world")
    assert dedup_key("ሰላም ለዓለም።") == dedup_key("ሰላምለዓለም")
    assert dedup_key("Hello") != dedup_key("Hello there")
