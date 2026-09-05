"""Unit tests for Ethiopic normalizer, text cleaner, and data preprocessing pipeline."""

from __future__ import annotations

import pandas as pd

from src.nmt_engine.data.preprocessor import (
    DataPreprocessor,
    EthiopicNormalizer,
    TextCleaner,
)


def test_ethiopic_normalizer_homophones():
    """Verifies homophone reduction across 7 vowel orders and labiovelars."""
    normalizer = EthiopicNormalizer()

    # ሀ series
    assert normalizer.normalize_homophones("ሐመር ኀይል") == "ሀመር ሀይል"
    assert normalizer.normalize_homophones("ሑዳዴ ኁልቆ") == "ሁዳዴ ሁልቆ"
    assert normalizer.normalize_homophones("ሒሳብ ኂሩት") == "ሂሳብ ሂሩት"
    assert normalizer.normalize_homophones("ሓምሌ ኃይል") == "ሃምሌ ሃይል"
    assert normalizer.normalize_homophones("ሔኖክ ኄር") == "ሄኖክ ሄር"
    assert normalizer.normalize_homophones("ሕግ ኅብረት") == "ህግ ህብረት"
    assert normalizer.normalize_homophones("ሖረ ኆኅት") == "ሆረ ሆህት"
    assert normalizer.normalize_homophones("ሗላ ኇላ") == "ኋላ ኋላ"

    # ሰ series
    assert normalizer.normalize_homophones("ሠላም ሡልጣን ሢመት") == "ሰላም ሱልጣን ሲመት"
    assert normalizer.normalize_homophones("ሥራ ሦስት ሧይ") == "ስራ ሶስት ሷይ"

    # አ series
    assert normalizer.normalize_homophones("ዐይኑ ዑደት ዒላማ") == "አይኑ ኡደት ኢላማ"
    assert normalizer.normalize_homophones("ዓመት ዔሊ ዕውቀት ዖፍ") == "ኣመት ኤሊ እውቀት ኦፍ"

    # ጸ series
    assert normalizer.normalize_homophones("ፀሐይ ፁም ፂም") == "ጸሀይ ጹም ጺም"
    assert normalizer.normalize_homophones("ፃድቅ ፄና ፅሕፈት ፆታ ፇይ") == "ጻድቅ ጼና ጽህፈት ጾታ ጿይ"


def test_ethiopic_numeral_parser():
    """Verifies exact multiplicative parsing of Ge'ez numerals to Arabic digits."""
    normalizer = EthiopicNormalizer()

    test_cases = [
        ("፱፻", "900"),
        ("፰፻፺፱", "899"),
        ("፩፻", "100"),
        ("፩፼", "10000"),
        ("፪፼፭፻፷፯", "20567"),
        ("፲፭", "15"),
        ("፳፩", "21"),
        ("፺፱", "99"),
        ("፻", "100"),
        ("፼", "10000"),
        ("፩፻፩", "101"),
        ("፱፻፺፱", "999"),
    ]

    for geez_num, expected in test_cases:
        assert normalizer.ethiopic_to_arabic_numerals(geez_num) == expected

    # In-sentence replacement
    sentence = "በዓመቱ ፲፭ ሺህ ተማሪዎች ተመረቁ።"
    assert "15 ሺህ" in normalizer.ethiopic_to_arabic_numerals(sentence)


def test_abbreviation_expansion():
    """Verifies expansion of common institutional and calendar abbreviations."""
    normalizer = EthiopicNormalizer()

    assert normalizer.expand_abbreviations("ዶ/ር አብይ") == "ዶክተር አብይ"
    assert normalizer.expand_abbreviations("ት/ቤት ገቡ") == "ትምህርት ቤት ገቡ"
    assert normalizer.expand_abbreviations("ፍ/ቤት ሄዱ") == "ፍርድ ቤት ሄዱ"
    assert normalizer.expand_abbreviations("ጽ/ቤት ደረሱ") == "ጽሕፈት ቤት ደረሱ"
    assert normalizer.expand_abbreviations("ፅ/ቤት ደረሱ") == "ጽሕፈት ቤት ደረሱ"
    assert normalizer.expand_abbreviations("በዓ/ም 2016") == "በዓመተ ምህረት 2016"


def test_punctuation_normalization_and_protection():
    """Verifies punctuation mapping while protecting numbers, times, and acronyms."""
    normalizer = EthiopicNormalizer()

    text = "ዋጋው 3.14 ዶላር ሲሆን ሰዓቱ 10:30 ነበር፡ ሰላም ለሁሉ...."
    normalized = normalizer.normalize_punctuation(text)

    # Decimals and times protected
    assert "3.14" in normalized
    assert "10:30" in normalized
    # Word divider replaced
    assert "፡" not in normalized
    # Repeated periods collapsed
    assert "...." not in normalized


def test_text_cleaner_junk_detection():
    """Verifies junk pair filtering for URLs, emails, HTML, and numeric noise."""
    cleaner = TextCleaner()

    assert cleaner.is_junk_pair("Visit https://example.com", "ድረ ገጹን ይጎብኙ")
    assert cleaner.is_junk_pair("Contact info@example.com", "ኢሜይል ያድርጉ")
    assert cleaner.is_junk_pair("<p>Paragraph</p>", "<p>አንቀጽ</p>")
    assert cleaner.is_junk_pair("12345 67890", "12345 67890")
    assert not cleaner.is_junk_pair("Peace and prosperity.", "ሰላምና ብልጽግና።")


def test_script_integrity():
    """Verifies that script character ratios validate correct language scripts."""
    cleaner = TextCleaner()

    # Valid parallel
    assert cleaner.has_script_integrity("ሰላም ለዓለም", "Peace to the world.")
    # Mismatched: Amharic side contains purely English
    assert not cleaner.has_script_integrity("Wrong script in Amharic", "Valid English.")
    # Mismatched: English side contains purely Ethiopic
    assert not cleaner.has_script_integrity("ትክክለኛ አማርኛ", "የተሳሳተ እንግሊዝኛ")


def test_data_preprocessor_pipeline():
    """Verifies end-to-end preprocessing, deduplication, and zero-leakage split."""
    preprocessor = DataPreprocessor(
        min_seq_len=2,
        max_seq_len=128,
        min_len_ratio=0.30,
        max_len_ratio=2.50,
        seed=42,
    )

    data = [
        {"amharic": "ሰላም ለዓለም ይሁን።", "english": "Peace be to the world."},
        {"amharic": "ዶ/ር አብይ ወደ ጽ/ቤት መጡ።", "english": "Dr. Abiy came to the office."},
        {
            "amharic": "ዶ/ር አብይ ወደ ፅ/ቤት መጡ።",
            "english": "Dr. Abiy came to the office.",
        },  # Homophone dupe
        {
            "amharic": "በዓመቱ ፲፭ ሺህ ተማሪዎች ተመረቁ።",
            "english": "In the year 15 thousand students graduated.",
        },
        {"amharic": "https://spam.com/link", "english": "Spam link"},  # Junk
        {"amharic": "አንድ", "english": "One"},  # 1 word -> length filter drops
    ] * 20  # Replicate to have sufficient rows for splitting

    df_raw = pd.DataFrame(data)
    df_train, df_val, df_test = preprocessor.process(df_raw, run_minhash=False)

    # Check that output columns exist
    assert "amharic" in df_train.columns
    assert "english" in df_train.columns

    # Verify zero leakage across splits
    train_keys = set(df_train["amharic"] + "|||" + df_train["english"])
    val_keys = set(df_val["amharic"] + "|||" + df_val["english"])
    test_keys = set(df_test["amharic"] + "|||" + df_test["english"])

    assert len(train_keys & val_keys) == 0
    assert len(train_keys & test_keys) == 0
    assert len(val_keys & test_keys) == 0
