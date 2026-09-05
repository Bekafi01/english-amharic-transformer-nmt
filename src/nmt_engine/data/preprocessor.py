"""Linguistic preprocessing, script normalization, and multi-stage deduplication for English-Amharic bitext."""

import re
import unicodedata

import langid
import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH

from src.nmt_engine.utils.logging import get_logger

logger = get_logger(__name__)

# Restrict language identification to Amharic and English
langid.set_languages(["am", "en"])


class EthiopicNormalizer:
    """Comprehensive normalization engine for Ethiopic (Ge'ez) script.

    Provides canonical homophone reduction across all 7 vowel orders and labiovelars,
    exact multiplicative parsing of Ethiopic numerals to Arabic digits,
    homophone-invariant abbreviation expansion, and punctuation normalization.
    """

    HOMOPHONE_MAP: dict[str, str] = {
        # ሀ series (1st - 7th order + labiovelars)
        "ሐ": "ሀ",
        "ኀ": "ሀ",
        "ሑ": "ሁ",
        "ኁ": "ሁ",
        "ሒ": "ሂ",
        "ኂ": "ሂ",
        "ሓ": "ሃ",
        "ኃ": "ሃ",
        "ሔ": "ሄ",
        "ኄ": "ሄ",
        "ሕ": "ህ",
        "ኅ": "ህ",
        "ሖ": "ሆ",
        "ኆ": "ሆ",
        "ሗ": "ኋ",
        "ኇ": "ኋ",
        # ሰ series
        "ሠ": "ሰ",
        "ሡ": "ሱ",
        "ሢ": "ሲ",
        "ሣ": "ሳ",
        "ሤ": "ሴ",
        "ሥ": "ስ",
        "ሦ": "ሶ",
        "ሧ": "ሷ",
        # አ series
        "ዐ": "አ",
        "ዑ": "ኡ",
        "ዒ": "ኢ",
        "ዓ": "ኣ",
        "ዔ": "ኤ",
        "ዕ": "እ",
        "ዖ": "ኦ",
        # ጸ series
        "ፀ": "ጸ",
        "ፁ": "ጹ",
        "ፂ": "ጺ",
        "ፃ": "ጻ",
        "ፄ": "ጼ",
        "ፅ": "ጽ",
        "ፆ": "ጾ",
        "ፇ": "ጿ",
    }

    ABBREVIATION_MAP: dict[str, str] = {
        "ዓ/ም": "ዓመተ ምህረት",
        "ዐ/ም": "ዓመተ ምህረት",
        "ኣ/ም": "ዓመተ ምህረት",
        "ዶ/ር": "ዶክተር",
        "ወ/ሮ": "ወይዘሮ",
        "ወ/ሪት": "ወይዘሪት",
        "ፍ/ቤት": "ፍርድ ቤት",
        "ት/ቤት": "ትምህርት ቤት",
        "ጽ/ቤት": "ጽሕፈት ቤት",
        "ፅ/ቤት": "ጽሕፈት ቤት",
    }

    PROTECT_PATTERNS: list[str] = [
        r"ዓ\.ም\.?",
        r"እ\.ኤ\.አ\.?",
        r"ወ\.ዘ\.ተ\.?",
        r"ዶ\.ር\.?",
        r"\d+\.\d+",
        r"\d+:\d+",
    ]

    ETHIOPIC_PUNCT_MAP: dict[str, str] = {
        ".": "።",
        ";": "፤",
        ",": "፣",
        ":": "፦",
    }

    ETHIOPIC_DIGIT_MAP: dict[str, int] = {
        "፩": 1,
        "፪": 2,
        "፫": 3,
        "፬": 4,
        "፭": 5,
        "፮": 6,
        "፯": 7,
        "፰": 8,
        "፱": 9,
        "፲": 10,
        "፳": 20,
        "፴": 30,
        "፵": 40,
        "፶": 50,
        "፷": 60,
        "፸": 70,
        "፹": 80,
        "፺": 90,
        "፻": 100,
        "፼": 10000,
    }

    def __init__(self) -> None:
        sorted_abbrevs = sorted(self.ABBREVIATION_MAP.keys(), key=len, reverse=True)
        self._abbrev_pattern = re.compile("|".join(re.escape(k) for k in sorted_abbrevs))
        self._protect_re = re.compile("|".join(self.PROTECT_PATTERNS))
        self._geez_numeral_re = re.compile(r"[፩፪፫፬፭፮፯፰፱፲፳፴፵፶፷፸፹፺፻፼]+")

    def parse_ethiopic_number(self, numeral_str: str) -> int:
        """Parse a sequence of Ethiopic numeral characters into an integer using multiplicative rules."""
        total = 0
        current = 0
        for ch in numeral_str:
            val = self.ETHIOPIC_DIGIT_MAP.get(ch, 0)
            if ch == "፻":
                current = current if current > 0 else 1
                total += current * 100
                current = 0
            elif ch == "፼":
                current = current if current > 0 else 1
                total += current * 10000
                current = 0
            else:
                current += val
        total += current
        return total

    def ethiopic_to_arabic_numerals(self, text: str) -> str:
        """Convert all Ethiopic (Ge'ez) numeral clusters to Arabic digits."""
        return self._geez_numeral_re.sub(lambda m: str(self.parse_ethiopic_number(m.group())), text)

    def expand_abbreviations(self, text: str) -> str:
        """Expand common institutional and calendar abbreviations to full lexical forms."""
        return self._abbrev_pattern.sub(lambda m: self.ABBREVIATION_MAP[m.group(0)], text)

    def normalize_homophones(self, text: str) -> str:
        """Collapse classical phonological homophone variants to canonical graphemes."""
        return "".join(self.HOMOPHONE_MAP.get(c, c) for c in text)

    def normalize_punctuation(self, text: str) -> str:
        """Standardize Ethiopic punctuation with placeholder protection for decimals and times."""
        # Map Ethiopic word divider (፡) to standard whitespace
        text = text.replace("፡", " ")

        # Protect decimals, times, acronyms
        placeholders: dict[str, str] = {}

        def protect(m: re.Match) -> str:
            key = f"__PROT_{len(placeholders)}__"
            placeholders[key] = m.group(0)
            return key

        text = self._protect_re.sub(protect, text)

        # Standardize Latin punctuation in Amharic text to Ethiopic punctuation
        text = "".join(self.ETHIOPIC_PUNCT_MAP.get(c, c) for c in text)

        # Restore protected patterns
        for key, val in placeholders.items():
            text = text.replace(key, val)

        # Collapse repeated punctuation marks
        text = re.sub(r"።{2,}", "።", text)
        text = re.sub(r"፣{2,}", "፣", text)
        text = re.sub(r"፤{2,}", "፤", text)
        text = re.sub(r"፦{2,}", "፦", text)
        text = re.sub(r"\.{2,}", ".", text)
        text = re.sub(r",{2,}", ",", text)
        text = re.sub(r";{2,}", ";", text)
        text = re.sub(r":{2,}", ":", text)
        text = re.sub(r"!{2,}", "!", text)
        text = re.sub(r"\?{2,}", "?", text)

        # Strip whitespace preceding terminal/clausal punctuation
        text = re.sub(r"\s+([።፣፤፦\.\,\;\:\!\?])", r"\1", text)
        return text

    def normalize(self, text: str) -> str:
        """Apply complete pipeline: NFC -> Abbreviations -> Numerals -> Homophones -> Punctuation -> Whitespace."""
        if not isinstance(text, str) or not text.strip():
            return ""

        text = unicodedata.normalize("NFC", text)
        text = self.expand_abbreviations(text)
        text = self.ethiopic_to_arabic_numerals(text)
        text = self.normalize_homophones(text)
        text = self.normalize_punctuation(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


class TextCleaner:
    """Gross filtering, language identification, and standard English cleaning."""

    ETHIOPIC_REGEX = re.compile(r"[\u1200-\u137F\u1380-\u1399\u2D80-\u2DDF]")
    LATIN_REGEX = re.compile(r"[a-zA-Z]")

    @classmethod
    def clean_english(cls, text: str) -> str:
        """Normalize English text: Unicode NFC, repeated punctuation collapse, whitespace stripping."""
        if not isinstance(text, str) or not text.strip():
            return ""
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"([\.\,\;\:\!\?])\1+", r"\1", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\s+([\.\,\;\:\!\?])", r"\1", text)
        return text

    @classmethod
    def is_junk_pair(cls, am_text: str, en_text: str) -> bool:
        """Check whether either side contains URLs, emails, HTML tags, or purely numeric/punctuation noise."""
        for text in (am_text, en_text):
            if re.search(r"https?://\S+|www\.\S+", text):
                return True
            if re.search(r"\S+@\S+\.\S+", text):
                return True
            if re.search(r"<[^>]+>", text):
                return True
            if re.match(r"^[\d\s\.,\-\+\(\):;]+$", text):
                return True
        return False

    @classmethod
    def has_script_integrity(cls, am_text: str, en_text: str, min_ratio: float = 0.40) -> bool:
        """Verify that Amharic contains >= min_ratio Ethiopic characters and English contains >= min_ratio Latin characters."""
        am_chars = len(am_text.replace(" ", ""))
        en_chars = len(en_text.replace(" ", ""))
        if am_chars == 0 or en_chars == 0:
            return False

        am_ethiopic_ratio = len(cls.ETHIOPIC_REGEX.findall(am_text)) / am_chars
        en_latin_ratio = len(cls.LATIN_REGEX.findall(en_text)) / en_chars
        return (am_ethiopic_ratio >= min_ratio) and (en_latin_ratio >= min_ratio)

    @classmethod
    def verify_langid(cls, text: str, expected_lang: str) -> bool:
        """Verify text matches expected language code via langid."""
        try:
            pred, _ = langid.classify(text)
            return pred == expected_lang
        except Exception:
            return False


class DataPreprocessor:
    """Production data preprocessing pipeline for English-Amharic parallel corpora.

    Coordinates gross filtering, linguistic normalization, sequence length/ratio filtering,
    exact deduplication, MinHash LSH fuzzy deduplication, and zero-leakage dataset partitioning.
    """

    def __init__(
        self,
        min_seq_len: int = 2,
        max_seq_len: int = 128,
        min_len_ratio: float = 0.30,
        max_len_ratio: float = 2.50,
        minhash_threshold: float = 0.90,
        minhash_num_perm: int = 128,
        seed: int = 42,
    ) -> None:
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.min_len_ratio = min_len_ratio
        self.max_len_ratio = max_len_ratio
        self.minhash_threshold = minhash_threshold
        self.minhash_num_perm = minhash_num_perm
        self.seed = seed
        self.ethiopic_normalizer = EthiopicNormalizer()
        self.text_cleaner = TextCleaner()

    def filter_gross_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute non-empty stripping, junk detection, script compliance, and language identification."""
        df = df.copy()
        df["amharic"] = df["amharic"].astype(str).str.strip()
        df["english"] = df["english"].astype(str).str.strip()

        # 1. Non-empty
        df = df[(df["amharic"].str.len() > 0) & (df["english"].str.len() > 0)].copy()

        # 2. Junk pair filtering
        junk_mask = df.apply(
            lambda r: self.text_cleaner.is_junk_pair(r["amharic"], r["english"]),
            axis=1,
        )
        df = df[~junk_mask].copy()

        # 3. Script integrity
        script_mask = df.apply(
            lambda r: self.text_cleaner.has_script_integrity(r["amharic"], r["english"]),
            axis=1,
        )
        df = df[script_mask].copy()

        # 4. Language ID verification
        lang_am_ok = df["amharic"].apply(lambda t: self.text_cleaner.verify_langid(t, "am"))
        lang_en_ok = df["english"].apply(lambda t: self.text_cleaner.verify_langid(t, "en"))
        df = df[lang_am_ok & lang_en_ok].reset_index(drop=True)

        return df

    def normalize_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply linguistic normalization to both language columns."""
        df = df.copy()
        df["amharic_norm"] = df["amharic"].apply(self.ethiopic_normalizer.normalize)
        df["english_norm"] = df["english"].apply(self.text_cleaner.clean_english)
        return df

    def filter_length_and_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enforce min/max sequence word length and cross-lingual length ratio constraints."""
        df = df.copy()
        df["am_words"] = df["amharic_norm"].apply(lambda x: len(x.split()))
        df["en_words"] = df["english_norm"].apply(lambda x: len(x.split()))

        # Enforce word bounds
        valid_len = (
            (df["am_words"] >= self.min_seq_len)
            & (df["am_words"] <= self.max_seq_len)
            & (df["en_words"] >= self.min_seq_len)
            & (df["en_words"] <= self.max_seq_len)
        )
        df = df[valid_len].copy()

        # Compute ratio
        df["len_ratio"] = df["am_words"] / df["en_words"]
        valid_ratio = (df["len_ratio"] >= self.min_len_ratio) & (
            df["len_ratio"] <= self.max_len_ratio
        )
        return df[valid_ratio].reset_index(drop=True)

    def deduplicate_exact(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop exact duplicate bitext pairs based on normalized text."""
        return df.drop_duplicates(subset=["amharic_norm", "english_norm"]).reset_index(drop=True)

    def deduplicate_minhash_lsh(self, df: pd.DataFrame, char_k: int = 3) -> pd.DataFrame:
        """Cluster fuzzy near-duplicate bitext using MinHash LSH and Union-Find connected components."""
        if len(df) <= 1:
            return df

        def get_minhash(text: str) -> MinHash:
            m = MinHash(num_perm=self.minhash_num_perm)
            norm = text.lower().strip()
            for i in range(max(1, len(norm) - char_k + 1)):
                m.update(norm[i : i + char_k].encode("utf-8"))
            return m

        lsh = MinHashLSH(threshold=self.minhash_threshold, num_perm=self.minhash_num_perm)
        minhashes = {}

        for idx, row in df.iterrows():
            combined = row["amharic_norm"] + " " + row["english_norm"]
            m = get_minhash(combined)
            minhashes[idx] = m
            lsh.insert(str(idx), m)

        # Union-Find with path compression
        parent = {idx: idx for idx in df.index}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                if rx < ry:
                    parent[ry] = rx
                else:
                    parent[rx] = ry

        for idx in df.index:
            for match_id in lsh.query(minhashes[idx]):
                match_idx = int(match_id)
                if match_idx != idx:
                    union(idx, match_idx)

        # Group indices into connected component clusters
        clusters: dict[int, list[int]] = {}
        for idx in df.index:
            clusters.setdefault(find(idx), []).append(idx)

        # Drop all but the single lowest index in each cluster
        to_drop: set[int] = set()
        for _, members in clusters.items():
            if len(members) > 1:
                keep_idx = min(members)
                for m in members:
                    if m != keep_idx:
                        to_drop.add(m)

        df_clean = df.drop(index=list(to_drop)).reset_index(drop=True)
        return df_clean

    def partition_dataset(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.90,
        val_ratio: float = 0.05,
        test_ratio: float = 0.05,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataset into Train/Val/Test partitions with deterministic seeding and assert zero leakage."""
        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5

        n = len(df)
        rng = np.random.default_rng(self.seed)
        shuffled_indices = rng.permutation(n)

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_indices = shuffled_indices[:n_train]
        val_indices = shuffled_indices[n_train : n_train + n_val]
        test_indices = shuffled_indices[n_train + n_val :]

        df_train = df.iloc[train_indices].reset_index(drop=True)
        df_val = df.iloc[val_indices].reset_index(drop=True)
        df_test = df.iloc[test_indices].reset_index(drop=True)

        # Zero-leakage verification
        train_keys = set(df_train["amharic"] + "|||" + df_train["english"])
        val_keys = set(df_val["amharic"] + "|||" + df_val["english"])
        test_keys = set(df_test["amharic"] + "|||" + df_test["english"])

        overlap_tv = len(train_keys & val_keys)
        overlap_tt = len(train_keys & test_keys)
        overlap_vt = len(val_keys & test_keys)

        if overlap_tv != 0 or overlap_tt != 0 or overlap_vt != 0:
            raise ValueError(
                f"Data leakage detected! Overlaps: Train-Val={overlap_tv}, "
                f"Train-Test={overlap_tt}, Val-Test={overlap_vt}"
            )

        return df_train, df_val, df_test

    def process(
        self,
        df_raw: pd.DataFrame,
        run_minhash: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Execute the full end-to-end preprocessing, cleaning, deduplication, and partitioning workflow."""
        logger.info("Step 1: Filtering gross noise and validating language integrity...")
        df_filtered = self.filter_gross_noise(df_raw)
        logger.info(f"Retained {len(df_filtered):,} pairs after gross filtering.")

        logger.info("Step 2: Normalizing Amharic and English text...")
        df_norm = self.normalize_text(df_filtered)

        logger.info("Step 3: Applying sequence length and ratio filtering...")
        df_len = self.filter_length_and_ratio(df_norm)
        logger.info(f"Retained {len(df_len):,} pairs after length & ratio filtering.")

        logger.info("Step 4: Running exact deduplication...")
        df_exact = self.deduplicate_exact(df_len)
        logger.info(f"Retained {len(df_exact):,} pairs after exact deduplication.")

        if run_minhash:
            logger.info("Step 5: Running MinHash LSH fuzzy near-duplicate clustering...")
            df_dedup = self.deduplicate_minhash_lsh(df_exact)
            logger.info(f"Retained {len(df_dedup):,} pairs after MinHash LSH deduplication.")
        else:
            df_dedup = df_exact

        # Standardize column schema
        cols = ["english_norm", "amharic_norm"]
        if "source" in df_dedup.columns:
            cols.append("source")
        if "am_words" in df_dedup.columns:
            cols.extend(["am_words", "en_words", "len_ratio"])

        df_clean = df_dedup[cols].rename(
            columns={"english_norm": "english", "amharic_norm": "amharic"}
        )

        logger.info("Step 6: Partitioning dataset into Train (90%), Val (5%), Test (5%)...")
        df_train, df_val, df_test = self.partition_dataset(df_clean)
        logger.info(
            f"Dataset partitioned: Train={len(df_train):,}, Val={len(df_val):,}, Test={len(df_test):,}."
        )

        return df_train, df_val, df_test
