"""Text normalization for English and Amharic. Pure functions, no I/O.

Applied identically at training and inference time, so keep every rule deterministic and
idempotent: `normalize_x(normalize_x(s)) == normalize_x(s)`.
"""

from __future__ import annotations

import html
import re
import unicodedata

# ---------------------------------------------------------------------------- tables

# Ethiopic homophones: several letter series are pronounced identically in Amharic.
# Folding each to one canonical series removes spelling variance the model cannot learn.
_HOMOPHONES: dict[str, str] = {
    # ሐ and ኀ series -> ሀ series (all 7 orders + labiovelar)
    "ሐ": "ሀ", "ኀ": "ሀ",
    "ሑ": "ሁ", "ኁ": "ሁ",
    "ሒ": "ሂ", "ኂ": "ሂ",
    "ሓ": "ሃ", "ኃ": "ሃ",
    "ሔ": "ሄ", "ኄ": "ሄ",
    "ሕ": "ህ", "ኅ": "ህ",
    "ሖ": "ሆ", "ኆ": "ሆ",
    "ሗ": "ኋ", "ኇ": "ኋ",
    # ሠ series -> ሰ series
    "ሠ": "ሰ", "ሡ": "ሱ", "ሢ": "ሲ", "ሣ": "ሳ", "ሤ": "ሴ", "ሥ": "ስ", "ሦ": "ሶ", "ሧ": "ሷ",
    # ዐ series -> አ series
    "ዐ": "አ", "ዑ": "ኡ", "ዒ": "ኢ", "ዓ": "ኣ", "ዔ": "ኤ", "ዕ": "እ", "ዖ": "ኦ",
    # ፀ series -> ጸ series
    "ፀ": "ጸ", "ፁ": "ጹ", "ፂ": "ጺ", "ፃ": "ጻ", "ፄ": "ጼ", "ፅ": "ጽ", "ፆ": "ጾ", "ፇ": "ጿ",
}  # fmt: skip
_HOMOPHONE_TABLE = str.maketrans(_HOMOPHONES)

_ETHIOPIC_DIGITS: dict[str, int] = {
    "፩": 1, "፪": 2, "፫": 3, "፬": 4, "፭": 5, "፮": 6, "፯": 7, "፰": 8, "፱": 9,
    "፲": 10, "፳": 20, "፴": 30, "፵": 40, "፶": 50, "፷": 60, "፸": 70, "፹": 80, "፺": 90,
    "፻": 100, "፼": 10_000,
}  # fmt: skip
_ETHIOPIC_NUMBER_RE = re.compile("[" + "".join(_ETHIOPIC_DIGITS) + "]+")

_QUOTES = str.maketrans({"“": '"', "”": '"', "„": '"', "‘": "'", "’": "'", "‚": "'", "´": "'"})

_ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\u2060\ufeff\u00ad]")
_CONTROL_RE = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS_RE = re.compile(r"\s+")

# Two wordspaces (often written "፡ ፡") is a common substitute for the full stop ።.
_DOUBLE_WORDSPACE_RE = re.compile(r"፡\s*፡")
_WORDSPACE = "፡"

# Detokenization: undo Moses-style tokenization found in MT560/Tanzil ("word ." "don 't").
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([።፣፤፥፦፧፨,.;:!?%)\]}])")
_SPACE_AFTER_OPEN_RE = re.compile(r"([(\[{])\s+")
_PADDED_QUOTES_RE = re.compile(r'"\s+([^"]*?)\s+"')  # " quoted text " -> "quoted text"
_NUMERIC_COLON_RE = re.compile(r"(\d):\s+(\d)")  # 10: 24 -> 10:24 (after space-before removal)
_EN_CLITIC_RE = re.compile(r"(\w) ?'\s?(s|t|re|ve|ll|d|m)\b", re.IGNORECASE)
_REPEATED_PUNCT_RE = re.compile(r"([።፣፤፥፦,;:!?])\1+")
_MOSES_ESCAPE_RE = re.compile(r"\s*@([-,.])@\s*")  # "wrong @-@ doers" -> "wrong-doers"

# ---------------------------------------------------------------------------- helpers


def _basic(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = html.unescape(text)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _CONTROL_RE.sub(" ", text)
    text = text.translate(_QUOTES)
    return _MOSES_ESCAPE_RE.sub(r"\1", text)


def _detokenize(text: str) -> str:
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = _SPACE_AFTER_OPEN_RE.sub(r"\1", text)
    text = _PADDED_QUOTES_RE.sub(r'"\1"', text)
    text = _NUMERIC_COLON_RE.sub(r"\1:\2", text)
    text = _REPEATED_PUNCT_RE.sub(r"\1", text)
    return text


def _squash_ws(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def ethiopic_number_to_int(numeral: str) -> int:
    """Parse a Ge'ez numeral (e.g. ፲፱፻፺፪ -> 1992). Digits ፩-፺ add up; ፻ multiplies by 100;
    ፼ multiplies everything accumulated so far by 10,000."""
    total = block = current = 0
    for ch in numeral:
        v = _ETHIOPIC_DIGITS[ch]
        if v < 100:
            current += v
        elif v == 100:
            block += (current or 1) * 100
            current = 0
        else:
            block += current
            total += (block or 1) * 10_000
            block = current = 0
    return total + block + current


# ---------------------------------------------------------------------------- public


def fold_homophones(text: str) -> str:
    return text.translate(_HOMOPHONE_TABLE)


def numerals_to_digits(text: str) -> str:
    return _ETHIOPIC_NUMBER_RE.sub(lambda m: str(ethiopic_number_to_int(m.group())), text)


def normalize_amharic(text: str, *, fold: bool = True, digits: bool = True) -> str:
    text = _basic(text)
    text = _DOUBLE_WORDSPACE_RE.sub("።", text)
    text = text.replace(_WORDSPACE, " ")
    if fold:
        text = fold_homophones(text)
    if digits:
        text = numerals_to_digits(text)
    text = _detokenize(text)
    return _squash_ws(text)


def normalize_english(text: str) -> str:
    text = _basic(text)
    text = _EN_CLITIC_RE.sub(r"\1'\2", text)
    text = _detokenize(text)
    return _squash_ws(text)


_KEY_STRIP_RE = re.compile(r"[\W_]+", re.UNICODE)


def dedup_key(text: str) -> str:
    """Aggressive key for exact-duplicate detection: casefolded letters and digits only."""
    return _KEY_STRIP_RE.sub("", text).casefold()
