"""Pair-level quality filters. Pure functions: return a reject reason or None (keep).

Rules are deliberately cheap (regex / character classes) so they run over ~17M pairs on a
Colab CPU in minutes. See PLAN.md D8 for why there is no language-ID model.
"""

from __future__ import annotations

import re

from amnmt.core.config import FilterConfig

_ETHIOPIC_RE = re.compile(r"[\u1200-\u137f\u1380-\u139f\u2d80-\u2ddf\uab00-\uab2f]")
_LATIN_RE = re.compile(r"[A-Za-z\u00c0-\u024f]")
_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)
_URL_RE = re.compile(r"(https?://|www\.)\S+", re.IGNORECASE)
_MARKUP_RE = re.compile(
    r"</?[a-z][^<>]*>"  # html/xml tags
    r"|&[a-z]+;|&#\d+;"  # leftover entities
    r"|%\(?[a-z_]*\)?[sdfBAYeIMpHS]"  # printf placeholders (%s, %d, %B, %(name)s)
    r"|\{\w*\}"  # {0}, {name}
    r"|_BAR_",  # GNOME mnemonic separator seen in OPUS-100
    re.IGNORECASE,
)


def _share(pattern: re.Pattern[str], text: str) -> float:
    letters = len(_LETTER_RE.findall(text))
    return len(pattern.findall(text)) / letters if letters else 0.0


def reject_reason(en: str, am: str, cfg: FilterConfig) -> str | None:
    """Return why the *normalized* pair should be dropped, or None to keep it."""
    if not en or not am:
        return "empty"
    if cfg.drop_identical and en.casefold() == am.casefold():
        return "identical"
    if len(en) < cfg.min_chars or len(am) < cfg.min_chars:
        return "too_short"
    if len(en) > cfg.max_chars or len(am) > cfg.max_chars:
        return "too_long"
    if len(en.split()) > cfg.max_words or len(am.split()) > cfg.max_words:
        return "too_many_words"
    ratio = len(en) / len(am)
    if ratio > cfg.max_char_ratio or ratio < 1.0 / cfg.max_char_ratio:
        return "ratio"
    if _share(_ETHIOPIC_RE, am) < cfg.min_am_ethiopic_share:
        return "am_script"
    if _share(_LATIN_RE, en) < cfg.min_en_latin_share or _ETHIOPIC_RE.search(en):
        return "en_script"
    if cfg.drop_urls and (_URL_RE.search(en) or _URL_RE.search(am)):
        return "url"
    if cfg.drop_markup and (_MARKUP_RE.search(en) or _MARKUP_RE.search(am)):
        return "markup"
    return None
