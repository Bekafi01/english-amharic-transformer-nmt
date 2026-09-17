import pytest

from amnmt.core.config import FilterConfig
from amnmt.data.filters import reject_reason

CFG = FilterConfig()
EN = "The quick brown fox jumps over the lazy dog."
AM = "ፈጣኑ ቡናማ ቀበሮ ሰነፉን ውሻ ዘሎ አለፈ።"


def test_good_pair_is_kept() -> None:
    assert reject_reason(EN, AM, CFG) is None


@pytest.mark.parametrize(
    ("en", "am", "reason"),
    [
        ("", AM, "empty"),
        (EN, "", "empty"),
        ("Level one", "Level one", "identical"),
        ("D", "ዲ", "too_short"),
        ("a" * 1001, AM, "too_long"),
        (" ".join(["word"] * 151), AM, "too_many_words"),
        (EN * 6, "ውሻ ዘሎ", "ratio"),
        ("ውሻ", EN * 6, "ratio"),
        ("Removing packages", "Removing packages now", "am_script"),
        (EN, "%d %B %Y %H:%M %I %p %A %e", "am_script"),
        ("ሰላም ለዓለም እንዴት ናችሁ", AM, "en_script"),
        ("12345 67890", AM, "en_script"),
        ("See https://example.com now", AM, "url"),
        (EN, "ዝርዝር መረጃ ለማግኘት ይህን ድረ ገጽ ይመልከቱ www.ex.am", "url"),
        ("Click <b>here</b> please", AM, "markup"),
        ("Hello %s, welcome", AM, "markup"),
        ("Value is {0} today", AM, "markup"),
        ("N-Z_BAR_Spanish", AM, "markup"),
    ],
)
def test_reject_reasons(en: str, am: str, reason: str) -> None:
    assert reject_reason(en, am, CFG) == reason


def test_amharic_with_some_latin_is_fine() -> None:
    assert reject_reason("GNOME is a desktop.", "ኖም (GNOME) የዴስክቶፕ ነው።", CFG) is None


def test_flags_can_disable_rules() -> None:
    cfg = FilterConfig(drop_urls=False, drop_markup=False)
    assert reject_reason("See https://example.com now", AM, cfg) is None
    assert reject_reason("Click <b>here</b> please", AM, cfg) is None
