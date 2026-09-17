import json
import zipfile
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from amnmt.core.config import Config
from amnmt.data.normalize import dedup_key
from amnmt.data.pipeline import build

GOOD = [
    ("The quick brown fox jumps over the lazy dog.", "ፈጣኑ ቡናማ ቀበሮ ሰነፉን ውሻ ዘሎ አለፈ።"),
    ("Good morning, how are you today?", "እንደምን አደርክ፣ ዛሬ እንዴት ነህ?"),
    ("The children are playing in the garden.", "ልጆቹ በአትክልት ስፍራ ውስጥ እየተጫወቱ ነው።"),
    ("Water boils at one hundred degrees.", "ውሃ በመቶ ዲግሪ ይፈላል።"),
    ("She reads a book every evening.", "እሷ በየምሽቱ መጽሐፍ ታነባለች።"),
    ("Addis Ababa is the capital of Ethiopia.", "አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት።"),
    ("My brother works at the hospital.", "ወንድሜ በሆስፒታል ውስጥ ይሰራል።"),
    ("The market opens early on Saturday.", "ገበያው ቅዳሜ ማለዳ ይከፈታል።"),
    ("Please close the door quietly.", "እባክህ በሩን በጸጥታ ዝጋ።"),
    ("The teacher explained the lesson well.", "መምህሩ ትምህርቱን በጥሩ ሁኔታ አብራርተዋል።"),
]
FLORES_DEV = [("Coffee originated in Ethiopia.", "ቡና መነሻው ኢትዮጵያ ነው።")]
FLORES_DEVTEST = [("The river flows to the north.", "ወንዙ ወደ ሰሜን ይፈሳል።")]


def _write_pairs(dir_: Path, stem: str, pairs: list[tuple[str, str]]) -> tuple[Path, Path]:
    en, am = dir_ / f"{stem}.en", dir_ / f"{stem}.am"
    en.write_text("\n".join(p[0] for p in pairs) + "\n", encoding="utf-8")
    am.write_text("\n".join(p[1] for p in pairs) + "\n", encoding="utf-8")
    return en, am


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[Config, Path]:
    # Fake FLORES layout: <root>/{dev,devtest}/{code}.{split}
    for split, pairs in (("dev", FLORES_DEV), ("devtest", FLORES_DEVTEST)):
        d = tmp_path / "flores" / split
        d.mkdir(parents=True)
        (d / f"eng_Latn.{split}").write_text(pairs[0][0] + "\n", encoding="utf-8")
        (d / f"amh_Ethi.{split}").write_text(pairs[0][1] + "\n", encoding="utf-8")

    # Source A: good pairs, Moses-tokenized, plus one exact dup and one FLORES leak.
    src_a = [(e.replace(".", " ."), a.replace("።", " ።")) for e, a in GOOD] + [
        GOOD[0],
        FLORES_DEV[0],
        ("Removing packages", "Removing packages"),  # identical -> reject
        ("See https://x.com now please", "አሁን ይህን ድረ ገጽ ይመልከቱ እባክህ https://x.com"),  # url -> reject
    ]
    # Source B: overlaps A on 3 pairs (should lose to A by priority), 2 new, 1 FLORES leak.
    src_b = [
        *GOOD[:3],
        ("The sun rises in the east.", "ፀሐይ በምሥራቅ ትወጣለች።"),
        ("Bread and butter for breakfast.", "ለቁርስ ዳቦ እና ቅቤ።"),
        (FLORES_DEVTEST[0][0].upper(), "ሌላ ነገር ነው ይህ ጽሁፍ።"),  # en matches devtest (case-insens.)
    ]
    a_en, a_am = _write_pairs(tmp_path, "a", src_a)
    b_en, b_am = _write_pairs(tmp_path, "b", src_b)

    cfg = Config.model_validate(
        {
            "project": {"name": "test", "seed": 7},
            "paths": {"root": str(tmp_path), "data_raw": "raw", "data_processed": "out"},
            "data": {
                "sources": [
                    {"name": "a", "kind": "local_pair", "en_file": str(a_en), "am_file": str(a_am)},
                    {"name": "b", "kind": "local_pair", "en_file": str(b_en), "am_file": str(b_am)},
                ],
                "flores": {"local_dir": str(tmp_path / "flores")},
                "holdout_size": 2,
                "shard_flush_rows": 3,  # force several flushes
            },
        }
    )
    return cfg, tmp_path / "out"


def test_build_end_to_end(workspace: tuple[Config, Path]) -> None:
    cfg, out = workspace
    card = build(cfg)

    for f in ("train", "train_holdout", "valid", "test", "data_card"):
        assert (out / f"{f}.parquet").exists() or (out / f"{f}.json").exists()

    # Per-source stats
    a, b = card["sources"]
    assert a["read"] == 14 and a["rejected"] == {"flores": 1, "identical": 1, "url": 1}
    assert a["kept"] == 11  # 10 good + 1 exact dup (dedup happens later)
    assert b["read"] == 6 and b["rejected"] == {"flores": 1} and b["kept"] == 5

    # Dedup: 16 kept rows -> 10 unique from A (dup collapsed) + 2 new from B
    assert card["dedup"] == {
        "before": 16,
        "after": 12,
        "removed": 4,
        "by_source": {"a": 10, "b": 2},
        "score_quantiles": {},  # local_pair sources carry no scores
        "train": 11,
        "train_holdout": 1,  # capped at after // 10
    }
    assert card["splits"] == {"train": 11, "train_holdout": 1, "valid": 1, "test": 1}

    train = pq.read_table(out / "train.parquet").to_pydict()
    holdout = pq.read_table(out / "train_holdout.parquet").to_pydict()
    assert set(train) == {"en", "am", "source", "score"} and set(train["score"]) == {None}
    all_en = set(train["en"]) | set(holdout["en"])
    all_am = set(train["am"]) | set(holdout["am"])
    assert len(all_en) == 12 and len(all_am) == 12

    # Detokenized + normalized text made it through
    assert "The quick brown fox jumps over the lazy dog." in all_en
    assert "ፈጣኑ ቡናማ ቀበሮ ሰነፉን ውሻ ዘሎ አለፈ።" in all_am
    # Homophone folding applied to B's new pair (ፀሐይ -> ጸሀይ, ምሥራቅ -> ምስራቅ)
    assert "ጸሀይ በምስራቅ ትወጣለች።" in all_am

    # Zero FLORES leakage on either side, compared on the dedup key
    flores_keys = {dedup_key(t) for p in FLORES_DEV + FLORES_DEVTEST for t in p}
    assert not ({dedup_key(t) for t in all_en | all_am} & flores_keys)

    valid = pq.read_table(out / "valid.parquet").to_pydict()
    assert valid["en_raw"] == [FLORES_DEV[0][0]] and valid["am"] == [FLORES_DEV[0][1]]


def test_rebuild_reuses_shards(workspace: tuple[Config, Path]) -> None:
    cfg, out = workspace
    first = build(cfg)
    a_mtime = (out / "shards" / "a.stats.json").stat().st_mtime_ns
    b_mtime = (out / "shards" / "b.stats.json").stat().st_mtime_ns
    second = build(cfg)  # reuse everything
    assert (out / "shards" / "a.stats.json").stat().st_mtime_ns == a_mtime
    assert first["dedup"] == second["dedup"]
    third = build(cfg, rebuild={"b"})  # selective: only b is re-streamed
    assert (out / "shards" / "a.stats.json").stat().st_mtime_ns == a_mtime
    assert (out / "shards" / "b.stats.json").stat().st_mtime_ns > b_mtime
    assert third["dedup"] == first["dedup"]
    fourth = build(cfg, rebuild={"all"})
    assert fourth["dedup"] == first["dedup"]
    card = json.loads((out / "data_card.json").read_text(encoding="utf-8"))
    assert card["splits"] == first["splits"]
    with pytest.raises(ValueError, match="unknown sources"):
        build(cfg, rebuild={"nope"})


def test_shuffle_is_deterministic(workspace: tuple[Config, Path]) -> None:
    cfg, out = workspace
    build(cfg)
    first = pq.read_table(out / "train.parquet").to_pydict()["en"]
    build(cfg, rebuild={"all"})
    assert pq.read_table(out / "train.parquet").to_pydict()["en"] == first


def test_legacy_shard_without_score_column_is_accepted(workspace: tuple[Config, Path]) -> None:
    """Shards built before the `score` column existed must still merge (read as NULL)."""
    cfg, out = workspace
    build(cfg)
    shard = out / "shards" / "a.parquet"
    legacy = pq.read_table(shard).drop_columns(["score"])
    pq.write_table(legacy, shard)
    card = build(cfg)  # reuses the legacy shard
    assert card["dedup"]["after"] == 12
    assert set(pq.read_table(out / "train.parquet").to_pydict()["score"]) == {None}


def test_opus_moses_zip_with_scores(tmp_path: Path) -> None:
    # Duplicate pair with two scores: dedup must keep the higher-scored copy.
    pairs = [*GOOD[:4], GOOD[0]]
    scores = [1.10, 1.25, 0.95, 1.30, 1.40]
    zip_path = tmp_path / "corpus.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("X.am-en.en", "".join(p[0] + "\n" for p in pairs))
        zf.writestr("X.am-en.am", "".join(p[1] + "\n" for p in pairs))
        zf.writestr("X.am-en.scores", "".join(f"{s}\n" for s in scores))
    for split, sample in (("dev", FLORES_DEV), ("devtest", FLORES_DEVTEST)):
        d = tmp_path / "flores" / split
        d.mkdir(parents=True)
        (d / f"eng_Latn.{split}").write_text(sample[0][0] + "\n", encoding="utf-8")
        (d / f"amh_Ethi.{split}").write_text(sample[0][1] + "\n", encoding="utf-8")

    cfg = Config.model_validate(
        {
            "project": {"name": "t"},
            "paths": {"root": str(tmp_path), "data_raw": "raw", "data_processed": "out"},
            "data": {
                "sources": [{"name": "x", "kind": "opus_moses", "url": zip_path.as_uri()}],
                "flores": {"local_dir": str(tmp_path / "flores")},
                "holdout_size": 0,
            },
        }
    )
    card = build(cfg)
    assert card["dedup"]["after"] == 4
    q = card["dedup"]["score_quantiles"]["x"]
    assert q["min"] == 0.95 and q["max"] == 1.4
    train = pq.read_table(tmp_path / "out" / "train.parquet").to_pydict()
    by_en = dict(zip(train["en"], train["score"], strict=True))
    assert by_en[GOOD[0][0]] == pytest.approx(1.40)  # higher-scored duplicate won
    assert by_en[GOOD[2][0]] == pytest.approx(0.95)
