"""Raw parallel-corpus loaders. Each yields raw `(en, am)` string pairs lazily.

Downloads are cached in `raw_dir` and never re-fetched if the file already exists.
"""

from __future__ import annotations

import io
import itertools
import urllib.request
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from amnmt.core.config import SourceConfig
from amnmt.core.logging import get_logger

log = get_logger(__name__)

Pair = tuple[str, str]


def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        log.info("cached  %s", dest.name)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("download %s -> %s", url, dest)
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    return dest


def _limit(pairs: Iterator[Pair], max_pairs: int | None) -> Iterator[Pair]:
    return itertools.islice(pairs, max_pairs) if max_pairs else pairs


# ---------------------------------------------------------------------------- opus_moses


def iter_opus_moses(url: str, raw_dir: Path, name: str) -> Iterator[Pair]:
    """OPUS 'moses' zips contain two aligned plain-text files: `<corpus>.am-en.en` / `.am`."""
    zip_path = download(url, raw_dir / f"{name}.zip")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        en_name = next(n for n in names if n.endswith(".en"))
        am_name = next(n for n in names if n.endswith(".am"))
        with (
            io.TextIOWrapper(zf.open(en_name), encoding="utf-8", errors="replace") as fen,
            io.TextIOWrapper(zf.open(am_name), encoding="utf-8", errors="replace") as fam,
        ):
            for en, am in zip(fen, fam, strict=False):
                yield en.rstrip("\n"), am.rstrip("\n")


# ---------------------------------------------------------------------------- hf


def _dig(row: dict[str, Any], dotted: str) -> str:
    cur: Any = row
    for part in dotted.split("."):
        cur = cur[part]
    return str(cur)


def iter_hf(
    path: str, subset: str | None, split: str, en_column: str, am_column: str
) -> Iterator[Pair]:
    from datasets import load_dataset

    ds = load_dataset(path, subset, split=split, streaming=True)
    for row in ds:
        yield _dig(row, en_column), _dig(row, am_column)


# ---------------------------------------------------------------------------- local_pair


def iter_local_pair(en_file: Path, am_file: Path) -> Iterator[Pair]:
    with (
        en_file.open(encoding="utf-8", errors="replace") as fen,
        am_file.open(encoding="utf-8", errors="replace") as fam,
    ):
        for en, am in zip(fen, fam, strict=False):
            yield en.rstrip("\n"), am.rstrip("\n")


# ---------------------------------------------------------------------------- dispatch


def iter_source(src: SourceConfig, raw_dir: Path) -> Iterator[Pair]:
    if src.kind == "opus_moses":
        assert src.url is not None
        pairs = iter_opus_moses(src.url, raw_dir, src.name)
    elif src.kind == "hf":
        assert src.path is not None
        pairs = iter_hf(src.path, src.subset, src.split, src.en_column, src.am_column)
    else:
        assert src.en_file is not None and src.am_file is not None
        pairs = iter_local_pair(src.en_file, src.am_file)
    return _limit(pairs, src.max_pairs)
