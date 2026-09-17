"""FLORES-200 dev/devtest loader. Evaluation-only data; the pipeline excludes it from train.

Official tarball layout: flores200_dataset/{dev,devtest}/{lang_code}.{dev,devtest}
"""

from __future__ import annotations

import tarfile
from pathlib import Path

from amnmt.core.config import FloresConfig
from amnmt.data.sources import Pair, download

SPLITS = ("dev", "devtest")
_TAR_ROOT = "flores200_dataset"


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _ensure_extracted(cfg: FloresConfig, raw_dir: Path) -> Path:
    out = raw_dir / _TAR_ROOT
    wanted = {f"{s}/{c}.{s}" for s in SPLITS for c in (cfg.en_code, cfg.am_code)}
    if all((out / w).exists() for w in wanted):
        return out
    tar_path = download(cfg.url, raw_dir / "flores200_dataset.tar.gz")
    with tarfile.open(tar_path) as tf:
        for member in tf.getmembers():
            # Members look like "./flores200_dataset/dev/eng_Latn.dev"; match on the tail.
            tail = "/".join(member.name.split("/")[-2:])
            if member.isfile() and tail in wanted:
                target = out / tail
                target.parent.mkdir(parents=True, exist_ok=True)
                src = tf.extractfile(member)
                assert src is not None
                target.write_bytes(src.read())
    missing = [w for w in wanted if not (out / w).exists()]
    if missing:
        raise FileNotFoundError(f"FLORES tarball lacks {missing}")
    return out


def load_flores(cfg: FloresConfig, raw_dir: Path) -> dict[str, list[Pair]]:
    """Return {"dev": [(en, am), ...], "devtest": [...]} with raw (un-normalized) text."""
    root = cfg.local_dir if cfg.local_dir is not None else _ensure_extracted(cfg, raw_dir)
    out: dict[str, list[Pair]] = {}
    for split in SPLITS:
        en = _read_lines(root / split / f"{cfg.en_code}.{split}")
        am = _read_lines(root / split / f"{cfg.am_code}.{split}")
        if len(en) != len(am):
            raise ValueError(f"FLORES {split}: {len(en)} en lines vs {len(am)} am lines")
        out[split] = list(zip(en, am, strict=True))
    return out
