"""Typed experiment configuration loaded from YAML.

Every phase adds its own section model here (DataConfig, ModelConfig, ...). Unknown keys are
rejected so a typo in a YAML file fails fast instead of silently using a default.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

_IDENT = r"^[a-z][a-z0-9_]*$"  # source names are interpolated into SQL; keep them plain


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProjectConfig(_Strict):
    name: str
    seed: int = 42
    device: Literal["auto", "cpu", "cuda"] = "auto"


class PathsConfig(_Strict):
    """All locations are relative to `root` unless absolute."""

    root: Path = Path(".")
    data_raw: Path = Path("data/raw")
    data_processed: Path = Path("data/processed")
    artifacts: Path = Path("artifacts")

    def resolve(self, name: str) -> Path:
        p: Path = getattr(self, name)
        return p if p.is_absolute() else (self.root / p).resolve()


# ------------------------------------------------------------------ Phase 1: data


class SourceConfig(_Strict):
    """One parallel-corpus source. Required fields depend on `kind`."""

    name: str = Field(pattern=_IDENT)
    kind: Literal["opus_moses", "hf", "local_pair"]
    license: str = ""
    max_pairs: int | None = None
    # opus_moses: zip with two aligned plain-text members (*.en, *.am)
    url: str | None = None
    # hf: datasets.load_dataset(path, subset, split=...) streamed; columns may be dotted
    path: str | None = None
    subset: str | None = None
    split: str = "train"
    en_column: str = "translation.en"
    am_column: str = "translation.am"
    # local_pair: two aligned text files
    en_file: Path | None = None
    am_file: Path | None = None

    @model_validator(mode="after")
    def _check_kind_fields(self) -> SourceConfig:
        required = {
            "opus_moses": ("url",),
            "hf": ("path",),
            "local_pair": ("en_file", "am_file"),
        }[self.kind]
        missing = [f for f in required if getattr(self, f) is None]
        if missing:
            raise ValueError(f"source {self.name!r} ({self.kind}) requires {missing}")
        return self


class NormalizeConfig(_Strict):
    fold_homophones: bool = True
    numerals_to_digits: bool = True


class FilterConfig(_Strict):
    min_chars: int = 2
    max_chars: int = 1000
    max_words: int = 150
    max_char_ratio: float = 3.0
    min_am_ethiopic_share: float = 0.6
    min_en_latin_share: float = 0.6
    drop_identical: bool = True
    drop_urls: bool = True
    drop_markup: bool = True


class FloresConfig(_Strict):
    url: str = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
    en_code: str = "eng_Latn"
    am_code: str = "amh_Ethi"
    # If set, read `{split}/{code}.{split}` files from here instead of downloading.
    local_dir: Path | None = None


class DataConfig(_Strict):
    sources: list[SourceConfig] = Field(min_length=1)
    normalize: NormalizeConfig = Field(default_factory=NormalizeConfig)
    filters: FilterConfig = Field(default_factory=FilterConfig)
    flores: FloresConfig = Field(default_factory=FloresConfig)
    holdout_size: int = 2000
    shard_flush_rows: int = 100_000

    @model_validator(mode="after")
    def _unique_names(self) -> DataConfig:
        names = [s.name for s in self.sources]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate source names: {names}")
        return self


# ------------------------------------------------------------------ Phase 2: tokenizer


class SubsetConfig(_Strict):
    """Which rows of train.parquet to use. Shared by tokenizer training and model training."""

    sources: list[str] | None = None  # None = all sources
    min_score: dict[str, float] = Field(
        default_factory=dict
    )  # per-source floor, e.g. {nllb: 1.068}

    @model_validator(mode="after")
    def _plain_names(self) -> SubsetConfig:
        for name in [*(self.sources or []), *self.min_score]:
            if not re.fullmatch(_IDENT, name):
                raise ValueError(f"source name {name!r} must match {_IDENT}")
        return self

    def sql_where(self) -> str:
        clauses: list[str] = []
        if self.sources is not None:
            quoted = ", ".join(f"'{s}'" for s in self.sources)
            clauses.append(f"source IN ({quoted})")
        for src, floor in self.min_score.items():
            clauses.append(f"(source <> '{src}' OR score >= {float(floor)})")
        return " AND ".join(clauses) if clauses else "TRUE"


class TokenizerConfig(_Strict):
    vocab_size: int = 32_000
    min_frequency: int = 2
    # Sentences per language sampled (reservoir, seeded) from the subset for BPE training.
    sample_per_lang: int = 2_000_000
    subset: SubsetConfig = Field(default_factory=SubsetConfig)


class Config(_Strict):
    project: ProjectConfig
    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataConfig | None = None
    tokenizer: TokenizerConfig | None = None

    def with_paths(self, **overrides: Path) -> Config:
        return self.model_copy(update={"paths": self.paths.model_copy(update=overrides)})


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Config.model_validate(raw)
