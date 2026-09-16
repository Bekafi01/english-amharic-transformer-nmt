"""Typed experiment configuration loaded from YAML.

Every phase adds its own section model here (DataConfig, ModelConfig, ...). Unknown keys are
rejected so a typo in a YAML file fails fast instead of silently using a default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


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


class Config(_Strict):
    project: ProjectConfig
    paths: PathsConfig = Field(default_factory=PathsConfig)


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Config.model_validate(raw)
