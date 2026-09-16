from pathlib import Path

import pytest
from pydantic import ValidationError

from amnmt.core.config import Config, load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_tiny_config_loads() -> None:
    cfg = load_config(CONFIGS / "tiny.yaml")
    assert cfg.project.name == "tiny"
    assert cfg.project.seed == 42


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Config.model_validate({"project": {"name": "x", "typo": 1}})


def test_paths_resolve_relative_to_root(tmp_path: Path) -> None:
    cfg = Config.model_validate({"project": {"name": "x"}, "paths": {"root": str(tmp_path)}})
    assert cfg.paths.resolve("artifacts") == (tmp_path / "artifacts").resolve()


def test_config_is_immutable() -> None:
    cfg = load_config(CONFIGS / "tiny.yaml")
    with pytest.raises(ValidationError):
        cfg.project.seed = 1  # type: ignore[misc]
