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


def test_all_committed_configs_load() -> None:
    for path in CONFIGS.glob("*.yaml"):
        cfg = load_config(path)
        assert cfg.data is not None, path.name


def test_source_requires_kind_specific_fields() -> None:
    base = {"project": {"name": "x"}}
    with pytest.raises(ValidationError, match="requires \\['url'\\]"):
        Config.model_validate({**base, "data": {"sources": [{"name": "s", "kind": "opus_moses"}]}})
    with pytest.raises(ValidationError, match="requires \\['path'\\]"):
        Config.model_validate({**base, "data": {"sources": [{"name": "s", "kind": "hf"}]}})
    with pytest.raises(ValidationError, match="en_file"):
        Config.model_validate(
            {**base, "data": {"sources": [{"name": "s", "kind": "local_pair", "am_file": "a"}]}}
        )


def test_duplicate_source_names_rejected() -> None:
    srcs = [{"name": "s", "kind": "hf", "path": "p"}] * 2
    with pytest.raises(ValidationError, match="duplicate"):
        Config.model_validate({"project": {"name": "x"}, "data": {"sources": srcs}})


def test_with_paths_override() -> None:
    cfg = load_config(CONFIGS / "tiny.yaml").with_paths(root=Path("/drive"))
    assert cfg.paths.root == Path("/drive")
    assert cfg.paths.data_processed == Path("data/processed/tiny")
