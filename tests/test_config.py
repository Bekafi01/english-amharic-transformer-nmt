"""Unit tests for configuration loading, Pydantic schemas, validation, and utilities."""

from pathlib import Path

import pytest
import torch

from src.nmt_engine.config import (
    AppConfig,
    BaseConfig,
    DataConfig,
    ModelArchConfig,
    ModelConfig,
    SplitsConfig,
    load_config,
)
from src.nmt_engine.utils.io import ensure_dir, load_json, load_yaml, save_json, save_yaml
from src.nmt_engine.utils.logging import get_logger, setup_logging, shutdown_logging
from src.nmt_engine.utils.seed import seed_everything


class TestConfigLoadingAndValidation:
    """Tests for Pydantic models and YAML configuration loader."""

    def test_default_config_instantiation(self) -> None:
        """Verifies default instantiations for all config objects."""
        base_cfg = BaseConfig()
        assert base_cfg.project_name == "english-amharic-transformer-nmt"
        assert base_cfg.seed == 42
        assert base_cfg.paths.raw_data_dir == "data/raw"

        data_cfg = DataConfig()
        assert data_cfg.splits.train == 0.90
        assert data_cfg.tokenizer.vocab_size == 32000
        assert data_cfg.preprocessing.normalize_homophones is True

        model_cfg = ModelConfig()
        assert model_cfg.model.d_model == 512
        assert model_cfg.model.nhead == 8
        assert model_cfg.scheduler.type == "noam"

        app_cfg = AppConfig(base=base_cfg, data=data_cfg, model=model_cfg)
        assert app_cfg.base.seed == 42

    def test_load_workspace_configs(self) -> None:
        """Verifies loading the project's actual configs directory."""
        workspace_cfg_dir = Path("configs")
        if workspace_cfg_dir.exists():
            cfg = load_config(workspace_cfg_dir)
            assert cfg.base.project_name == "english-amharic-transformer-nmt"
            assert cfg.data.tokenizer.vocab_size == 32000
            assert cfg.model.model.d_model == 512
            assert "opus100" in cfg.data.sources

    def test_mock_config_loading(self, mock_app_config: AppConfig) -> None:
        """Verifies loading from custom mock config files."""
        assert mock_app_config.base.project_name == "test-nmt"
        assert mock_app_config.model.model.d_model == 128
        assert mock_app_config.data.splits.train == 0.8

    def test_splits_sum_validation(self) -> None:
        """Ensures ValueError is raised when train/val/test do not sum to 1.0."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            SplitsConfig(train=0.8, val=0.1, test=0.2)

    def test_d_model_nhead_divisibility(self) -> None:
        """Ensures ValueError is raised when d_model is not divisible by nhead."""
        with pytest.raises(ValueError, match="must be divisible by nhead"):
            ModelArchConfig(d_model=512, nhead=7)

    def test_create_directories(self, mock_app_config: AppConfig, temp_dir: Path) -> None:
        """Verifies directory creation helper."""
        mock_app_config.create_directories()
        assert Path(mock_app_config.base.paths.raw_data_dir).exists()
        assert Path(mock_app_config.base.paths.checkpoint_dir).exists()

    def test_effective_device(self, mock_app_config: AppConfig) -> None:
        """Verifies device fallback mechanism."""
        mock_app_config.base.device = "cpu"
        dev = mock_app_config.get_effective_device()
        assert dev.type == "cpu"


class TestUtilities:
    """Tests for seed reproducibility, logging, and I/O utilities."""

    def test_seed_everything(self) -> None:
        """Verifies deterministic random number generation across seeds."""
        seed_everything(123)
        t1 = torch.randn(5, 5)
        seed_everything(123)
        t2 = torch.randn(5, 5)
        assert torch.allclose(t1, t2)

    def test_yaml_io(self, temp_dir: Path) -> None:
        """Tests save_yaml and load_yaml."""
        test_file = temp_dir / "test.yaml"
        payload = {"key": "value", "numbers": [1, 2, 3], "nested": {"a": True}}
        saved_path = save_yaml(payload, test_file)
        assert saved_path.exists()
        loaded = load_yaml(test_file)
        assert loaded == payload

    def test_json_io(self, temp_dir: Path) -> None:
        """Tests save_json and load_json."""
        test_file = temp_dir / "test.json"
        payload = {"metric": "bleu", "score": 28.5, "unicode": "ሰላም"}
        save_json(payload, test_file)
        loaded = load_json(test_file)
        assert loaded == payload

    def test_ensure_dir(self, temp_dir: Path) -> None:
        """Tests recursive directory creation."""
        nested = temp_dir / "a" / "b" / "c"
        assert not nested.exists()
        res = ensure_dir(nested)
        assert res.exists()

    def test_logging_setup(self, temp_dir: Path) -> None:
        """Tests setup_logging with file output."""
        log_file = temp_dir / "test.log"
        setup_logging(level="DEBUG", log_file=str(log_file), rich_formatting=False)
        logger = get_logger("test_logger")
        logger.info("Test log message")
        assert log_file.exists()
        with open(log_file, encoding="utf-8") as f:
            content = f.read()
        assert "Test log message" in content
        shutdown_logging()
