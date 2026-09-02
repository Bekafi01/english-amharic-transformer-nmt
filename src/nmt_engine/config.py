"""Configuration management and Pydantic validation for English-Amharic NMT."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, Field, field_validator, model_validator

from src.nmt_engine.utils.io import ensure_dir, load_yaml

# ==============================================================================
# Base Configuration Models
# ==============================================================================


class PathsConfig(BaseModel):
    """Paths for data, models, and output artifacts."""

    raw_data_dir: str = Field(default="data/raw", description="Directory for raw corpus archives")
    processed_data_dir: str = Field(
        default="data/processed", description="Directory for processed splits"
    )
    tokenizer_dir: str = Field(
        default="artifacts/tokenizers", description="Directory for tokenizer vocabularies"
    )
    checkpoint_dir: str = Field(
        default="artifacts/checkpoints", description="Directory for model checkpoints"
    )
    onnx_dir: str = Field(
        default="artifacts/onnx", description="Directory for exported ONNX models"
    )
    benchmarks_dir: str = Field(
        default="artifacts/benchmarks", description="Directory for evaluation and ablation reports"
    )
    tensorboard_dir: str = Field(
        default="artifacts/tensorboard", description="Directory for TensorBoard telemetry"
    )


class LoggingConfig(BaseModel):
    """Logging settings."""

    level: str = Field(default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR)")
    log_to_file: bool = Field(default=True, description="Whether to write logs to a file")
    log_file: str = Field(default="artifacts/training.log", description="Path to log file")
    rich_formatting: bool = Field(
        default=True, description="Enable colorized Rich console formatting"
    )


class BaseConfig(BaseModel):
    """Global project settings."""

    project_name: str = Field(
        default="english-amharic-transformer-nmt", description="Project identifier"
    )
    seed: int = Field(default=42, description="Random seed for reproducibility")
    device: str = Field(
        default="cuda", description="Target hardware device ('cuda', 'cpu', or 'mps')"
    )
    num_workers: int = Field(default=4, description="DataLoader worker subprocesses")
    paths: PathsConfig = Field(default_factory=PathsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# ==============================================================================
# Data Configuration Models
# ==============================================================================


class SourceItemConfig(BaseModel):
    """Configuration for an individual parallel data source."""

    type: Literal["huggingface", "url", "local"] = Field(
        default="huggingface", description="Source ingestion protocol"
    )
    path: str | None = Field(default=None, description="HuggingFace dataset path or local file")
    subset: str | None = Field(default=None, description="HuggingFace dataset subset/name")
    url: str | None = Field(default=None, description="Direct download URL if type is 'url'")
    license: str | None = Field(default=None, description="Corpus licensing information")
    description: str | None = Field(default=None, description="Corpus notes and domain summary")
    enabled: bool = Field(default=True, description="Whether to ingest this source")


class SamplingConfig(BaseModel):
    """Sampling constraints per source and globally."""

    max_samples_per_source: int = Field(
        default=250000, gt=0, description="Max sentence pairs per individual source"
    )
    total_max_samples: int = Field(
        default=500000, gt=0, description="Max aggregated parallel sentence pairs"
    )


class SplitsConfig(BaseModel):
    """Dataset partition ratios."""

    train: float = Field(default=0.90, gt=0.0, lt=1.0)
    val: float = Field(default=0.05, gt=0.0, lt=1.0)
    test: float = Field(default=0.05, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_sum(self) -> SplitsConfig:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-4:
            raise ValueError(
                f"Train ({self.train}), Val ({self.val}), Test ({self.test}) splits must sum to 1.0 (got {total})"
            )
        return self


class PreprocessingConfig(BaseModel):
    """Ethiopic and English cleaning, homophone normalization, and length filtering."""

    normalize_homophones: bool = Field(
        default=True,
        description="Map Amharic homophones (ሀ/ሐ/ኀ/ሃ -> ሀ, ዐ/አ -> አ, ጸ/ፀ -> ጸ, ሠ/ሰ -> ሰ)",
    )
    normalize_punctuation: bool = Field(
        default=True,
        description="Standardize Ethiopic word dividers (፡) and punctuation (። ፣ ፤ ፥ ፦)",
    )
    normalize_numerals: bool = Field(
        default=True, description="Standardize Ethiopic and Arabic numerals"
    )
    remove_extra_whitespace: bool = Field(default=True, description="Collapse duplicate whitespace")
    min_seq_len: int = Field(default=2, ge=1, description="Minimum tokens per sentence")
    max_seq_len: int = Field(default=128, ge=8, description="Maximum tokens per sentence")
    min_len_ratio: float = Field(
        default=0.5, gt=0.0, description="Minimum source/target length ratio"
    )
    max_len_ratio: float = Field(
        default=2.0, gt=0.0, description="Maximum source/target length ratio"
    )
    deduplicate: bool = Field(default=True, description="Remove identical duplicate sentence pairs")

    @model_validator(mode="after")
    def validate_ratios(self) -> PreprocessingConfig:
        if self.min_len_ratio >= self.max_len_ratio:
            raise ValueError(
                f"min_len_ratio ({self.min_len_ratio}) must be < max_len_ratio ({self.max_len_ratio})"
            )
        if self.min_seq_len >= self.max_seq_len:
            raise ValueError(
                f"min_seq_len ({self.min_seq_len}) must be < max_seq_len ({self.max_seq_len})"
            )
        return self


class SpecialTokensConfig(BaseModel):
    """Special tokens used across tokenizer and Transformer model."""

    pad: str = Field(default="<pad>", description="Padding token")
    unk: str = Field(default="<unk>", description="Unknown token")
    bos: str = Field(default="<bos>", description="Beginning of sequence token")
    eos: str = Field(default="<eos>", description="End of sequence token")
    mask: str = Field(default="<mask_src>", description="Mask token")


class TokenizerConfig(BaseModel):
    """Subword tokenizer hyperparameters."""

    type: Literal["bpe", "wordpiece"] = Field(default="bpe", description="Tokenization algorithm")
    vocab_size: int = Field(default=32000, ge=1000, description="Target vocabulary size")
    shared_vocab: bool = Field(
        default=True,
        description="Unified vocabulary across Latin and Ethiopic scripts enabling weight tying",
    )
    min_frequency: int = Field(default=2, ge=1, description="Minimum token frequency for merge")
    special_tokens: SpecialTokensConfig = Field(default_factory=SpecialTokensConfig)


class DataConfig(BaseModel):
    """Aggregate data ingestion, preprocessing, and tokenization configuration."""

    sources: dict[str, SourceItemConfig] = Field(default_factory=dict)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    splits: SplitsConfig = Field(default_factory=SplitsConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)


# ==============================================================================
# Model & Training Configuration Models
# ==============================================================================


class ModelArchConfig(BaseModel):
    """Transformer sequence-to-sequence architecture hyperparameters."""

    architecture: str = Field(default="Seq2SeqTransformer", description="Model architecture name")
    d_model: int = Field(default=512, ge=64, description="Hidden embedding dimension")
    nhead: int = Field(default=8, ge=1, description="Number of parallel attention heads")
    num_encoder_layers: int = Field(default=6, ge=1, description="Encoder layer stack depth")
    num_decoder_layers: int = Field(default=6, ge=1, description="Decoder layer stack depth")
    dim_feedforward: int = Field(default=2048, ge=128, description="FeedForward inner dimension")
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0, description="Dropout probability")
    activation: Literal["relu", "gelu"] = Field(
        default="relu", description="FeedForward activation"
    )
    layer_norm_eps: float = Field(default=1.0e-5, gt=0.0, description="LayerNorm epsilon")
    norm_type: Literal["pre_ln", "post_ln"] = Field(
        default="pre_ln", description="LayerNorm placement (Pre-LN recommended for stability)"
    )
    positional_encoding: Literal["sinusoidal", "learned"] = Field(
        default="sinusoidal", description="Positional encoding mechanism"
    )
    max_position_embeddings: int = Field(
        default=512, ge=64, description="Maximum sequence length support"
    )
    tie_weights: bool = Field(
        default=True,
        description="Tie source embeddings, target embeddings, and output projection matrix",
    )

    @field_validator("d_model")
    @classmethod
    def validate_d_model_divisible(cls, v: int, info: Any) -> int:
        return v

    @model_validator(mode="after")
    def validate_d_model_nhead(self) -> ModelArchConfig:
        if self.d_model % self.nhead != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead})")
        return self


class TrainingConfig(BaseModel):
    """Optimization and training loop settings."""

    batch_size: int = Field(default=32, ge=1, description="Per-device batch size")
    bucket_batching: bool = Field(
        default=True, description="Group similar sequence lengths to minimize padding overhead"
    )
    num_buckets: int = Field(default=10, ge=1, description="Number of length buckets")
    gradient_accumulation_steps: int = Field(
        default=2, ge=1, description="Gradient accumulation steps"
    )
    max_epochs: int = Field(default=30, ge=1, description="Maximum training epochs")
    early_stopping_patience: int = Field(
        default=5, ge=1, description="Patience epochs before early stopping"
    )
    label_smoothing: float = Field(
        default=0.1, ge=0.0, lt=1.0, description="Cross-entropy label smoothing epsilon"
    )
    mixed_precision: Literal["fp16", "bf16", "no"] = Field(
        default="fp16", description="Automatic mixed precision (AMP) mode"
    )
    clip_grad_norm: float = Field(default=1.0, gt=0.0, description="Max gradient norm clipping")


class OptimizerConfig(BaseModel):
    """Optimizer settings."""

    type: Literal["adamw", "adam"] = Field(default="adamw", description="Optimizer type")
    lr: float = Field(default=0.0005, gt=0.0, description="Base learning rate")
    betas: list[float] = Field(default=[0.9, 0.98], description="Adam beta1, beta2 parameters")
    eps: float = Field(default=1.0e-9, gt=0.0, description="Adam numerical epsilon")
    weight_decay: float = Field(default=0.0001, ge=0.0, description="Weight decay factor")

    @field_validator("betas")
    @classmethod
    def validate_betas(cls, v: list[float]) -> list[float]:
        if len(v) != 2:
            raise ValueError("Optimizer betas must contain exactly 2 values [beta1, beta2]")
        if not (0.0 <= v[0] < 1.0 and 0.0 <= v[1] < 1.0):
            raise ValueError("Betas must be in [0.0, 1.0)")
        return v


class SchedulerConfig(BaseModel):
    """Learning rate scheduler settings."""

    type: Literal["noam", "cosine", "linear"] = Field(
        default="noam", description="Scheduler policy"
    )
    warmup_steps: int = Field(default=4000, ge=0, description="Linear warmup step count")


class InferenceConfig(BaseModel):
    """Decoding and inference parameters."""

    beam_size: int = Field(default=4, ge=1, description="Beam search width (1 = greedy)")
    length_penalty: float = Field(
        default=0.6, ge=0.0, description="Alpha parameter for beam search length normalization"
    )
    max_decode_len: int = Field(
        default=128, ge=1, description="Maximum decoded tokens before cutoff"
    )
    repetition_penalty: float = Field(
        default=1.2, ge=1.0, description="Penalty for repeating token sequences"
    )


class ModelConfig(BaseModel):
    """Aggregate model architecture, training, and decoding configuration."""

    model: ModelArchConfig = Field(default_factory=ModelArchConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)


# ==============================================================================
# Aggregated AppConfig & Loader
# ==============================================================================


class AppConfig(BaseModel):
    """Root configuration uniting Base, Data, and Model configurations."""

    base: BaseConfig = Field(default_factory=BaseConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)

    def create_directories(self) -> None:
        """Creates all directories configured under base.paths."""
        paths = self.base.paths
        for _, path_str in paths.model_dump().items():
            ensure_dir(path_str)

    def get_effective_device(self) -> torch.device:
        """Determines target PyTorch device based on user preference and hardware availability."""
        requested = self.base.device.lower()
        if requested == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif (
            requested == "mps"
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")
        return torch.device("cpu")


def load_config(
    config_dir: str | Path = "configs",
    base_config_path: str | Path | None = None,
    data_config_path: str | Path | None = None,
    model_config_path: str | Path | None = None,
) -> AppConfig:
    """Loads and validates base, data, and model YAML configuration files into AppConfig.

    Args:
        config_dir: Directory containing base_config.yaml, data_config.yaml, and model_config.yaml.
        base_config_path: Custom path to base configuration YAML.
        data_config_path: Custom path to data configuration YAML.
        model_config_path: Custom path to model configuration YAML.

    Returns:
        Validated AppConfig instance.
    """
    cdir = Path(config_dir)

    b_path = Path(base_config_path) if base_config_path else cdir / "base_config.yaml"
    d_path = Path(data_config_path) if data_config_path else cdir / "data_config.yaml"
    m_path = Path(model_config_path) if model_config_path else cdir / "model_config.yaml"

    base_dict = load_yaml(b_path) if b_path.exists() else {}
    data_dict = load_yaml(d_path) if d_path.exists() else {}
    model_dict = load_yaml(m_path) if m_path.exists() else {}

    base_cfg = BaseConfig(**base_dict)
    data_cfg = DataConfig(**data_dict)
    model_cfg = ModelConfig(**model_dict)

    app_cfg = AppConfig(base=base_cfg, data=data_cfg, model=model_cfg)
    return app_cfg
