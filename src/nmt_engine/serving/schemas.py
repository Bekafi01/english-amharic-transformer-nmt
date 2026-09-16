"""Pydantic schemas for NMT inference REST API requests and responses."""

from typing import Literal

from pydantic import BaseModel, Field


class TranslationRequest(BaseModel):
    """Schema for single-text translation requests."""

    text: str = Field(
        ...,
        min_length=1,
        description="Source text to translate.",
        examples=["Prime Minister arrived in Addis Ababa."],
    )
    source_lang: Literal["en", "am"] | None = Field(
        None,
        description="Source language code ('en' or 'am'). If omitted, automatically detected from script.",
        examples=["en"],
    )
    target_lang: Literal["en", "am"] | None = Field(
        None,
        description="Target language code ('en' or 'am'). If omitted, inverts the detected source.",
        examples=["am"],
    )
    method: Literal["beam", "greedy"] = Field(
        "beam",
        description="Autoregressive decoding strategy: 'beam' (vectorized beam search) or 'greedy'.",
        examples=["beam"],
    )
    beam_size: int = Field(
        4,
        ge=1,
        le=16,
        description="Number of candidate hypotheses maintained during beam search.",
        examples=[4],
    )
    length_penalty: float = Field(
        0.6,
        ge=0.0,
        le=2.0,
        description="Length normalization exponent alpha (Wu et al., 2016).",
        examples=[0.6],
    )
    repetition_penalty: float = Field(
        1.2,
        ge=1.0,
        le=3.0,
        description="Multiplicative repetition penalty theta applied to previously generated tokens.",
        examples=[1.2],
    )
    max_len: int = Field(
        128,
        ge=4,
        le=512,
        description="Maximum generated sequence length in subword tokens.",
        examples=[128],
    )
    clean_input: bool = Field(
        True,
        description="Whether to apply Ethiopic orthographic and punctuation normalization before translation.",
        examples=[True],
    )


class TranslationResponse(BaseModel):
    """Schema for single-text translation responses."""

    source_text: str = Field(..., description="Original input text.")
    translated_text: str = Field(..., description="Predicted translation.")
    source_lang: str = Field(..., description="Resolved source language code ('en' or 'am').")
    target_lang: str = Field(..., description="Resolved target language code ('en' or 'am').")
    latency_ms: float = Field(..., description="Inference execution latency in milliseconds.")
    method: str = Field(..., description="Decoding method used ('beam' or 'greedy').")
    beam_size: int | None = Field(None, description="Beam size used if beam search.")


class BatchTranslationRequest(BaseModel):
    """Schema for multi-sequence batch translation requests."""

    texts: list[str] = Field(
        ...,
        min_length=1,
        description="List of source sentences to translate.",
        examples=[["Hello world", "How are you?"]],
    )
    source_lang: Literal["en", "am"] | None = Field(
        None,
        description="Source language code ('en' or 'am'). If omitted, detected per sentence.",
    )
    target_lang: Literal["en", "am"] | None = Field(
        None,
        description="Target language code ('en' or 'am'). If omitted, inverts detected source.",
    )
    method: Literal["beam", "greedy"] = Field(
        "beam", description="Decoding method ('beam' or 'greedy')."
    )
    beam_size: int = Field(4, ge=1, le=16, description="Beam width.")
    length_penalty: float = Field(0.6, ge=0.0, le=2.0, description="Length normalization penalty.")
    repetition_penalty: float = Field(1.2, ge=1.0, le=3.0, description="Repetition penalty.")
    max_len: int = Field(128, ge=4, le=512, description="Maximum sequence length.")
    clean_input: bool = Field(True, description="Linguistic normalization flag.")


class BatchTranslationResponse(BaseModel):
    """Schema for multi-sequence batch translation responses."""

    translations: list[TranslationResponse] = Field(
        ..., description="List of individual translation results."
    )
    total_latency_ms: float = Field(..., description="Total wall-clock latency for the batch.")
    count: int = Field(..., description="Total number of processed items.")


class HealthResponse(BaseModel):
    """Schema for service health and hardware diagnostic checks."""

    status: str = Field(..., examples=["healthy"])
    model_loaded: bool = Field(..., description="Whether the neural translation engine is loaded.")
    device: str = Field(..., description="Active compute device ('cuda', 'mps', 'cpu').")
    cuda_available: bool = Field(..., description="True if NVIDIA CUDA runtime is active.")
    gpu_device_name: str | None = Field(None, description="Active GPU model name if CUDA.")
    vram_allocated_mb: float | None = Field(
        None, description="Current PyTorch VRAM consumption in MB."
    )
    uptime_seconds: float = Field(..., description="Service elapsed uptime in seconds.")


class LanguagePair(BaseModel):
    """Schema for supported translation language directions."""

    source: str = Field(..., examples=["en"])
    target: str = Field(..., examples=["am"])
    direction: str = Field(..., examples=["en -> am"])


class LanguagesResponse(BaseModel):
    """Schema listing all supported bidirectional language pairs."""

    supported_pairs: list[LanguagePair] = Field(..., description="Supported language pairs.")


class ModelInfoResponse(BaseModel):
    """Schema detailing loaded Seq2Seq Transformer model architecture and metadata."""

    model_name: str = Field(..., examples=["Pre-LN Seq2Seq Transformer"])
    version: str = Field(..., examples=["1.0.0"])
    parameters: int = Field(..., description="Total trainable parameter count.")
    vocab_size: int = Field(..., description="Subword vocabulary size.")
    d_model: int = Field(..., description="Model hidden dimension.")
    n_heads: int = Field(..., description="Number of multi-head attention heads.")
    n_layers: int = Field(..., description="Number of encoder and decoder layers.")
    d_ff: int = Field(..., description="Position-wise feed-forward dimension.")
    device: str = Field(..., description="Active compute device.")
    checkpoint_path: str | None = Field(None, description="Loaded checkpoint file path.")
    tokenizer_path: str | None = Field(None, description="Loaded tokenizer file path.")
