"""Production REST serving layer for English-Amharic NMT."""

from src.nmt_engine.serving.app import app, create_app
from src.nmt_engine.serving.schemas import (
    BatchTranslationRequest,
    BatchTranslationResponse,
    HealthResponse,
    LanguagePair,
    LanguagesResponse,
    ModelInfoResponse,
    TranslationRequest,
    TranslationResponse,
)

__all__ = [
    "app",
    "create_app",
    "TranslationRequest",
    "TranslationResponse",
    "BatchTranslationRequest",
    "BatchTranslationResponse",
    "HealthResponse",
    "LanguagesResponse",
    "LanguagePair",
    "ModelInfoResponse",
]
