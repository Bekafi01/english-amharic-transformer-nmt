"""FastAPI REST inference service for English-Amharic Neural Machine Translation."""

import os
import re
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.nmt_engine.inference.translator import Translator
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
from src.nmt_engine.utils.logging import get_logger

logger = get_logger(__name__)

# Server launch timestamp for uptime tracking
_START_TIME = time.time()

# Ethiopic Unicode block regex (Ethiopic, Ethiopic Supplement, Ethiopic Extended, Extended-A)
ETHIOPIC_PATTERN = re.compile(r"[\u1200-\u137f\u1380-\u139f\u2d80-\u2ddf\uab00-\uab2f]")


def detect_script_language(text: str) -> str:
    """Detects whether text is primarily Ethiopic ('am') or Latin ('en')."""
    if ETHIOPIC_PATTERN.search(text):
        return "am"
    return "en"


def create_app(translator: Translator | None = None) -> FastAPI:
    """Application factory constructing the FastAPI instance with optional pre-loaded Translator."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        # Startup logic
        logger.info("Initializing English-Amharic NMT REST service...")

        if getattr(app.state, "translator", None) is None:
            # Check environment variables for model and tokenizer paths
            model_path_str = os.getenv("NMT_MODEL_PATH", "checkpoints/best_model.pt")
            tok_path_str = os.getenv(
                "NMT_TOKENIZER_PATH", "artifacts/tokenizers/joint_bpe_32k.json"
            )
            device_str = os.getenv("NMT_DEVICE", "auto")

            model_p = Path(model_path_str)
            tok_p = Path(tok_path_str)

            if model_p.exists() and tok_p.exists():
                try:
                    logger.info(
                        f"Loading model checkpoint from {model_p} with tokenizer {tok_p} on {device_str}..."
                    )
                    app.state.translator = Translator(
                        model_path=model_p,
                        tokenizer_path=tok_p,
                        device=device_str,
                    )
                    logger.info("Translator successfully loaded and ready for inference.")
                except Exception as exc:
                    logger.warning(
                        f"Failed to load translation model on startup: {exc}. Service will start in standby mode."
                    )
                    app.state.translator = None
            else:
                logger.info(
                    f"Model checkpoint ({model_p}) or tokenizer ({tok_p}) not found. Service running in standby mode."
                )
                app.state.translator = None

        yield

        # Shutdown logic
        logger.info("Shutting down English-Amharic NMT REST service.")

    app = FastAPI(
        title="English-Amharic Transformer NMT API",
        description="Production REST API for high-performance bidirectional English <-> Amharic translation.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Allow CORS for front-end integration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Attach pre-loaded translator if provided
    app.state.translator = translator

    @app.get("/", tags=["System"])
    async def root() -> dict[str, str]:
        """Root status and documentation directory."""
        return {
            "service": "English-Amharic Neural Machine Translation Engine",
            "version": "1.0.0",
            "status": "online",
            "documentation": "/docs",
        }

    @app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
    async def health_check() -> HealthResponse:
        """Returns service status, device info, GPU memory diagnostics, and uptime."""
        trans: Translator | None = getattr(app.state, "translator", None)
        cuda_avail = torch.cuda.is_available()

        gpu_name = None
        vram_mb = None
        device_name = "cpu"

        if trans is not None:
            device_name = str(trans.device)

        if cuda_avail:
            gpu_name = torch.cuda.get_device_name(0)
            vram_mb = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2)

        uptime = round(time.time() - _START_TIME, 2)

        return HealthResponse(
            status="healthy",
            model_loaded=(trans is not None),
            device=device_name,
            cuda_available=cuda_avail,
            gpu_device_name=gpu_name,
            vram_allocated_mb=vram_mb,
            uptime_seconds=uptime,
        )

    @app.get("/languages", response_model=LanguagesResponse, tags=["Configuration"])
    async def get_supported_languages() -> LanguagesResponse:
        """Lists supported bidirectional translation directions."""
        return LanguagesResponse(
            supported_pairs=[
                LanguagePair(source="en", target="am", direction="en -> am"),
                LanguagePair(source="am", target="en", direction="am -> en"),
            ]
        )

    @app.get("/models", response_model=ModelInfoResponse, tags=["Configuration"])
    async def get_model_info() -> ModelInfoResponse:
        """Returns architectural parameters, layer dimensions, and vocabulary size."""
        trans: Translator | None = getattr(app.state, "translator", None)
        if trans is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Translation model is not loaded. Ensure checkpoint and tokenizer are configured.",
            )

        model = trans.model
        param_count = sum(p.numel() for p in model.parameters())

        return ModelInfoResponse(
            model_name="Pre-LN Seq2Seq Transformer",
            version="1.0.0",
            parameters=param_count,
            vocab_size=trans.tokenizer.vocab_size,
            d_model=model.d_model,
            n_heads=model.encoder.layers[0].self_attn.n_heads if model.encoder.layers else 8,
            n_layers=len(model.encoder.layers),
            d_ff=model.encoder.layers[0].ffn.linear1.out_features if model.encoder.layers else 2048,
            device=str(trans.device),
        )

    @app.post("/translate", response_model=TranslationResponse, tags=["Translation"])
    async def translate_single(request: TranslationRequest) -> TranslationResponse:
        """Translates a single sentence between English and Amharic."""
        trans: Translator | None = getattr(app.state, "translator", None)
        if trans is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Translation model is not loaded. Checkpoints must be available to serve inference.",
            )

        text = request.text.strip()
        if not text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Input text cannot be empty or solely whitespace.",
            )

        # Resolve source language
        src_lang = request.source_lang
        if src_lang is None:
            src_lang = detect_script_language(text)

        # Resolve target language
        tgt_lang = request.target_lang
        if tgt_lang is None:
            tgt_lang = "am" if src_lang == "en" else "en"

        if src_lang == tgt_lang:
            return TranslationResponse(
                source_text=text,
                translated_text=text,
                source_lang=src_lang,
                target_lang=tgt_lang,
                latency_ms=0.0,
                method=request.method,
                beam_size=request.beam_size if request.method == "beam" else None,
            )

        t0 = time.perf_counter()
        try:
            translated = trans.translate(
                text=text,
                source_lang=src_lang,
                target_lang=tgt_lang,
                method=request.method,
                beam_size=request.beam_size,
                length_penalty=request.length_penalty,
                repetition_penalty=request.repetition_penalty,
                max_len=request.max_len,
            )
        except Exception as exc:
            logger.error(f"Inference error during translation: {exc}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Inference execution failed: {str(exc)}",
            ) from exc
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        return TranslationResponse(
            source_text=text,
            translated_text=translated,
            source_lang=src_lang,
            target_lang=tgt_lang,
            latency_ms=latency_ms,
            method=request.method,
            beam_size=request.beam_size if request.method == "beam" else None,
        )

    @app.post("/translate/batch", response_model=BatchTranslationResponse, tags=["Translation"])
    async def translate_batch(request: BatchTranslationRequest) -> BatchTranslationResponse:
        """Translates a batch of sentences between English and Amharic."""
        trans: Translator | None = getattr(app.state, "translator", None)
        if trans is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Translation model is not loaded. Checkpoints must be available to serve inference.",
            )

        if not request.texts:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Batch 'texts' list cannot be empty.",
            )

        results: list[TranslationResponse] = []
        t0_total = time.perf_counter()

        for raw_text in request.texts:
            text = raw_text.strip()
            if not text:
                results.append(
                    TranslationResponse(
                        source_text=raw_text,
                        translated_text="",
                        source_lang=request.source_lang or "en",
                        target_lang=request.target_lang or "am",
                        latency_ms=0.0,
                        method=request.method,
                        beam_size=request.beam_size if request.method == "beam" else None,
                    )
                )
                continue

            src_lang = request.source_lang or detect_script_language(text)
            tgt_lang = request.target_lang or ("am" if src_lang == "en" else "en")

            if src_lang == tgt_lang:
                results.append(
                    TranslationResponse(
                        source_text=text,
                        translated_text=text,
                        source_lang=src_lang,
                        target_lang=tgt_lang,
                        latency_ms=0.0,
                        method=request.method,
                        beam_size=request.beam_size if request.method == "beam" else None,
                    )
                )
                continue

            t0_item = time.perf_counter()
            translated = trans.translate(
                text=text,
                source_lang=src_lang,
                target_lang=tgt_lang,
                method=request.method,
                beam_size=request.beam_size,
                length_penalty=request.length_penalty,
                repetition_penalty=request.repetition_penalty,
                max_len=request.max_len,
            )
            item_latency = round((time.perf_counter() - t0_item) * 1000, 2)

            results.append(
                TranslationResponse(
                    source_text=text,
                    translated_text=translated,
                    source_lang=src_lang,
                    target_lang=tgt_lang,
                    latency_ms=item_latency,
                    method=request.method,
                    beam_size=request.beam_size if request.method == "beam" else None,
                )
            )

        total_latency_ms = round((time.perf_counter() - t0_total) * 1000, 2)

        return BatchTranslationResponse(
            translations=results,
            total_latency_ms=total_latency_ms,
            count=len(results),
        )

    return app


# Default app instance
app = create_app()
