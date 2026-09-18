"""FastAPI service. Configure with environment variables:

    AMNMT_CHECKPOINT   path to best.pt (required)
    AMNMT_TOKENIZER    path to tokenizer.json (default: <artifacts>/tokenizer/ next to the run)
    AMNMT_DEVICE       cpu | cuda (default: auto)
    AMNMT_MAX_CHARS    per-sentence input cap (default 2000)

Run:  uvicorn amnmt.serving.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

from amnmt import __version__
from amnmt.core.logging import get_logger
from amnmt.inference.translator import Translator
from amnmt.serving.schemas import HealthResponse, TranslateRequest, TranslateResponse

log = get_logger(__name__)


def load_translator_from_env() -> tuple[Translator, str]:
    ckpt = os.environ.get("AMNMT_CHECKPOINT")
    if not ckpt:
        raise RuntimeError("AMNMT_CHECKPOINT is not set")
    tok = os.environ.get("AMNMT_TOKENIZER") or None
    device = os.environ.get("AMNMT_DEVICE") or None
    log.info("loading %s (tokenizer=%s, device=%s)", ckpt, tok or "auto", device or "auto")
    translator = Translator.from_checkpoint(ckpt, tok, device)
    name = f"{Path(ckpt).parent.name}/{Path(ckpt).name}"
    return translator, name


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    translator, name = load_translator_from_env()
    app.state.translator = translator
    app.state.model_name = name
    app.state.max_chars = int(os.environ.get("AMNMT_MAX_CHARS", "2000"))
    yield


app = FastAPI(
    title="amnmt — English ↔ Amharic translation",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    tr: Translator = request.app.state.translator
    return HealthResponse(
        status="ok",
        model=request.app.state.model_name,
        device=str(tr.device),
        vocab_size=tr.tok.vocab_size,
        parameters=tr.model.num_parameters(),
    )


@app.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest, request: Request) -> TranslateResponse:
    max_chars: int = request.app.state.max_chars
    too_long = [i for i, t in enumerate(req.texts) if len(t) > max_chars]
    if too_long:
        raise HTTPException(
            status_code=413, detail=f"texts {too_long} exceed {max_chars} characters"
        )
    tr: Translator = request.app.state.translator
    out = tr.translate(
        req.texts, req.direction, beam_size=req.beam_size, length_penalty=req.length_penalty
    )
    return TranslateResponse(
        direction=req.direction, translations=out, model=request.app.state.model_name
    )
