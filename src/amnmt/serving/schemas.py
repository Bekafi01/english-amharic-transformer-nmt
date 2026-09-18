"""Request/response models for the REST API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Direction = Literal["en-am", "am-en"]


class TranslateRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=64)
    direction: Direction
    beam_size: int = Field(default=4, ge=1, le=8)
    length_penalty: float = Field(default=1.0, ge=0.0, le=2.0)

    model_config = {
        "json_schema_extra": {"examples": [{"texts": ["Good morning."], "direction": "en-am"}]}
    }


class TranslateResponse(BaseModel):
    direction: Direction
    translations: list[str]
    model: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model: str
    device: str
    vocab_size: int
    parameters: int
