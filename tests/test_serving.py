"""Unit tests for FastAPI REST serving endpoints and Pydantic schemas."""

from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from src.nmt_engine.data.tokenizer import JointBpeTokenizer
from src.nmt_engine.inference.translator import Translator
from src.nmt_engine.models.transformer import Seq2SeqTransformer
from src.nmt_engine.serving.app import create_app


@pytest.fixture
def mock_translator(tmp_path: Path) -> Translator:
    """Provides a lightweight Translator instance for fast API endpoint verification."""
    texts = [
        "Hello world",
        "ሰላም ዓለም",
        "Good morning",
        "እንደምን አደሩ",
        "Thank you",
        "አመሰግናለሁ",
    ]

    tok = JointBpeTokenizer()
    tok.train_from_iterator(iter([[t] for t in texts]), vocab_size=100, min_frequency=1)
    tok_path = tmp_path / "tokenizer.json"
    tok.save(tok_path)

    model = Seq2SeqTransformer(
        vocab_size=tok.vocab_size,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        pad_id=tok.pad_id,
        tie_weights=True,
    )
    model.eval()

    ckpt_path = tmp_path / "best_model.pt"
    torch.save({"model_state": model.state_dict()}, ckpt_path)

    return Translator(
        model=model,
        tokenizer=tok,
        device="cpu",
        beam_size=2,
        max_len=16,
    )


@pytest.fixture
def client_with_model(mock_translator: Translator) -> TestClient:
    """TestClient with an initialized mock Translator."""
    app = create_app(translator=mock_translator)
    return TestClient(app)


@pytest.fixture
def client_standby() -> TestClient:
    """TestClient in standby mode (no model loaded)."""
    app = create_app(translator=None)
    return TestClient(app)


def test_root_endpoint(client_standby: TestClient) -> None:
    """Verifies the root greeting and documentation endpoint."""
    response = client_standby.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "/docs" in data["documentation"]


def test_health_endpoint_standby(client_standby: TestClient) -> None:
    """Verifies health check in standby mode."""
    response = client_standby.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is False
    assert data["uptime_seconds"] >= 0.0


def test_health_endpoint_loaded(client_with_model: TestClient) -> None:
    """Verifies health check when model is loaded."""
    response = client_with_model.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["device"] == "cpu"


def test_languages_endpoint(client_standby: TestClient) -> None:
    """Verifies listing supported translation language directions."""
    response = client_standby.get("/languages")
    assert response.status_code == 200
    data = response.json()
    assert len(data["supported_pairs"]) == 2
    directions = [p["direction"] for p in data["supported_pairs"]]
    assert "en -> am" in directions
    assert "am -> en" in directions


def test_model_info_standby(client_standby: TestClient) -> None:
    """Verifies 503 response when requesting model info without loaded model."""
    response = client_standby.get("/models")
    assert response.status_code == 503
    assert "not loaded" in response.json()["detail"]


def test_model_info_loaded(client_with_model: TestClient) -> None:
    """Verifies model metadata when model is loaded."""
    response = client_with_model.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "Pre-LN Seq2Seq Transformer"
    assert data["parameters"] > 0
    assert data["d_model"] == 64
    assert data["n_heads"] == 4
    assert data["n_layers"] == 2
    assert data["d_ff"] == 128


def test_translate_standby_error(client_standby: TestClient) -> None:
    """Verifies 503 response when translating without loaded model."""
    payload = {"text": "Hello world", "source_lang": "en", "target_lang": "am"}
    response = client_standby.post("/translate", json=payload)
    assert response.status_code == 503


def test_translate_empty_text_error(client_with_model: TestClient) -> None:
    """Verifies 422/400 validation error on empty string."""
    payload = {"text": "", "source_lang": "en", "target_lang": "am"}
    response = client_with_model.post("/translate", json=payload)
    assert response.status_code == 422  # Pydantic min_length=1 validation error


def test_translate_single_beam(client_with_model: TestClient) -> None:
    """Verifies single-sentence beam search translation."""
    payload = {
        "text": "Hello world",
        "source_lang": "en",
        "target_lang": "am",
        "method": "beam",
        "beam_size": 2,
    }
    response = client_with_model.post("/translate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["source_text"] == "Hello world"
    assert isinstance(data["translated_text"], str)
    assert data["source_lang"] == "en"
    assert data["target_lang"] == "am"
    assert data["latency_ms"] >= 0.0
    assert data["method"] == "beam"
    assert data["beam_size"] == 2


def test_translate_single_greedy(client_with_model: TestClient) -> None:
    """Verifies single-sentence greedy search translation."""
    payload = {
        "text": "Hello world",
        "source_lang": "en",
        "target_lang": "am",
        "method": "greedy",
    }
    response = client_with_model.post("/translate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["method"] == "greedy"
    assert data["beam_size"] is None


def test_translate_auto_language_detection(client_with_model: TestClient) -> None:
    """Verifies automatic script detection for Amharic text."""
    payload = {
        "text": "ሰላም ዓለም",
        "method": "greedy",
    }
    response = client_with_model.post("/translate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["source_lang"] == "am"
    assert data["target_lang"] == "en"


def test_translate_identical_languages(client_with_model: TestClient) -> None:
    """Verifies identity mapping when source and target languages match."""
    payload = {
        "text": "Hello world",
        "source_lang": "en",
        "target_lang": "en",
    }
    response = client_with_model.post("/translate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["translated_text"] == "Hello world"
    assert data["latency_ms"] == 0.0


def test_translate_batch(client_with_model: TestClient) -> None:
    """Verifies batch translation endpoint."""
    payload = {
        "texts": ["Hello world", "Good morning", "Thank you"],
        "source_lang": "en",
        "target_lang": "am",
        "method": "beam",
        "beam_size": 2,
    }
    response = client_with_model.post("/translate/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 3
    assert len(data["translations"]) == 3
    assert data["total_latency_ms"] >= 0.0
    for item in data["translations"]:
        assert isinstance(item["translated_text"], str)
        assert item["source_lang"] == "en"
        assert item["target_lang"] == "am"
