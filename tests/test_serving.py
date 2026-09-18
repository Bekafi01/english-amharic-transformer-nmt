from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from amnmt.training.trainer import Trainer
from conftest import AM, EN, make_workspace, training_config


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = make_workspace(tmp_path_factory.mktemp("serve_ws"))
    run_dir = root / "artifacts" / "runs" / "srv"
    Trainer(training_config(root, max_steps=12, eval_every=12, save_every=12), run_dir).train()
    return run_dir / "best.pt"


@pytest.fixture(scope="module")
def client(checkpoint: Path) -> Iterator[TestClient]:
    mp = pytest.MonkeyPatch()
    mp.setenv("AMNMT_CHECKPOINT", str(checkpoint))
    mp.setenv("AMNMT_DEVICE", "cpu")
    mp.setenv("AMNMT_MAX_CHARS", "200")
    from amnmt.serving.app import app

    with TestClient(app) as c:  # `with` runs the lifespan (model load)
        yield c
    mp.undo()


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model"] == "srv/best.pt"
    assert body["device"] == "cpu" and body["vocab_size"] > 0 and body["parameters"] > 0


def test_translate_both_directions(client: TestClient) -> None:
    r = client.post("/translate", json={"texts": EN[:2], "direction": "en-am", "beam_size": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["direction"] == "en-am" and len(body["translations"]) == 2
    assert all(isinstance(t, str) for t in body["translations"])
    r = client.post("/translate", json={"texts": [AM[0], ""], "direction": "am-en"})
    assert r.status_code == 200 and r.json()["translations"][1] == ""


@pytest.mark.parametrize(
    "payload",
    [
        {"texts": [], "direction": "en-am"},
        {"texts": ["x"], "direction": "fr-en"},
        {"texts": ["x"], "direction": "en-am", "beam_size": 0},
        {"texts": ["x"], "direction": "en-am", "beam_size": 9},
        {"texts": ["x"] * 65, "direction": "en-am"},
        {"direction": "en-am"},
    ],
)
def test_validation_errors(client: TestClient, payload: dict[str, object]) -> None:
    assert client.post("/translate", json=payload).status_code == 422


def test_too_long_input_is_413(client: TestClient) -> None:
    r = client.post("/translate", json={"texts": ["a" * 201], "direction": "en-am"})
    assert r.status_code == 413 and "exceed" in r.json()["detail"]


def test_missing_checkpoint_env_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AMNMT_CHECKPOINT", raising=False)
    from amnmt.serving.app import load_translator_from_env

    with pytest.raises(RuntimeError, match="AMNMT_CHECKPOINT"):
        load_translator_from_env()
