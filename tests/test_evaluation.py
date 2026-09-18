import json
from pathlib import Path

import pytest

from amnmt.evaluation.flores import evaluate, run_and_save, to_markdown
from amnmt.evaluation.metrics import bleu, chrf, score
from amnmt.inference.translator import Translator
from amnmt.training.trainer import Trainer
from conftest import AM, EN, make_workspace, training_config

# ----------------------------------------------------------------------------- metrics


def test_perfect_hypotheses_score_100() -> None:
    s = score(EN, EN)
    assert s.bleu == 100.0 and s.chrf == 100.0 and s.spbleu is None
    assert score(AM, AM).chrf == 100.0


def test_empty_hypotheses_score_zero() -> None:
    s = score([""] * len(EN), EN)
    assert s.bleu == 0.0 and s.chrf == 0.0


def test_partial_match_is_between() -> None:
    hyps = ["The quick brown fox jumps over the dog.", *EN[1:]]
    assert 0 < bleu(hyps, EN) < 100 and 0 < chrf(hyps, EN) < 100
    assert chrf(hyps, EN) > chrf(["", *EN[1:]], EN)


def test_metrics_match_sacrebleu_reference_values() -> None:
    # Pinned from sacrebleu 2.6 (BLEU 13a, chrF++ word_order=2); guards the wrapper config.
    hyp = ["the cat sat on the mat"]
    ref = ["the cat is on the mat"]
    assert bleu(hyp, ref) == pytest.approx(37.99, abs=0.05)
    assert chrf(hyp, ref) == pytest.approx(66.36, abs=0.05)


def test_length_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="hypotheses vs"):
        score(["a"], ["a", "b"])


# ----------------------------------------------------------------------------- evaluator


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = make_workspace(tmp_path_factory.mktemp("eval_ws"))
    run_dir = root / "artifacts" / "runs" / "e"
    Trainer(training_config(root, max_steps=12, eval_every=12, save_every=12), run_dir).train()
    return run_dir / "best.pt", root / "data" / "processed" / "valid.parquet"


def test_evaluate_both_directions(trained: tuple[Path, Path]) -> None:
    ckpt, valid = trained
    tr = Translator.from_checkpoint(ckpt, device="cpu")
    res = evaluate(tr, valid, beam_size=2, limit=3, n_examples=2)
    assert res["n"] == 3 and set(res["directions"]) == {"en-am", "am-en"}
    d = res["directions"]["en-am"]
    assert set(d["vs_normalized_ref"]) == {"bleu", "chrf", "spbleu"}
    assert 0 <= d["vs_normalized_ref"]["chrf"] <= 100
    assert len(d["examples"]) == 2 and d["examples"][0]["src"] == EN[4]
    md = to_markdown(res, "t")
    assert "| en-am |" in md and "| am-en |" in md and "## Examples" in md


def test_run_and_save_writes_reports(trained: tuple[Path, Path]) -> None:
    ckpt, valid = trained
    out = ckpt.parent
    res = run_and_save(ckpt, valid, out, beam_size=1, limit=2)
    assert res["checkpoint"] == str(ckpt)
    saved = json.loads((out / "eval_valid_beam1.json").read_text(encoding="utf-8"))
    assert saved["directions"]["am-en"]["vs_raw_ref"]["bleu"] >= 0
    assert (out / "eval_valid_beam1.md").read_text(encoding="utf-8").startswith("# e / best.pt")
