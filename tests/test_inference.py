from pathlib import Path

import pytest
import torch
from torch import nn

from amnmt.core.config import ModelConfig
from amnmt.inference.beam import BeamConfig, beam_search
from amnmt.inference.translator import Translator, default_tokenizer_path
from amnmt.model.greedy import greedy_decode
from amnmt.model.transformer import Seq2SeqTransformer
from amnmt.training.trainer import Trainer
from conftest import AM, EN, make_workspace, training_config

PAD, BOS, EOS = 0, 1, 2
V = 40


@pytest.fixture(scope="module")
def reversal_model() -> tuple[Seq2SeqTransformer, torch.Tensor]:
    """Small model trained to reverse sequences; good enough that beam and greedy agree often."""
    torch.manual_seed(3)
    cfg = ModelConfig(
        d_model=64,
        n_heads=4,
        d_ff=128,
        n_encoder_layers=2,
        n_decoder_layers=2,
        dropout=0.0,
        max_len=32,
    )
    m = Seq2SeqTransformer(cfg, vocab_size=V, pad_id=PAD).train()
    n, length = 64, 6
    core = torch.randint(3, V, (n, length))
    src = torch.cat([core, torch.full((n, 1), EOS)], dim=1)
    tgt = core.flip(1)
    tgt_in = torch.cat([torch.full((n, 1), BOS), tgt], dim=1)
    tgt_out = torch.cat([tgt, torch.full((n, 1), EOS)], dim=1)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    for _ in range(250):
        loss = nn.functional.cross_entropy(m(src, tgt_in).reshape(-1, V), tgt_out.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return m.eval(), src


def _seq_logprob(model: Seq2SeqTransformer, src: torch.Tensor, tokens: list[int]) -> float:
    tgt_in = torch.tensor([[BOS, *tokens]])
    tgt_out = torch.tensor([[*tokens, EOS]])
    with torch.no_grad():
        logp = torch.log_softmax(model(src, tgt_in).float(), -1)
    return float(logp.gather(2, tgt_out[..., None]).sum())


def test_beam_size_one_equals_greedy(
    reversal_model: tuple[Seq2SeqTransformer, torch.Tensor],
) -> None:
    m, src = reversal_model
    batch = src[:8]
    greedy = greedy_decode(m, batch, BOS, EOS, max_len=20)
    beam = beam_search(m, batch, BOS, EOS, BeamConfig(beam_size=1, length_penalty=0.0))
    for g, h in zip(greedy, beam, strict=True):
        g_list = g.tolist()
        g_tokens = g_list[: g_list.index(EOS)] if EOS in g_list else g_list
        assert h.tokens == g_tokens


def test_beam_finds_no_worse_sequence_than_greedy(
    reversal_model: tuple[Seq2SeqTransformer, torch.Tensor],
) -> None:
    m, src = reversal_model
    for i in range(6):
        one = src[i : i + 1]
        g = greedy_decode(m, one, BOS, EOS, max_len=20)[0].tolist()
        g_tokens = g[: g.index(EOS)] if EOS in g else g
        h = beam_search(m, one, BOS, EOS, BeamConfig(beam_size=4, length_penalty=0.0))[0]
        assert _seq_logprob(m, one, h.tokens) >= _seq_logprob(m, one, g_tokens) - 1e-4
        assert h.score == pytest.approx(_seq_logprob(m, one, h.tokens), abs=1e-3)


def test_beam_batching_is_order_and_padding_invariant(
    reversal_model: tuple[Seq2SeqTransformer, torch.Tensor],
) -> None:
    m, src = reversal_model
    cfg = BeamConfig(beam_size=3)
    batch = src[:5]
    together = beam_search(m, batch, BOS, EOS, cfg)
    alone = [beam_search(m, batch[i : i + 1], BOS, EOS, cfg)[0] for i in range(5)]
    assert [h.tokens for h in together] == [h.tokens for h in alone]
    padded = torch.cat([batch, torch.full((5, 3), PAD)], dim=1)
    assert [h.tokens for h in beam_search(m, padded, BOS, EOS, cfg)] == [h.tokens for h in together]


def test_beam_reverses_most_inputs(reversal_model: tuple[Seq2SeqTransformer, torch.Tensor]) -> None:
    m, src = reversal_model
    hyps = beam_search(m, src[:32], BOS, EOS, BeamConfig(beam_size=4))
    expected = src[:32, :-1].flip(1).tolist()
    acc = sum(h.tokens == e for h, e in zip(hyps, expected, strict=True)) / 32
    assert acc > 0.8
    assert all(EOS not in h.tokens and BOS not in h.tokens for h in hyps)


def test_forbidden_ids_are_never_generated(
    reversal_model: tuple[Seq2SeqTransformer, torch.Tensor],
) -> None:
    m, src = reversal_model
    banned = tuple(range(3, 20))
    hyps = beam_search(m, src[:4], BOS, EOS, BeamConfig(beam_size=2, forbidden_ids=banned))
    assert all(t not in banned for h in hyps for t in h.tokens)


# ----------------------------------------------------------------------------- translator


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = make_workspace(tmp_path_factory.mktemp("ckpt_ws"))
    run_dir = root / "artifacts" / "runs" / "ckpt"
    cfg = training_config(root, max_steps=12, eval_every=12, save_every=12)
    Trainer(cfg, run_dir).train()
    return run_dir / "best.pt"


def test_default_tokenizer_path(checkpoint: Path) -> None:
    assert default_tokenizer_path(checkpoint).exists()


def test_translator_end_to_end(checkpoint: Path) -> None:
    tr = Translator.from_checkpoint(checkpoint, device="cpu")
    out = tr.translate([EN[0], "", EN[1]], "en-am", beam_size=2)
    assert len(out) == 3 and out[1] == ""
    assert all(isinstance(s, str) for s in out)
    out_rev = tr.translate([AM[0]], "am-en", beam_size=1)
    assert len(out_rev) == 1
    with pytest.raises(ValueError, match="direction"):
        tr.translate(["x"], "fr-en")  # type: ignore[arg-type]


def test_translator_normalizes_input_before_encoding(checkpoint: Path) -> None:
    tr = Translator.from_checkpoint(checkpoint, device="cpu")
    raw, folded = "ሰላም ፡ ፡ ዓለም", "ሰላም። ኣለም"
    assert tr.normalize(raw, "am") == folded
    assert torch.equal(tr.encode_source([raw], "am-en"), tr.encode_source([folded], "am-en"))
    src = tr.encode_source(["Hello world"], "en-am")[0].tolist()
    assert src[0] == tr.tok.lang_id("am") and src[-1] == tr.tok.eos_id
