import pytest
import torch
from torch import Tensor, nn

from amnmt.core.config import ModelConfig
from amnmt.model.layers import Decoder, Encoder, MultiHeadAttention
from amnmt.model.masks import attention_mask, causal_mask, padding_mask
from amnmt.model.transformer import Seq2SeqTransformer

torch.manual_seed(0)
PAD, BOS, EOS = 0, 1, 2
V = 50
CFG = ModelConfig(
    d_model=32, n_heads=4, d_ff=64, n_encoder_layers=2, n_decoder_layers=2, dropout=0.0, max_len=64
)


def _model(cfg: ModelConfig = CFG) -> Seq2SeqTransformer:
    return Seq2SeqTransformer(cfg, vocab_size=V, pad_id=PAD).eval()


# ----------------------------------------------------------------------------- masks


def test_masks() -> None:
    ids = torch.tensor([[5, 6, PAD], [7, PAD, PAD]])
    assert padding_mask(ids, PAD).tolist() == [[True, True, False], [True, False, False]]
    assert causal_mask(3).tolist() == [[1, 0, 0], [1, 1, 0], [1, 1, 1]]
    assert attention_mask(None, None) is None
    m = attention_mask(padding_mask(ids, PAD), causal_mask(3))
    assert m.shape == (2, 1, 3, 3)
    assert m[1, 0].tolist() == [[1, 0, 0], [1, 0, 0], [1, 0, 0]]  # only key 0 is real


# ----------------------------------------------------------------------------- shapes


def test_forward_shapes_and_tying() -> None:
    m = _model()
    src = torch.randint(3, V, (2, 7))
    tgt = torch.randint(3, V, (2, 5))
    logits = m(src, tgt)
    assert logits.shape == (2, 5, V)
    assert m.output(torch.zeros(1, 1, CFG.d_model)).shape == (1, 1, V)
    # 3-way tying: embedding matrix is the only V x d parameter
    big = [n for n, p in m.named_parameters() if p.shape == (V, CFG.d_model)]
    assert big == ["embed.weight"]


def test_sequence_longer_than_max_len_is_rejected() -> None:
    m = _model()
    with pytest.raises(ValueError, match="exceeds max_len"):
        m(torch.randint(3, V, (1, CFG.max_len + 1)), torch.randint(3, V, (1, 2)))


# ----------------------------------------------------------------------------- semantics


def test_decoder_is_causal() -> None:
    m = _model()
    src = torch.randint(3, V, (1, 6))
    tgt = torch.randint(3, V, (1, 5))
    base = m(src, tgt)
    tgt2 = tgt.clone()
    tgt2[0, 3] = (tgt2[0, 3] + 1) % V or 3  # change a later token
    changed = m(src, tgt2)
    assert torch.allclose(base[0, :3], changed[0, :3], atol=1e-6)
    assert not torch.allclose(base[0, 3:], changed[0, 3:])


def test_source_padding_does_not_change_outputs() -> None:
    m = _model()
    src = torch.randint(3, V, (1, 6))
    tgt = torch.randint(3, V, (1, 4))
    padded = torch.cat([src, torch.full((1, 3), PAD)], dim=1)
    assert torch.allclose(m(src, tgt), m(padded, tgt), atol=1e-5)
    mem, _ = m.encode(src)
    mem_p, _ = m.encode(padded)
    assert torch.allclose(mem, mem_p[:, :6], atol=1e-5)


def test_batch_items_are_independent() -> None:
    m = _model()
    src = torch.randint(3, V, (2, 6))
    tgt = torch.randint(3, V, (2, 4))
    both = m(src, tgt)
    solo = m(src[:1], tgt[:1])
    assert torch.allclose(both[:1], solo, atol=1e-5)


# ----------------------------------------------------------------------------- parity


def _copy_mha(ours: MultiHeadAttention, ref: nn.MultiheadAttention) -> None:
    d = ours.q_proj.in_features
    w, b = ref.in_proj_weight.data, ref.in_proj_bias.data
    for i, proj in enumerate((ours.q_proj, ours.k_proj, ours.v_proj)):
        proj.weight.data.copy_(w[i * d : (i + 1) * d])
        proj.bias.data.copy_(b[i * d : (i + 1) * d])
    ours.out_proj.load_state_dict(ref.out_proj.state_dict())


def _copy_encoder(ours: Encoder, ref: nn.TransformerEncoder) -> None:
    for ol, rl in zip(ours.layers, ref.layers, strict=True):
        _copy_mha(ol.self_attn, rl.self_attn)
        ol.ff.linear1.load_state_dict(rl.linear1.state_dict())
        ol.ff.linear2.load_state_dict(rl.linear2.state_dict())
        ol.norm1.load_state_dict(rl.norm1.state_dict())
        ol.norm2.load_state_dict(rl.norm2.state_dict())
    assert ref.norm is not None
    ours.norm.load_state_dict(ref.norm.state_dict())


def _copy_decoder(ours: Decoder, ref: nn.TransformerDecoder) -> None:
    for ol, rl in zip(ours.layers, ref.layers, strict=True):
        _copy_mha(ol.self_attn, rl.self_attn)
        _copy_mha(ol.cross_attn, rl.multihead_attn)
        ol.ff.linear1.load_state_dict(rl.linear1.state_dict())
        ol.ff.linear2.load_state_dict(rl.linear2.state_dict())
        ol.norm1.load_state_dict(rl.norm1.state_dict())
        ol.norm2.load_state_dict(rl.norm2.state_dict())
        ol.norm3.load_state_dict(rl.norm3.state_dict())
    assert ref.norm is not None
    ours.norm.load_state_dict(ref.norm.state_dict())


def test_parity_with_torch_nn_transformer() -> None:
    """Same weights, same masks -> same outputs as torch's reference Pre-LN Transformer."""
    d, h, ff = 32, 4, 64
    ref = nn.Transformer(
        d_model=d,
        nhead=h,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=ff,
        dropout=0.0,
        batch_first=True,
        norm_first=True,
    ).eval()
    enc = Encoder(2, d, h, ff, 0.0, "relu").eval()
    dec = Decoder(2, d, h, ff, 0.0, "relu").eval()
    _copy_encoder(enc, ref.encoder)
    _copy_decoder(dec, ref.decoder)

    b, s, t = 3, 7, 5
    src = torch.randn(b, s, d)
    tgt = torch.randn(b, t, d)
    src_real = torch.ones(b, s, dtype=torch.bool)
    src_real[0, -2:] = False  # pad the first item
    src_real[2, -1] = False

    # torch convention: key_padding_mask True = ignore; tgt_mask True = NOT allowed
    ref_out: Tensor = ref(
        src,
        tgt,
        tgt_mask=~causal_mask(t),
        src_key_padding_mask=~src_real,
        memory_key_padding_mask=~src_real,
    )
    memory = enc(src, attention_mask(src_real))
    ours = dec(tgt, memory, attention_mask(None, causal_mask(t)), attention_mask(src_real))
    assert torch.allclose(ours, ref_out, atol=1e-5), (ours - ref_out).abs().max()


# ----------------------------------------------------------------------------- overfit


def test_overfits_tiny_synthetic_task() -> None:
    """32 pairs, target = reversed source. Loss must collapse within 300 CPU steps."""
    torch.manual_seed(1)
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
    n, length = 32, 8
    src = torch.randint(3, V, (n, length))
    tgt = src.flip(1)
    src = torch.cat([src, torch.full((n, 1), EOS)], dim=1)
    tgt_in = torch.cat([torch.full((n, 1), BOS), tgt], dim=1)
    tgt_out = torch.cat([tgt, torch.full((n, 1), EOS)], dim=1)

    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    loss = torch.tensor(float("inf"))
    steps = 0
    for _ in range(300):
        steps += 1
        logits = m(src, tgt_in)
        loss = nn.functional.cross_entropy(logits.reshape(-1, V), tgt_out.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if loss.item() < 0.05:
            break
    assert loss.item() < 0.1, f"loss {loss.item():.3f} after {steps} steps"
    m.eval()
    assert (m(src, tgt_in).argmax(-1) == tgt_out).float().mean() > 0.99
