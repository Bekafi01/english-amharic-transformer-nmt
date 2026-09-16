"""Unit tests for custom Seq2Seq Transformer architecture and 3-way weight tying."""

import torch

from src.nmt_engine.models.transformer import (
    Decoder,
    Encoder,
    MultiHeadAttention,
    PositionalEncoding,
    PositionwiseFeedForward,
    Seq2SeqTransformer,
)


def test_positional_encoding() -> None:
    pe = PositionalEncoding(d_model=64, max_len=100, dropout=0.0)
    x = torch.zeros(2, 20, 64)
    out = pe(x)
    assert out.shape == (2, 20, 64)
    # Output should not be all zeros because PE is added
    assert not torch.allclose(out, torch.zeros_like(out))


def test_multi_head_attention() -> None:
    mha = MultiHeadAttention(d_model=64, n_heads=4, dropout=0.0)
    q = torch.randn(2, 10, 64)
    k = torch.randn(2, 12, 64)
    v = torch.randn(2, 12, 64)
    mask = torch.ones(2, 1, 10, 12, dtype=torch.bool)
    mask[:, :, :, -2:] = False  # Mask last 2 positions

    out = mha(q, k, v, mask=mask)
    assert out.shape == (2, 10, 64)


def test_feed_forward() -> None:
    ffn_relu = PositionwiseFeedForward(d_model=64, d_ff=256, dropout=0.0, activation="relu")
    ffn_gelu = PositionwiseFeedForward(d_model=64, d_ff=256, dropout=0.0, activation="gelu")
    x = torch.randn(2, 10, 64)

    out_relu = ffn_relu(x)
    out_gelu = ffn_gelu(x)
    assert out_relu.shape == (2, 10, 64)
    assert out_gelu.shape == (2, 10, 64)


def test_encoder_and_decoder_stacks() -> None:
    encoder = Encoder(n_layers=2, d_model=64, n_heads=4, d_ff=128, dropout=0.0)
    decoder = Decoder(n_layers=2, d_model=64, n_heads=4, d_ff=128, dropout=0.0)

    src = torch.randn(2, 8, 64)
    tgt = torch.randn(2, 6, 64)
    src_mask = torch.ones(2, 1, 1, 8, dtype=torch.bool)
    tgt_mask = torch.tril(torch.ones(6, 6, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)

    memory = encoder(src, src_mask)
    assert memory.shape == (2, 8, 64)

    dec_out = decoder(tgt, memory, tgt_mask, src_mask)
    assert dec_out.shape == (2, 6, 64)


def test_seq2seq_transformer_weight_tying() -> None:
    model_tied = Seq2SeqTransformer(
        vocab_size=1000,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        tie_weights=True,
    )
    # Memory identity verification
    assert model_tied.output_proj.weight is model_tied.shared_embedding.weight

    model_untied = Seq2SeqTransformer(
        vocab_size=1000,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        tie_weights=False,
    )
    assert model_untied.output_proj.weight is not model_untied.shared_embedding.weight


def test_seq2seq_transformer_forward_backward() -> None:
    vocab_size = 500
    model = Seq2SeqTransformer(
        vocab_size=vocab_size,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        pad_id=0,
        tie_weights=True,
    )

    B, L_s, L_t = 2, 8, 7
    src = torch.randint(1, vocab_size, (B, L_s))
    tgt = torch.randint(1, vocab_size, (B, L_t))
    src_mask = torch.ones(B, 1, 1, L_s, dtype=torch.bool)
    tgt_mask = torch.tril(torch.ones(L_t, L_t, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)

    logits = model(src, tgt, src_mask, tgt_mask)
    assert logits.shape == (B, L_t, vocab_size)

    # Verify backward gradient flow
    loss = logits.sum()
    loss.backward()

    assert model.shared_embedding.weight.grad is not None
    assert model.output_proj.weight.grad is not None
