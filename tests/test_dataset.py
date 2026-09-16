"""Unit tests for PyTorch translation dataset, bucketing batch sampler, and padding collate."""

from pathlib import Path

import pandas as pd
import pytest
import torch

from src.nmt_engine.data.dataset import (
    BucketBatchSampler,
    TranslationDataset,
    make_causal_mask,
    make_pad_mask,
    pad_collate_fn,
)


@pytest.fixture
def mock_tokenized_parquet(tmp_path: Path) -> Path:
    """Generates a synthetic tokenized dataset parquet file."""
    data = {
        "src_ids": [
            [5, 12, 34, 3],
            [5, 88, 3],
            [5, 10, 20, 30, 40, 50, 60, 3],
            [5, 15, 25, 35, 45, 3],
            [5, 7, 3],
            [5, 101, 102, 103, 104, 105, 106, 107, 108, 3],
        ],
        "tgt_ids": [
            [2, 70, 80, 3],
            [2, 90, 3],
            [2, 11, 22, 33, 44, 55, 66, 3],
            [2, 16, 26, 36, 46, 3],
            [2, 8, 3],
            [2, 201, 202, 203, 204, 205, 206, 207, 208, 3],
        ],
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "mock_tokenized.parquet"
    df.to_parquet(file_path, index=False)
    return file_path


def test_translation_dataset(mock_tokenized_parquet: Path) -> None:
    dataset = TranslationDataset(mock_tokenized_parquet)
    assert len(dataset) == 6
    assert len(dataset.lengths) == 6
    assert dataset.lengths[0] == 4  # max(len(src), len(tgt)) for first sample

    sample = dataset[1]
    assert sample["src_ids"] == [5, 88, 3]
    assert sample["tgt_ids"] == [2, 90, 3]
    assert sample["src_len"] == 3
    assert sample["tgt_len"] == 3


def test_make_pad_mask() -> None:
    x = torch.tensor([[1, 2, 0, 0], [1, 2, 3, 0]])
    mask = make_pad_mask(x, pad_id=0)
    assert mask.shape == (2, 1, 1, 4)
    assert mask[0, 0, 0, :2].all()
    assert not mask[0, 0, 0, 2:].any()
    assert mask[1, 0, 0, :3].all()
    assert not mask[1, 0, 0, 3].item()


def test_make_causal_mask() -> None:
    mask = make_causal_mask(size=4)
    assert mask.shape == (1, 1, 4, 4)
    # Lower triangular should be True, upper strictly False
    assert mask[0, 0, 0, 0].item() is True
    assert mask[0, 0, 0, 1].item() is False
    assert mask[0, 0, 3, 3].item() is True
    assert mask[0, 0, 3, 0].item() is True


def test_pad_collate_fn() -> None:
    batch = [
        ([5, 12, 3], [2, 70, 80, 3]),
        ([5, 88, 99, 100, 3], [2, 90, 3]),
    ]
    src, tgt, src_mask, tgt_mask = pad_collate_fn(batch, pad_id=0)

    assert src.shape == (2, 5)  # padded to max src length 5
    assert tgt.shape == (2, 4)  # padded to max tgt length 4
    assert src_mask.shape == (2, 1, 1, 5)
    assert tgt_mask.shape == (2, 1, 4, 4)

    # Check padding tokens in first sample src (length 3, so last 2 are 0)
    assert src[0, 3].item() == 0
    assert src[0, 4].item() == 0
    assert src_mask[0, 0, 0, 3].item() is False
    assert src_mask[0, 0, 0, 4].item() is False

    # Check causal mask in tgt_mask: position 0 cannot attend to position 1
    assert tgt_mask[0, 0, 0, 1].item() is False


def test_bucket_batch_sampler() -> None:
    lengths = [10, 12, 11, 45, 48, 50, 100, 105, 102, 15, 16, 52]
    batch_size = 3
    sampler = BucketBatchSampler(
        lengths=lengths,
        batch_size=batch_size,
        num_buckets=3,
        shuffle=False,
        drop_last=False,
    )

    batches = list(sampler)
    assert len(batches) == len(sampler)
    all_indices = [idx for b in batches for idx in b]
    assert sorted(all_indices) == list(range(len(lengths)))

    # Test drop_last=True
    sampler_drop = BucketBatchSampler(
        lengths=lengths,
        batch_size=batch_size,
        num_buckets=3,
        shuffle=False,
        drop_last=True,
    )
    batches_drop = list(sampler_drop)
    assert len(batches_drop) == len(sampler_drop)
    for b in batches_drop:
        assert len(b) == batch_size
