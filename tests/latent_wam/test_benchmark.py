from pathlib import Path

import pytest

from scripts.benchmark_latent_wam_training import RepeatedDataset, select_sample, summarize_seconds


class _SequenceDataset:
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> int:
        return (10, 20)[index]


def test_repeated_dataset_repeats_public_samples_to_requested_length() -> None:
    dataset = RepeatedDataset(_SequenceDataset(), length=5)

    assert len(dataset) == 5
    assert [dataset[index] for index in range(5)] == [10, 20, 10, 20, 10]


def test_repeated_dataset_repeats_sample_locality_with_public_samples() -> None:
    dataset = RepeatedDataset(_SequenceDataset(), length=5, sample_locality=[3, 7])

    assert dataset.sample_locality == [3, 7, 3, 7, 3]


class _ShardedCacheDataset(_SequenceDataset):
    samples = [{"token": "alpha"}, {"token": "beta"}]


class _SampleCacheDataset(_SequenceDataset):
    samples = [
        (Path("log/alpha/features.gz"), Path("log/alpha/targets.gz")),
        (Path("log/beta/features.gz"), Path("log/beta/targets.gz")),
    ]


@pytest.mark.parametrize("dataset", [_ShardedCacheDataset(), _SampleCacheDataset()])
def test_select_sample_supports_sharded_and_sample_cache_metadata(dataset: _SequenceDataset) -> None:
    selected = select_sample(dataset, "beta")

    assert len(selected) == 1
    assert selected[0] == 20


def test_select_sample_rejects_unknown_token() -> None:
    with pytest.raises(ValueError, match="not present"):
        select_sample(_ShardedCacheDataset(), "missing")


def test_duration_summary_reports_interpolated_percentiles() -> None:
    summary = summarize_seconds([1.0, 2.0, 3.0, 4.0, 5.0])

    assert summary == {
        "mean": 3.0,
        "p50": 3.0,
        "p95": 4.8,
        "p99": 4.96,
        "min": 1.0,
        "max": 5.0,
    }
