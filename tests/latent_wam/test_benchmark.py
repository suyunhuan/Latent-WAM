from scripts.benchmark_latent_wam_training import RepeatedDataset, summarize_seconds


class _SequenceDataset:
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> int:
        return (10, 20)[index]


def test_repeated_dataset_repeats_public_samples_to_requested_length() -> None:
    dataset = RepeatedDataset(_SequenceDataset(), length=5)

    assert len(dataset) == 5
    assert [dataset[index] for index in range(5)] == [10, 20, 10, 20, 10]


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
