from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_latent_wam_epdms_parallel import (
    chunk_tokens,
    cpu_group_for_worker,
    merge_trajectory_parts,
    order_score_rows,
    parse_args,
    parse_devices,
    partition_tokens,
    resolve_pdm_worker_count,
    stack_feature_batch,
)


def test_partition_tokens_is_balanced_contiguous_and_ordered() -> None:
    tokens = [f"token-{index}" for index in range(10)]

    shards = partition_tokens(tokens, 3)

    assert [len(shard) for shard in shards] == [4, 3, 3]
    assert [token for shard in shards for token in shard] == tokens


def test_partition_tokens_avoids_empty_shards() -> None:
    assert partition_tokens(["a", "b"], 4) == [["a"], ["b"]]
    assert partition_tokens([], 4) == []


def test_chunk_tokens_creates_dynamic_tasks() -> None:
    assert chunk_tokens(["a", "b", "c", "d", "e"], 2) == [
        ["a", "b"],
        ["c", "d"],
        ["e"],
    ]
    with pytest.raises(ValueError, match="positive"):
        chunk_tokens(["a"], 0)


def test_parse_devices_rejects_empty_and_duplicate_values() -> None:
    assert parse_devices("0, 1") == ["0", "1"]
    with pytest.raises(ValueError, match="at least one"):
        parse_devices(",")
    with pytest.raises(ValueError, match="duplicates"):
        parse_devices("0,0")


def test_resolve_pdm_worker_count_uses_cpu_aware_auto_default() -> None:
    assert resolve_pdm_worker_count(0, token_count=100, cpu_count=22) == 8
    assert resolve_pdm_worker_count(0, token_count=3, cpu_count=22) == 3
    assert resolve_pdm_worker_count(12, token_count=100, cpu_count=22) == 12
    with pytest.raises(ValueError, match="non-negative"):
        resolve_pdm_worker_count(-1, token_count=100, cpu_count=22)


def test_cpu_group_for_worker_assigns_disjoint_compact_groups() -> None:
    cpu_ids = [4, 5, 6, 7, 8, 9]

    assert cpu_group_for_worker(1, cpu_ids, 2) == (4, 5)
    assert cpu_group_for_worker(2, cpu_ids, 2) == (6, 7)
    assert cpu_group_for_worker(3, cpu_ids, 2) == (8, 9)
    assert cpu_group_for_worker(4, cpu_ids, 2) == (4, 5)


def test_parallel_cli_parses_inference_and_pdm_worker_counts() -> None:
    args = parse_args(
        [
            "--checkpoint",
            "model.safetensors",
            "--navsim-log-path",
            "logs",
            "--sensor-path",
            "sensors",
            "--metric-cache",
            "cache",
            "--output",
            "scores.csv",
            "--devices",
            "0,1",
            "--workers-per-device",
            "2",
            "--inference-batch-size",
            "16",
            "--pdm-workers",
            "8",
            "--pdm-chunk-size",
            "32",
        ]
    )

    assert args.checkpoint == Path("model.safetensors")
    assert args.devices == ["0", "1"]
    assert args.inference_workers_per_device == 2
    assert args.inference_batch_size == 16
    assert args.pdm_workers == 8
    assert args.pdm_chunk_size == 32


def test_stack_feature_batch_stacks_every_feature_key() -> None:
    import torch

    rows = [
        {"images": torch.tensor([index]), "ego_status": torch.tensor([index + 10])}
        for index in range(3)
    ]

    batch = stack_feature_batch(rows, torch.device("cpu"))

    assert batch["images"].tolist() == [[0], [1], [2]]
    assert batch["ego_status"].tolist() == [[10], [11], [12]]


def test_merge_trajectory_parts_restores_token_order(tmp_path: Path) -> None:
    part_a = tmp_path / "a.npy"
    part_b = tmp_path / "b.npy"
    np.save(part_a, np.asarray([[[2.0]], [[0.0]]]))
    np.save(part_b, np.asarray([[[1.0]]]))

    output = merge_trajectory_parts(
        [part_a, part_b],
        [["c", "a"], ["b"]],
        ["a", "b", "c"],
        tmp_path / "merged.npy",
    )

    assert np.load(output).reshape(-1).tolist() == [0.0, 1.0, 2.0]


def test_merge_trajectory_parts_rejects_missing_tokens(tmp_path: Path) -> None:
    part = tmp_path / "part.npy"
    np.save(part, np.asarray([[[1.0]]]))

    with pytest.raises(ValueError, match="missing"):
        merge_trajectory_parts([part], [["a"]], ["a", "b"], tmp_path / "merged.npy")


def test_order_score_rows_restores_order_and_rejects_incomplete_rows() -> None:
    rows = pd.DataFrame(
        [
            {"token": "b", "valid": True, "score": 0.4},
            {"token": "a", "valid": True, "score": 0.8},
        ]
    )

    ordered = order_score_rows(rows, ["a", "b"])

    assert ordered["token"].tolist() == ["a", "b"]
    with pytest.raises(ValueError, match="missing"):
        order_score_rows(rows.iloc[:1], ["a", "b"])
    with pytest.raises(ValueError, match="duplicate"):
        order_score_rows(pd.concat([rows, rows.iloc[:1]]), ["a", "b"])
