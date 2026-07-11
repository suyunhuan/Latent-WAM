import os
import subprocess
import sys

import pytest

from navsim.agents.latent_wam.distributed_cache import DistributedCacheContext, build_frame_cache_plan, plan_shard_ids


def _frame(token: str, timestamp: int, log_name: str = "log") -> dict:
    return {
        "token": token,
        "timestamp": timestamp,
        "log_name": log_name,
        "cams": {
            name: {"data_path": f"{token}-{name}.jpg", "cam_intrinsic": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}
            for name in ("CAM_L0", "CAM_F0", "CAM_R0")
        },
        "ego_dynamic_state": [0.0, 0.0, 0.0, 0.0],
        "driving_command": [1, 0, 0, 0],
    }


def _sample(sample_token: str, start: int, shared_current: str | None = None) -> list[dict]:
    frames = [_frame(f"{sample_token}-frame-{index}", start + index) for index in range(12)]
    if shared_current is not None:
        frames[3] = _frame(shared_current, 3)
    return frames


def test_global_plan_deduplicates_frames_and_packs_log_time_order() -> None:
    scenes = {
        "sample-b": _sample("sample-b", 100, shared_current="shared"),
        "sample-a": _sample("sample-a", 0, shared_current="shared"),
    }

    plan = build_frame_cache_plan(scenes, frame_indices=(0, 3, 7, 11), frames_per_shard=4)

    assert len(plan.samples) == 2
    assert len(plan.frames) == 7
    assert [frame.token for frame in plan.frames] == [
        "sample-a-frame-0",
        "shared",
        "sample-a-frame-7",
        "sample-a-frame-11",
        "sample-b-frame-0",
        "sample-b-frame-7",
        "sample-b-frame-11",
    ]
    assert [(frame.shard_id, frame.row_index) for frame in plan.frames] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 0),
        (1, 1),
        (1, 2),
    ]
    assert plan.sources["shared"].sample_token == "sample-a"
    assert plan.sources["shared"].frame_index == 3


def test_global_plan_rejects_conflicting_repeated_frame_token() -> None:
    scenes = {
        "sample-a": _sample("sample-a", 0, shared_current="shared"),
        "sample-b": _sample("sample-b", 100, shared_current="shared"),
    }
    scenes["sample-b"][3]["timestamp"] = 999

    with pytest.raises(ValueError, match="conflicting source data"):
        build_frame_cache_plan(scenes, frame_indices=(0, 3, 7, 11), frames_per_shard=64)


def test_shard_ownership_is_balanced_and_deterministic() -> None:
    assert plan_shard_ids(num_shards=5, world_size=2, rank=0) == [0, 2, 4]
    assert plan_shard_ids(num_shards=5, world_size=2, rank=1) == [1, 3]


def test_distributed_context_reads_torchrun_environment() -> None:
    context = DistributedCacheContext.from_environment(
        {"RANK": "3", "LOCAL_RANK": "1", "WORLD_SIZE": "8", "LATENT_WAM_DIST_BACKEND": "gloo"},
        cuda_available=False,
    )

    assert (context.rank, context.local_rank, context.world_size, context.backend) == (3, 1, 8, "gloo")


@pytest.mark.skipif(
    os.environ.get("RUN_DISTRIBUTED_CACHE_SMOKE") != "1",
    reason="requires permission to open a local TCPStore socket",
)
def test_two_process_gloo_coordination_smoke() -> None:
    environment = dict(os.environ)
    environment["LATENT_WAM_DIST_BACKEND"] = "gloo"
    environment["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "--module",
            "scripts.smoke_distributed_cache",
        ],
        cwd=os.getcwd(),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert '"world_size": 2' in result.stdout
