import os
import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPOSITORY_ROOT / "scripts/training/run_latent_wam_training_multi_node.sh"


def _base_environment(tmp_path: Path) -> dict[str, str]:
    capture_path = tmp_path / "torchrun-args.txt"
    fake_torchrun = tmp_path / "torchrun"
    fake_torchrun.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURE_PATH"\n', encoding="utf-8")
    fake_torchrun.chmod(0o755)
    environment = os.environ.copy()
    for name in (
        "NAVSIM_DEVKIT_ROOT",
        "NAVSIM_EXP_ROOT",
        "OPENSCENE_DATA_ROOT",
        "PYTHON_BIN",
        "NAVSIM_EXPERIMENT_UID",
        "NNODES",
        "NPROC_PER_NODE",
        "NODE_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
        "WORLD_MIRROR_PATH",
        "DINOV2_PATH",
        "CACHE_STAGING_PATH",
    ):
        environment.pop(name, None)
    environment.update(
        {
            "TORCHRUN_BIN": str(fake_torchrun),
            "CAPTURE_PATH": str(capture_path),
        }
    )
    return environment


def test_multi_node_launcher_maps_cluster_environment_to_torchrun(tmp_path: Path) -> None:
    environment = _base_environment(tmp_path)
    environment.update(
        {
            "MLP_WORKER_NUM": "2",
            "MLP_ROLE_INDEX": "1",
            "MLP_WORKER_0_HOST": "trainer-0",
            "MLP_WORKER_0_PORT": "23456",
            "NPROC_PER_NODE": "8",
        }
    )

    subprocess.run(
        ["bash", str(LAUNCHER), "trainer.params.limit_train_batches=5"],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
    )

    arguments = (tmp_path / "torchrun-args.txt").read_text(encoding="utf-8").splitlines()
    assert arguments[:6] == [
        "--nnodes=2",
        "--nproc_per_node=8",
        "--node_rank=1",
        "--master_addr=trainer-0",
        "--master_port=23456",
        str(REPOSITORY_ROOT / "navsim/planning/script/run_training.py"),
    ]
    assert "trainer.params.devices=8" in arguments
    assert "trainer.params.num_nodes=2" in arguments
    assert "dataloader.params.batch_size=8" in arguments
    assert "experiment_uid=navtrain_per_device_bs8_e25_lr2e-4" in arguments
    assert "cache_path=/mnt/oss_data/latent-wam/cache" in arguments
    assert "agent.config.geometry.model_id=/mnt/workspace/wlb/world4drive/checkpoints/HunyuanWorld-Mirror" in arguments
    assert "agent.config.vision.model_id=/mnt/workspace/wlb/world4drive/checkpoints/dinov2-base" in arguments
    assert arguments[-1] == "trainer.params.limit_train_batches=5"


def test_multi_node_launcher_allows_explicit_shared_experiment_uid(tmp_path: Path) -> None:
    environment = _base_environment(tmp_path)
    environment["NAVSIM_EXPERIMENT_UID"] = "shared-run"
    environment["NNODES"] = "2"
    environment["NPROC_PER_NODE"] = "8"

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    arguments = (tmp_path / "torchrun-args.txt").read_text(encoding="utf-8").splitlines()
    assert "experiment_uid=shared-run" in arguments
