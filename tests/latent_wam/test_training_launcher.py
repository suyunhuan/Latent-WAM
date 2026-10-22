import os
import subprocess
from pathlib import Path

from pytest import MonkeyPatch

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
        "TRAINING_CONFIG",
        "TRAIN_TEST_SPLIT",
        "WORLD_MIRROR_PATH",
        "DINOV2_PATH",
        "CACHE_STAGING_PATH",
        "BATCH_SIZE",
        "NUM_WORKERS",
        "PRECISION",
        "MAX_EPOCHS",
        "ACCUMULATE_GRAD_BATCHES",
        "LIMIT_VAL_BATCHES",
        "CACHE_SAMPLER_MODE",
        "LEARNING_RATE",
        "MLP_WORKER_NUM",
        "MLP_ROLE_INDEX",
        "MLP_WORKER_0_HOST",
        "MLP_WORKER_0_PORT",
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
    assert "--config-name=all_navtrain_training" in arguments
    assert "train_test_split=trainval" in arguments
    assert "trainer.params.num_nodes=2" in arguments
    assert "dataloader.params.batch_size=8" in arguments
    assert "experiment_uid=navtrain_bs512_epoch100_lr2e-4_global" in arguments
    assert "cache_path=/mnt/oss_data/latent-wam/cache" in arguments
    assert f"agent.config.geometry.model_id={REPOSITORY_ROOT / 'checkpoints/HunyuanWorld-Mirror'}" in arguments
    assert f"agent.config.vision.model_id={REPOSITORY_ROOT / 'checkpoints/dinov2-base'}" in arguments
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


def test_multi_node_launcher_can_select_disjoint_navtrain_config(tmp_path: Path) -> None:
    environment = _base_environment(tmp_path)
    environment["TRAINING_CONFIG"] = "default_training"

    subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
    )

    arguments = (tmp_path / "torchrun-args.txt").read_text(encoding="utf-8").splitlines()
    assert "--config-name=default_training" in arguments
    assert "train_test_split=navtrain" in arguments


def test_launcher_defaults_compose_reproduction_recipe(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    from hydra import compose, initialize_config_dir

    environment = _base_environment(tmp_path)
    subprocess.run(["bash", str(LAUNCHER)], cwd=REPOSITORY_ROOT, env=environment, check=True)
    arguments = (tmp_path / "torchrun-args.txt").read_text(encoding="utf-8").splitlines()
    assert arguments[:2] == ["--nnodes=8", "--nproc_per_node=8"]
    assert arguments[6] == "--config-name=all_navtrain_training"
    monkeypatch.setenv("NAVSIM_EXP_ROOT", str(tmp_path))
    monkeypatch.setenv("OPENSCENE_DATA_ROOT", str(tmp_path))
    with initialize_config_dir(
        config_dir=str(REPOSITORY_ROOT / "navsim/planning/script/config/training"), version_base="1.3"
    ):
        config = compose(config_name="all_navtrain_training", overrides=arguments[7:])
    assert config.agent._target_ == "navsim.agents.latent_wam.agent.LatentWAMAgent"
    assert config.agent.build_target_encoder
    assert config.trainer.params.num_nodes == config.trainer.params.devices == 8
    assert config.dataloader.params.batch_size == 8
    assert config.trainer.params.accumulate_grad_batches == 1
    assert config.trainer.params.max_epochs == 100
    assert config.trainer.params.limit_val_batches == 5
    assert config.trainer.params.precision == "bf16-mixed"
    assert config.agent.config.training.learning_rate == 2e-4
    assert config.agent.config.training.temporal_attention_mask == "block_causal"
    assert config.cache_sampler_mode == "global"
    assert config.use_cache_without_dataset and not config.force_cache_computation
    assert set(config.val_logs).issubset(config.train_logs)


def test_launcher_retains_environment_and_hydra_overrides(tmp_path: Path) -> None:
    environment = _base_environment(tmp_path)
    environment.update(
        NNODES="2",
        NPROC_PER_NODE="4",
        BATCH_SIZE="3",
        MAX_EPOCHS="2",
        ACCUMULATE_GRAD_BATCHES="2",
        LIMIT_VAL_BATCHES="0",
        CACHE_SAMPLER_MODE="locality",
    )
    subprocess.run(
        ["bash", str(LAUNCHER), "trainer.params.max_epochs=3"],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
    )
    arguments = (tmp_path / "torchrun-args.txt").read_text(encoding="utf-8").splitlines()
    assert arguments[:2] == ["--nnodes=2", "--nproc_per_node=4"]
    for value in [
        "trainer.params.num_nodes=2",
        "trainer.params.devices=4",
        "dataloader.params.batch_size=3",
        "trainer.params.max_epochs=2",
        "trainer.params.accumulate_grad_batches=2",
        "trainer.params.limit_val_batches=0",
        "cache_sampler_mode=locality",
    ]:
        assert value in arguments
    assert arguments[-1] == "trainer.params.max_epochs=3"
