import os
import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPOSITORY_ROOT / "scripts/evaluation/run_latent_wam_epdms_parallel.sh"


def _base_environment(tmp_path: Path) -> dict[str, str]:
    capture_path = tmp_path / "evaluation-args.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURE_PATH"\n', encoding="utf-8")
    fake_python.chmod(0o755)
    environment = os.environ.copy()
    for name in (
        "NAVSIM_DEVKIT_ROOT",
        "NAVSIM_EXP_ROOT",
        "OPENSCENE_DATA_ROOT",
        "LATENT_WAM_CHECKPOINT",
        "METRIC_CACHE_PATH",
        "EVALUATION_OUTPUT",
        "EVALUATION_DEVICES",
        "INFERENCE_WORKERS_PER_DEVICE",
        "INFERENCE_BATCH_SIZE",
        "PDM_WORKERS",
        "PDM_CHUNK_SIZE",
        "PDM_THREADS_PER_WORKER",
        "DINOV2_PATH",
        "PYTHON_BIN",
    ):
        environment.pop(name, None)
    environment.update(
        {
            "PYTHON_BIN": str(fake_python),
            "CAPTURE_PATH": str(capture_path),
            "NAVSIM_EXP_ROOT": str(tmp_path / "exp"),
            "OPENSCENE_DATA_ROOT": str(tmp_path / "dataset"),
            "LATENT_WAM_CHECKPOINT": str(tmp_path / "model.safetensors"),
        }
    )
    return environment


def test_parallel_evaluation_launcher_maps_environment_and_derives_workspace(tmp_path: Path) -> None:
    environment = _base_environment(tmp_path)
    environment.update(
        {
            "DINOV2_PATH": str(tmp_path / "dinov2-base"),
            "EVALUATION_DEVICES": "0,2",
            "INFERENCE_WORKERS_PER_DEVICE": "2",
            "INFERENCE_BATCH_SIZE": "16",
            "PDM_WORKERS": "6",
        }
    )

    subprocess.run(
        ["bash", str(LAUNCHER), "--max-scenarios", "100"],
        cwd=tmp_path,
        env=environment,
        check=True,
    )

    arguments = (tmp_path / "evaluation-args.txt").read_text(encoding="utf-8").splitlines()
    assert arguments[0] == str(REPOSITORY_ROOT / "scripts/evaluate_latent_wam_epdms_parallel.py")
    assert arguments[arguments.index("--checkpoint") + 1] == str(tmp_path / "model.safetensors")
    assert arguments[arguments.index("--navsim-log-path") + 1] == str(tmp_path / "dataset/navsim_logs/test")
    assert arguments[arguments.index("--sensor-path") + 1] == str(tmp_path / "dataset/sensor_blobs/test")
    assert arguments[arguments.index("--metric-cache") + 1] == str(tmp_path / "exp/metric_cache")
    assert arguments[arguments.index("--output") + 1] == str(tmp_path / "exp/latent_wam_eval/navtest.csv")
    assert arguments[arguments.index("--devices") + 1] == "0,2"
    assert arguments[arguments.index("--inference-workers-per-device") + 1] == "2"
    assert arguments[arguments.index("--inference-batch-size") + 1] == "16"
    assert arguments[arguments.index("--pdm-workers") + 1] == "6"
    assert arguments[arguments.index("--vision-model") + 1] == str(tmp_path / "dinov2-base")
    assert arguments[-2:] == ["--max-scenarios", "100"]


def test_parallel_evaluation_launcher_requires_checkpoint(tmp_path: Path) -> None:
    environment = _base_environment(tmp_path)
    environment.pop("LATENT_WAM_CHECKPOINT")

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "LATENT_WAM_CHECKPOINT" in result.stderr
