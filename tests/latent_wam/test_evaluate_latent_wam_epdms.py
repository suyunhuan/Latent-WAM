import lzma
import pickle
from pathlib import Path

import pytest

from navsim.evaluate.latent_wam_epdms import EpdmsEvaluationConfig, _MetricCacheStore, load_scene_filter
from scripts.evaluate_latent_wam_epdms import parse_args


def test_load_navtest_scene_filter() -> None:
    scene_filter = load_scene_filter("navtest")

    assert scene_filter.num_history_frames == 4
    assert scene_filter.num_future_frames == 10
    assert scene_filter.has_route is True
    assert scene_filter.log_names


def test_evaluation_cli_parses_formal_and_smoke_options() -> None:
    args = parse_args(
        [
            "--checkpoint",
            "model.safetensors",
            "--navsim-log-path",
            "logs",
            "--sensor-path",
            "sensors",
            "--metric-cache",
            "metric-cache",
            "--output",
            "scores.csv",
            "--vision-model",
            "dinov2-base",
            "--max-scenarios",
            "100",
        ]
    )

    config = EpdmsEvaluationConfig(
        checkpoint_path=args.checkpoint,
        navsim_log_path=args.navsim_log_path,
        sensor_path=args.sensor_path,
        metric_cache_path=args.metric_cache,
        vision_model=args.vision_model,
        max_scenarios=args.max_scenarios,
    )
    assert config.checkpoint_path == Path("model.safetensors")
    assert config.max_scenarios == 100
    assert config.human_penalty_filter is True


def test_evaluation_cli_rejects_non_positive_smoke_limit() -> None:
    with pytest.raises(ValueError, match="positive"):
        parse_args(
            [
                "--checkpoint",
                "model.safetensors",
                "--navsim-log-path",
                "logs",
                "--sensor-path",
                "sensors",
                "--metric-cache",
                "metric-cache",
                "--output",
                "scores.csv",
                "--max-scenarios",
                "0",
            ]
        )


def test_legacy_metric_cache_layout_is_read_only_compatible(tmp_path: Path) -> None:
    cache_file = tmp_path / "log" / "unknown" / "token" / "metric_cache.pkl"
    cache_file.parent.mkdir(parents=True)
    with lzma.open(cache_file, "wb") as handle:
        pickle.dump({"token": "token"}, handle)

    store = _MetricCacheStore(tmp_path)

    assert store.tokens == ["token"]
    assert store.get_from_token("token") == {"token": "token"}
