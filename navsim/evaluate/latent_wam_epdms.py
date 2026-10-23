"""Public single-stage ePDMS evaluation for Latent WAM."""

from __future__ import annotations

import lzma
import pickle
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence

if TYPE_CHECKING:
    import pandas as pd

    from navsim.common.dataclasses import SceneFilter


@dataclass(frozen=True)
class EpdmsEvaluationConfig:
    """Configuration for a reproducible Latent WAM ePDMS evaluation."""

    checkpoint_path: Path
    navsim_log_path: Path
    sensor_path: Path
    metric_cache_path: Path
    vision_model: Optional[str] = None
    split: str = "navtest"
    max_scenarios: Optional[int] = None
    human_penalty_filter: bool = True
    tokens: Optional[Sequence[str]] = None
    # Inference mode selection; default matches the released command-index path.
    mode_select: str = "command"
    hybrid_margin: float = 0.0


class _MetricCacheStore:
    """Read standard NAVSIM caches and the legacy per-token cache layout."""

    def __init__(self, cache_path: Path) -> None:
        from navsim.common.dataloader import MetricCacheLoader

        metadata_path = cache_path / "metadata"
        metadata_files = list(metadata_path.glob("*.csv")) if metadata_path.is_dir() else []
        if metadata_files:
            self._standard_loader = MetricCacheLoader(cache_path)
            self._legacy_paths: dict[str, Path] = {}
            return

        legacy_paths = sorted(cache_path.rglob("metric_cache.pkl"))
        self._standard_loader = None
        self._legacy_paths = {path.parent.name: path for path in legacy_paths}
        if not self._legacy_paths:
            raise FileNotFoundError(
                f"No NAVSIM metric cache found under {cache_path}; expected metadata/*.csv "
                "or per-token metric_cache.pkl files"
            )

    @property
    def tokens(self) -> list[str]:
        """Return tokens available in the cache."""
        if self._standard_loader is not None:
            return self._standard_loader.tokens
        return list(self._legacy_paths)

    def get_from_token(self, token: str) -> Any:
        """Load one metric cache entry."""
        if self._standard_loader is not None:
            return self._standard_loader.get_from_token(token)
        with lzma.open(self._legacy_paths[token], "rb") as handle:
            return pickle.load(handle)


def load_scene_filter(split: str) -> SceneFilter:
    """Load a repository-provided NAVSIM scene filter by split name."""
    import yaml

    from navsim.common.dataclasses import SceneFilter

    filter_path = (
        Path(__file__).resolve().parents[1]
        / "planning"
        / "script"
        / "config"
        / "common"
        / "train_test_split"
        / "scene_filter"
        / f"{split}.yaml"
    )
    if not filter_path.is_file():
        raise FileNotFoundError(f"Unknown NAVSIM split {split!r}: {filter_path}")

    config = yaml.safe_load(filter_path.read_text(encoding="utf-8")) or {}
    scene_filter_fields = {field.name for field in fields(SceneFilter)}
    return SceneFilter(**{key: value for key, value in config.items() if key in scene_filter_fields})


def _find_previous_tokens(scene_loader: Any, target_tokens: Sequence[str], history_frames: int) -> dict[str, str]:
    """Find adjacent previous frames needed by two-frame extended comfort."""
    records: list[tuple[str, str, int]] = []
    for token, scene_dict in scene_loader.scene_frames_dicts.items():
        frame = scene_dict[history_frames - 1]
        records.append((token, frame["log_name"], int(frame["timestamp"])))

    records_by_log: dict[str, list[tuple[str, int]]] = {}
    for token, log_name, timestamp in records:
        records_by_log.setdefault(log_name, []).append((token, timestamp))

    target_set = set(target_tokens)
    previous_tokens: dict[str, str] = {}
    for log_records in records_by_log.values():
        log_records.sort(key=lambda item: item[1])
        for (previous_token, previous_time), (current_token, current_time) in zip(log_records, log_records[1:]):
            if current_token in target_set and 0 < (current_time - previous_time) / 1e6 <= 0.55:
                previous_tokens[current_token] = previous_token
    return previous_tokens


def _select_tokens(
    scene_loader: Any, metric_cache_loader: Any, config: EpdmsEvaluationConfig
) -> tuple[list[str], list[str]]:
    """Select target tokens and their optional previous frames."""
    cache_tokens = set(metric_cache_loader.tokens)
    available_tokens = [token for token in scene_loader.tokens if token in cache_tokens]

    if config.tokens is not None:
        requested_tokens = list(dict.fromkeys(str(token) for token in config.tokens))
        missing_tokens = [token for token in requested_tokens if token not in scene_loader.tokens]
        if missing_tokens:
            raise ValueError(f"Requested tokens are not present in split {config.split!r}: {missing_tokens[:5]}")
        missing_cache = [token for token in requested_tokens if token not in cache_tokens]
        if missing_cache:
            raise ValueError(f"Requested tokens have no metric cache: {missing_cache[:5]}")
        target_tokens = requested_tokens
    else:
        if config.max_scenarios is None:
            missing_cache = [token for token in scene_loader.tokens if token not in cache_tokens]
            if missing_cache:
                raise ValueError(
                    "Metric cache is incomplete for the requested split; "
                    f"missing {len(missing_cache)} tokens, including {missing_cache[:5]}"
                )
        target_tokens = available_tokens

    if config.max_scenarios is not None:
        if config.max_scenarios < 1:
            raise ValueError("max_scenarios must be positive")
        target_tokens = target_tokens[: config.max_scenarios]

    previous_tokens = _find_previous_tokens(
        scene_loader,
        target_tokens,
        history_frames=scene_loader._scene_filter.num_history_frames,
    )
    eval_tokens = list(
        dict.fromkeys(target_tokens + [token for token in previous_tokens.values() if token in cache_tokens])
    )
    if not target_tokens:
        raise ValueError("No tokens with metric cache are available for evaluation")
    return target_tokens, eval_tokens


def build_latent_wam_config(config: EpdmsEvaluationConfig):
    """Build LatentWAMConfig for evaluation, including inference mode_select."""
    from navsim.agents.latent_wam.config import LatentWAMConfig, RuntimeConfig, VisionConfig
    from navsim.agents.latent_wam.mode_select import resolve_mode_select

    runtime = RuntimeConfig(
        mode_select=resolve_mode_select(config.mode_select),
        hybrid_margin=float(config.hybrid_margin),
    )
    if config.vision_model:
        return LatentWAMConfig(vision=VisionConfig(model_id=config.vision_model), runtime=runtime)
    return LatentWAMConfig(runtime=runtime)


def evaluate_epdms(config: EpdmsEvaluationConfig) -> pd.DataFrame:
    """Evaluate Latent WAM and return a standard NAVSIM ePDMS DataFrame.

    Model inference is performed on the available CUDA device. PDM simulation,
    geometric scoring, and final pandas aggregation remain CPU operations in
    the current NAVSIM implementation.
    """
    import numpy as np
    import pandas as pd
    import torch
    from nuplan.common.actor_state.state_representation import StateSE2
    from nuplan.common.geometry.convert import relative_to_absolute_poses
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    from navsim.agents.latent_wam.agent import LatentWAMAgent
    from navsim.common.dataclasses import PDMResults, SensorConfig
    from navsim.common.dataloader import SceneLoader
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.planning.script.run_pdm_score_one_stage import (
        compute_final_scores,
        create_scene_aggregators,
        infer_start_adjacent_mapping,
    )
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.traffic_agents_policies.log_replay_traffic_agents import LogReplayTrafficAgents

    if not torch.cuda.is_available():
        raise RuntimeError("Latent WAM ePDMS evaluation requires a CUDA-compatible accelerator")

    scene_filter = load_scene_filter(config.split)
    scene_loader = SceneLoader(
        original_sensor_path=None,
        data_path=config.navsim_log_path,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_loader = _MetricCacheStore(config.metric_cache_path)
    target_tokens, eval_tokens = _select_tokens(scene_loader, metric_cache_loader, config)

    model_config = build_latent_wam_config(config)
    agent = LatentWAMAgent(config=model_config, checkpoint_path=str(config.checkpoint_path))
    agent.initialize()
    agent.eval()

    model_scene_loader = SceneLoader(
        original_sensor_path=config.sensor_path,
        data_path=config.navsim_log_path,
        scene_filter=replace(scene_filter, tokens=eval_tokens),
        sensor_config=agent.get_sensor_config(),
    )
    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    simulator = PDMSimulator(proposal_sampling=sampling)
    scorer = PDMScorer(
        proposal_sampling=sampling,
        config=PDMScorerConfig(human_penalty_filter=config.human_penalty_filter),
    )
    traffic_agents_policy = LogReplayTrafficAgents(sampling)

    score_rows: list[pd.DataFrame] = []
    for token in eval_tokens:
        metric_cache = metric_cache_loader.get_from_token(token)
        agent_input = model_scene_loader.get_agent_input_from_token(token)
        with torch.inference_mode():
            trajectory = agent.compute_trajectory(agent_input)

        score_row, ego_simulated_states = pdm_score(
            metric_cache=metric_cache,
            model_trajectory=trajectory,
            future_sampling=sampling,
            simulator=simulator,
            scorer=scorer,
            traffic_agents_policy=traffic_agents_policy,
        )
        score_row["valid"] = True
        score_row["log_name"] = metric_cache.log_name
        score_row["frame_type"] = metric_cache.scene_type
        score_row["start_time"] = metric_cache.timepoint.time_s
        end_pose = StateSE2(
            x=trajectory.poses[-1, 0],
            y=trajectory.poses[-1, 1],
            heading=trajectory.poses[-1, 2],
        )
        absolute_endpoint = relative_to_absolute_poses(metric_cache.ego_state.rear_axle, [end_pose])[0]
        score_row["endpoint_x"] = absolute_endpoint.x
        score_row["endpoint_y"] = absolute_endpoint.y
        score_row["start_point_x"] = metric_cache.ego_state.rear_axle.x
        score_row["start_point_y"] = metric_cache.ego_state.rear_axle.y
        score_row["ego_simulated_states"] = [ego_simulated_states]
        score_row["token"] = token
        score_rows.append(score_row)

    score_df = pd.concat(score_rows, ignore_index=True)
    score_df = create_scene_aggregators(infer_start_adjacent_mapping(score_df), score_df, sampling)
    score_df = compute_final_scores(score_df)
    score_df = score_df[score_df["token"].isin(target_tokens)].copy()

    score_columns = [
        column
        for column in score_df.columns
        if (
            (
                any(result_field.name in column for result_field in fields(PDMResults))
                or column in {"two_frame_extended_comfort", "score"}
            )
            and column != "pdm_score"
        )
    ]
    average_row = score_df[score_columns].mean(skipna=True)
    average_row["token"] = "average_all_frames"
    average_row["valid"] = bool(score_df["valid"].all())
    score_df = score_df[["token", "valid"] + score_columns]
    score_df.loc[len(score_df)] = average_row

    return score_df.reset_index(drop=True)
