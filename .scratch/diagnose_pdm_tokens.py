#!/usr/bin/env python3
"""Inspect raw PDM components for a small set of anomalous tokens."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

sys.path.insert(0, str(Path(__file__).resolve().parent))

from recompute_latent_wam_csv_pdm import (  # noqa: E402
    _load_metric_cache,
    _load_navtest_tokens,
    _previous_tokens,
    _read_reference,
)
from navsim.agents.latent_wam.agent import LatentWAMAgent  # noqa: E402
from navsim.agents.latent_wam.config import LatentWAMConfig, VisionConfig  # noqa: E402
from navsim.common.dataclasses import SceneFilter, SensorConfig  # noqa: E402
from navsim.common.dataloader import SceneLoader  # noqa: E402
from navsim.evaluate.pdm_score import pdm_score  # noqa: E402
from navsim.planning.script.run_pdm_score_one_stage import (  # noqa: E402
    compute_final_scores,
    create_scene_aggregators,
    infer_start_adjacent_mapping,
)
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (  # noqa: E402
    PDMScorer,
    PDMScorerConfig,
)
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator  # noqa: E402
from navsim.traffic_agents_policies.log_replay_traffic_agents import LogReplayTrafficAgents  # noqa: E402


DEFAULT_TOKENS = [
    "8fb6a2364e0c53f9",
    "72c898c5c6a45939",
    "555af87f71415a3d",
    "deb455f53c805643",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--navsim-log-path", type=Path, required=True)
    parser.add_argument("--sensor-path", type=Path, required=True)
    parser.add_argument("--vision-model", type=Path, required=True)
    parser.add_argument("--metric-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", nargs="+", default=DEFAULT_TOKENS)
    parser.add_argument(
        "--human-penalty-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def _scalar(row: pd.Series, key: str) -> float | None:
    value = row[key]
    return None if pd.isna(value) else float(value)


def _serialise_raw_row(row: pd.Series) -> dict[str, object]:
    return {
        "pdm_score_raw": _scalar(row, "pdm_score"),
        "no_at_fault_collisions": _scalar(row, "no_at_fault_collisions"),
        "drivable_area_compliance": _scalar(row, "drivable_area_compliance"),
        "driving_direction_compliance": _scalar(row, "driving_direction_compliance"),
        "traffic_light_compliance": _scalar(row, "traffic_light_compliance"),
        "ego_progress": _scalar(row, "ego_progress"),
        "time_to_collision_within_bound": _scalar(row, "time_to_collision_within_bound"),
        "lane_keeping": _scalar(row, "lane_keeping"),
        "history_comfort": _scalar(row, "history_comfort"),
        "multiplicative_metrics_prod": _scalar(row, "multiplicative_metrics_prod"),
        "weighted_metrics": np.asarray(row["weighted_metrics"]).tolist(),
        "weighted_metrics_array": np.asarray(row["weighted_metrics_array"]).tolist(),
    }


def main() -> None:
    args = _parse_args()
    _, reference = _read_reference(args.csv, limit=10_000)
    navtest_tokens = set(_load_navtest_tokens())
    targets = list(dict.fromkeys(args.tokens))
    if any(token not in navtest_tokens for token in targets):
        raise ValueError("Every diagnostic token must be in the official navtest split")

    no_sensor_loader = SceneLoader(
        original_sensor_path=None,
        data_path=args.navsim_log_path,
        scene_filter=SceneFilter(
            num_history_frames=4,
            num_future_frames=10,
            frame_interval=1,
            has_route=True,
            tokens=list(navtest_tokens),
        ),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    previous = _previous_tokens(no_sensor_loader, targets)
    eval_tokens = list(dict.fromkeys(targets + list(previous.values())))

    config = LatentWAMConfig(vision=VisionConfig(model_id=str(args.vision_model)))
    agent = LatentWAMAgent(config=config, checkpoint_path=str(args.checkpoint))
    agent.initialize()
    agent.eval()
    model_loader = SceneLoader(
        original_sensor_path=args.sensor_path,
        data_path=args.navsim_log_path,
        scene_filter=SceneFilter(
            num_history_frames=4,
            num_future_frames=10,
            frame_interval=1,
            has_route=True,
            tokens=eval_tokens,
        ),
        sensor_config=agent.get_sensor_config(),
    )

    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    simulator = PDMSimulator(proposal_sampling=sampling)
    scorer = PDMScorer(
        proposal_sampling=sampling,
        config=PDMScorerConfig(human_penalty_filter=args.human_penalty_filter),
    )
    traffic_policy = LogReplayTrafficAgents(sampling)
    rows: list[pd.DataFrame] = []
    trajectories: dict[str, dict[str, object]] = {}

    for token in eval_tokens:
        metric_cache = _load_metric_cache(args.metric_cache, token)
        agent_input = model_loader.get_agent_input_from_token(token)
        with torch.inference_mode():
            trajectory = agent.compute_trajectory(agent_input)
            trajectory_repeat = agent.compute_trajectory(agent_input)
        trajectory_diff = float(np.max(np.abs(trajectory.poses - trajectory_repeat.poses)))
        if token in targets:
            trajectories[token] = {
                "trajectory_max_abs_repeat_diff": trajectory_diff,
                "trajectory_poses": trajectory.poses.tolist(),
                "metric_cache_log_name": metric_cache.log_name,
                "metric_cache_time_s": metric_cache.timepoint.time_s,
                "metric_cache_map_name": metric_cache.map_parameters.map_name,
            }

        score_row, ego_states = pdm_score(
            metric_cache=metric_cache,
            model_trajectory=trajectory,
            future_sampling=simulator.proposal_sampling,
            simulator=simulator,
            scorer=scorer,
            traffic_agents_policy=traffic_policy,
        )
        score_row["valid"] = True
        score_row["log_name"] = metric_cache.log_name
        score_row["frame_type"] = metric_cache.scene_type
        score_row["start_time"] = metric_cache.timepoint.time_s
        score_row["start_point_x"] = metric_cache.ego_state.rear_axle.x
        score_row["start_point_y"] = metric_cache.ego_state.rear_axle.y
        score_row["ego_simulated_states"] = [ego_states]
        score_row["token"] = token
        rows.append(score_row)

    score_df = pd.concat(rows, ignore_index=True)
    raw_rows = {row["token"].iloc[0]: _serialise_raw_row(row.iloc[0]) for _, row in score_df.groupby("token")}
    score_df = create_scene_aggregators(infer_start_adjacent_mapping(score_df), score_df, sampling)
    score_df = compute_final_scores(score_df).set_index("token")

    results = {}
    for token in targets:
        raw = raw_rows[token]
        final = score_df.loc[token]
        results[token] = {
            "reference": reference.get(token),
            "previous_token": previous.get(token),
            "raw": raw,
            "final": {
                "two_frame_extended_comfort": _scalar(final, "two_frame_extended_comfort"),
                "score": _scalar(final, "score"),
            },
            "trajectory": trajectories[token],
        }

    output = {
        "checkpoint": str(args.checkpoint.resolve()),
        "tokens": targets,
        "previous_tokens": previous,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
