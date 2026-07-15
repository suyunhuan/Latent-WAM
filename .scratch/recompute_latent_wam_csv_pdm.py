#!/usr/bin/env python3
"""Recompute current Latent WAM PDM scores for a CSV prefix."""

from __future__ import annotations

import argparse
import csv
import json
import lzma
import os
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.latent_wam.agent import LatentWAMAgent
from navsim.agents.latent_wam.config import LatentWAMConfig, VisionConfig
from navsim.common.dataclasses import Scene, SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
from navsim.planning.script.run_pdm_score_one_stage import (
    compute_final_scores,
    create_scene_aggregators,
    infer_start_adjacent_mapping,
)
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
    PDMScorer,
    PDMScorerConfig,
)
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.traffic_agents_policies.log_replay_traffic_agents import LogReplayTrafficAgents


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--navsim-log-path", type=Path, required=True)
    parser.add_argument("--sensor-path", type=Path, required=True)
    parser.add_argument("--vision-model", type=Path, required=True)
    parser.add_argument("--metric-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    return parser.parse_args()


def _read_reference(path: Path, limit: int) -> tuple[list[str], dict[str, dict[str, Any]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, skipinitialspace=True))[:limit]
    if not rows:
        raise ValueError(f"CSV contains no rows: {path}")
    token_key = next((key for key in rows[0] if key.strip() == "token"), None)
    score_key = next((key for key in rows[0] if key.strip() == "score"), None)
    if token_key is None or score_key is None:
        raise ValueError("CSV must contain token and score columns")
    reference: dict[str, dict[str, Any]] = {}
    for row in rows:
        token = str(row[token_key]).strip()
        reference[token] = {
            "reference_score": float(row[score_key]),
            "reference_two_frame_extended_comfort": float(
                row[next(key for key in row if key.strip() == "two_frame_extended_comfort")]
            )
            if row[next(key for key in row if key.strip() == "two_frame_extended_comfort")].strip()
            else None,
        }
    return list(reference), reference


def _previous_tokens(scene_loader: SceneLoader, targets: list[str]) -> dict[str, str]:
    records: list[tuple[str, str, int]] = []
    for token, scene_dict in scene_loader.scene_frames_dicts.items():
        frame = scene_dict[scene_loader._scene_filter.num_history_frames - 1]
        records.append((token, frame["log_name"], int(frame["timestamp"])))

    by_log: dict[str, list[tuple[str, int]]] = {}
    for token, log_name, timestamp in records:
        by_log.setdefault(log_name, []).append((token, timestamp))

    previous: dict[str, str] = {}
    target_set = set(targets)
    for log_records in by_log.values():
        log_records.sort(key=lambda item: item[1])
        for (previous_token, previous_time), (current_token, current_time) in zip(
            log_records, log_records[1:]
        ):
            if current_token in target_set and 0 < (current_time - previous_time) / 1e6 <= 0.55:
                previous[current_token] = previous_token
    return previous


def _load_navtest_tokens() -> list[str]:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "navsim"
        / "planning"
        / "script"
        / "config"
        / "common"
        / "train_test_split"
        / "scene_filter"
        / "navtest.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return [str(token) for token in config["tokens"]]


def _build_metric_cache(
    scene_loader: SceneLoader,
    tokens: list[str],
    cache_path: Path,
) -> None:
    cache_path.mkdir(parents=True, exist_ok=True)
    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    processor = MetricCacheProcessor(
        cache_path=str(cache_path),
        force_feature_computation=False,
        proposal_sampling=sampling,
    )
    sensor_config = SensorConfig.build_no_sensors()
    for index, token in enumerate(tokens, start=1):
        scene_dict = scene_loader.scene_frames_dicts[token]
        scene = Scene.from_scene_dict_list(
            scene_dict,
            None,
            num_history_frames=scene_loader._scene_filter.num_history_frames,
            num_future_frames=scene_loader._scene_filter.num_future_frames,
            sensor_config=sensor_config,
        )
        scenario = NavSimScenario(
            scene,
            map_root=os.environ["NUPLAN_MAPS_ROOT"],
            map_version="nuplan-maps-v1.0",
        )
        processor.compute_and_save_metric_cache(scenario)
        print(f"metric_cache={index}/{len(tokens)} token={token}", flush=True)


def _load_metric_cache(cache_path: Path, token: str) -> Any:
    candidates = list(cache_path.glob(f"*/*/{token}/metric_cache.pkl"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one metric cache for token {token}, found {candidates}")
    with lzma.open(candidates[0], "rb") as handle:
        return pickle.load(handle)


def main() -> None:
    args = _parse_args()
    if args.limit < 1:
        raise ValueError("--limit must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("PDM replay requires a CUDA-compatible accelerator")

    targets, reference = _read_reference(args.csv, args.limit)
    no_sensor_filter = SceneFilter(
        num_history_frames=4,
        num_future_frames=10,
        frame_interval=1,
        has_route=True,
        tokens=_load_navtest_tokens(),
    )
    all_scene_loader = SceneLoader(
        original_sensor_path=None,
        data_path=args.navsim_log_path,
        scene_filter=no_sensor_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    available = set(all_scene_loader.scene_frames_dicts)
    missing = [token for token in targets if token not in available]
    if missing:
        raise RuntimeError(f"Missing target tokens from SceneLoader: {missing[:5]}")

    previous = _previous_tokens(all_scene_loader, targets)
    eval_tokens = list(dict.fromkeys(targets + list(previous.values())))
    print(
        json.dumps(
            {
                "targets": len(targets),
                "previous_tokens": len(previous),
                "tokens_to_recompute": len(eval_tokens),
            }
        ),
        flush=True,
    )
    _build_metric_cache(all_scene_loader, eval_tokens, args.metric_cache)

    config = LatentWAMConfig(vision=VisionConfig(model_id=str(args.vision_model)))
    agent = LatentWAMAgent(config=config, checkpoint_path=str(args.checkpoint))
    agent.initialize()
    agent.eval()
    model_scene_loader = SceneLoader(
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
        config=PDMScorerConfig(human_penalty_filter=True),
    )
    traffic_agents_policy = LogReplayTrafficAgents(sampling)

    rows: list[pd.DataFrame] = []
    for index, token in enumerate(eval_tokens, start=1):
        start = time.perf_counter()
        metric_cache = _load_metric_cache(args.metric_cache, token)
        agent_input = model_scene_loader.get_agent_input_from_token(token)
        with torch.inference_mode():
            trajectory = agent.compute_trajectory(agent_input)
        score_row, ego_simulated_states = pdm_score(
            metric_cache=metric_cache,
            model_trajectory=trajectory,
            future_sampling=simulator.proposal_sampling,
            simulator=simulator,
            scorer=scorer,
            traffic_agents_policy=traffic_agents_policy,
        )
        score_row["valid"] = True
        score_row["log_name"] = metric_cache.log_name
        score_row["frame_type"] = metric_cache.scene_type
        score_row["start_time"] = metric_cache.timepoint.time_s
        score_row["endpoint_x"] = np.nan
        score_row["endpoint_y"] = np.nan
        score_row["start_point_x"] = metric_cache.ego_state.rear_axle.x
        score_row["start_point_y"] = metric_cache.ego_state.rear_axle.y
        score_row["ego_simulated_states"] = [ego_simulated_states]
        score_row["token"] = token
        rows.append(score_row)
        print(f"pdm_score={index}/{len(eval_tokens)} token={token} elapsed={time.perf_counter() - start:.3f}s", flush=True)

    score_df = pd.concat(rows, ignore_index=True)
    score_df = create_scene_aggregators(
        infer_start_adjacent_mapping(score_df), score_df, sampling
    )
    score_df = compute_final_scores(score_df)
    score_df = score_df.set_index("token")
    result_rows = []
    for token in targets:
        row = score_df.loc[token]
        recomputed_score = float(row["score"])
        reference_score = reference[token]["reference_score"]
        result_rows.append(
            {
                "token": token,
                "recomputed_score": recomputed_score,
                "reference_score": reference_score,
                "absolute_diff": abs(recomputed_score - reference_score),
                "recomputed_two_frame_extended_comfort": (
                    None
                    if pd.isna(row["two_frame_extended_comfort"])
                    else float(row["two_frame_extended_comfort"])
                ),
                "reference_two_frame_extended_comfort": reference[token][
                    "reference_two_frame_extended_comfort"
                ],
            }
        )
    diffs = np.asarray([row["absolute_diff"] for row in result_rows], dtype=np.float64)
    output = {
        "csv": str(args.csv.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "metric_cache": str(args.metric_cache.resolve()),
        "requested_tokens": len(targets),
        "recomputed_tokens": len(result_rows),
        "previous_tokens_used": len(previous),
        "mean_recomputed_score": float(np.mean([row["recomputed_score"] for row in result_rows])),
        "mean_reference_score": float(np.mean([row["reference_score"] for row in result_rows])),
        "mean_absolute_diff": float(np.mean(diffs)),
        "max_absolute_diff": float(np.max(diffs)),
        "matching_at_1e-5": int(np.sum(diffs <= 1e-5)),
        "results": result_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
