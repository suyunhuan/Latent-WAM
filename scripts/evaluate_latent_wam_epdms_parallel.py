#!/usr/bin/env python3
"""Run Latent WAM ePDMS with accelerator inference and a CPU PDM pool."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import fields, replace
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import SceneLoader
from navsim.evaluate.latent_wam_epdms import (
    EpdmsEvaluationConfig,
    _MetricCacheStore,
    _select_tokens,
    load_scene_filter,
)

_PDM_WORKER_STATE: dict | None = None


def parse_devices(value: str) -> list[str]:
    """Parse a comma-separated list of physical CUDA/PPU device identifiers."""
    devices = [device.strip() for device in value.split(",") if device.strip()]
    if not devices:
        raise ValueError("--devices must contain at least one device")
    if len(devices) != len(set(devices)):
        raise ValueError("--devices must not contain duplicates")
    return devices


def partition_tokens(tokens: Sequence[str], num_partitions: int) -> list[list[str]]:
    """Split tokens into balanced contiguous shards for sensor-data locality."""
    if num_partitions < 1:
        raise ValueError("num_partitions must be positive")
    if not tokens:
        return []
    num_partitions = min(num_partitions, len(tokens))
    shard_size, remainder = divmod(len(tokens), num_partitions)
    shards: list[list[str]] = []
    begin = 0
    for index in range(num_partitions):
        end = begin + shard_size + (1 if index < remainder else 0)
        shards.append(list(tokens[begin:end]))
        begin = end
    return shards


def chunk_tokens(tokens: Sequence[str], chunk_size: int) -> list[list[str]]:
    """Create small tasks that a CPU process pool can schedule dynamically."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    return [
        list(tokens[begin : begin + chunk_size])
        for begin in range(0, len(tokens), chunk_size)
    ]


def available_cpu_ids() -> list[int]:
    """Return CPUs available to this process, respecting its cpuset."""
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return list(range(os.cpu_count() or 1))


def resolve_pdm_worker_count(requested: int, token_count: int, cpu_count: int) -> int:
    """Resolve zero-as-auto PDM worker configuration."""
    if requested < 0:
        raise ValueError("--pdm-workers must be non-negative")
    if token_count < 1:
        raise ValueError("token_count must be positive")
    if cpu_count < 1:
        raise ValueError("cpu_count must be positive")
    worker_count = min(8, cpu_count) if requested == 0 else requested
    return min(worker_count, token_count)


def cpu_group_for_worker(
    worker_identity: int, cpu_ids: Sequence[int], threads_per_worker: int
) -> tuple[int, ...]:
    """Assign a stable, compact CPU group to one spawned PDM worker."""
    if not cpu_ids:
        raise ValueError("cpu_ids must not be empty")
    if threads_per_worker < 1:
        raise ValueError("threads_per_worker must be positive")
    width = min(threads_per_worker, len(cpu_ids))
    begin = ((worker_identity - 1) * width) % len(cpu_ids)
    return tuple(cpu_ids[(begin + offset) % len(cpu_ids)] for offset in range(width))


def order_score_rows(
    score_rows: pd.DataFrame, expected_tokens: Sequence[str]
) -> pd.DataFrame:
    """Validate score-row completeness and restore a prescribed token order."""
    duplicated = score_rows.loc[score_rows["token"].duplicated(), "token"].tolist()
    if duplicated:
        raise ValueError(f"duplicate score tokens: {duplicated[:5]}")
    actual_tokens = set(score_rows["token"].astype(str))
    expected_set = set(expected_tokens)
    missing = [token for token in expected_tokens if token not in actual_tokens]
    unexpected = sorted(actual_tokens - expected_set)
    if missing or unexpected:
        raise ValueError(
            f"incomplete scores: missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    token_order = {token: index for index, token in enumerate(expected_tokens)}
    ordered = score_rows.copy()
    ordered["_token_order"] = ordered["token"].map(token_order)
    return (
        ordered.sort_values("_token_order")
        .drop(columns="_token_order")
        .reset_index(drop=True)
    )


def stack_feature_batch(feature_rows: Sequence[dict], device) -> dict:
    """Stack per-scene feature dictionaries and transfer one inference batch."""
    import torch

    if not feature_rows:
        raise ValueError("feature_rows must not be empty")
    expected_keys = set(feature_rows[0])
    if any(set(row) != expected_keys for row in feature_rows[1:]):
        raise ValueError("all feature rows must have identical keys")
    return {
        key: torch.stack([row[key] for row in feature_rows]).to(
            device, non_blocking=True
        )
        for key in feature_rows[0]
    }


def _latent_wam_model_config(config: EpdmsEvaluationConfig):
    """Build the exact model configuration shared by inference and PDM workers."""
    from navsim.agents.latent_wam.config import LatentWAMConfig, VisionConfig

    return (
        LatentWAMConfig(vision=VisionConfig(model_id=config.vision_model))
        if config.vision_model
        else LatentWAMConfig()
    )


def infer_trajectories_batched(
    config: EpdmsEvaluationConfig,
    tokens: Sequence[str],
    inference_batch_size: int,
) -> np.ndarray:
    """Infer exact token trajectories without running CPU PDM scoring."""
    import torch
    from tqdm import tqdm

    from navsim.agents.latent_wam.agent import LatentWAMAgent

    if inference_batch_size < 1:
        raise ValueError("inference_batch_size must be positive")
    if not tokens:
        raise ValueError("tokens must not be empty")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Latent WAM ePDMS evaluation requires a CUDA-compatible accelerator"
        )

    scene_filter = load_scene_filter(config.split)
    model_config = _latent_wam_model_config(config)
    agent = LatentWAMAgent(
        config=model_config, checkpoint_path=str(config.checkpoint_path)
    )
    agent.initialize()
    agent.eval()
    model_scene_loader = SceneLoader(
        original_sensor_path=config.sensor_path,
        data_path=config.navsim_log_path,
        scene_filter=replace(scene_filter, tokens=list(tokens)),
        sensor_config=agent.get_sensor_config(),
    )
    feature_builder = agent.get_feature_builders()[0]
    model_device = agent.model.scene_embeds.device
    batches: list[np.ndarray] = []
    for begin in tqdm(
        range(0, len(tokens), inference_batch_size),
        desc="Batched trajectory inference",
    ):
        batch_tokens = tokens[begin : begin + inference_batch_size]
        feature_rows = [
            feature_builder.compute_features(
                model_scene_loader.get_agent_input_from_token(token)
            )
            for token in batch_tokens
        ]
        features = stack_feature_batch(feature_rows, model_device)
        with torch.inference_mode():
            batches.append(agent.forward(features)["trajectory"].detach().cpu().numpy())

    trajectories = np.concatenate(batches, axis=0)
    if len(trajectories) != len(tokens):
        raise RuntimeError(
            f"trajectory count mismatch: expected {len(tokens)}, got {len(trajectories)}"
        )
    return trajectories


def write_numpy_atomic(path: Path, values: np.ndarray) -> None:
    """Write one NumPy array atomically without implicit suffix changes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    with temporary.open("wb") as file_handle:
        np.save(file_handle, values)
    os.replace(temporary, path)


def merge_trajectory_parts(
    part_paths: Sequence[Path],
    token_shards: Sequence[Sequence[str]],
    expected_tokens: Sequence[str],
    output_path: Path,
) -> Path:
    """Merge accelerator outputs into one token-ordered memory-mappable array."""
    if len(part_paths) != len(token_shards) or not part_paths:
        raise ValueError(
            "trajectory parts and token shards must be non-empty and aligned"
        )
    expected_order = {token: index for index, token in enumerate(expected_tokens)}
    if len(expected_order) != len(expected_tokens):
        raise ValueError("expected trajectory tokens must be unique")

    merged: np.ndarray | None = None
    seen: set[str] = set()
    for part_path, shard in zip(part_paths, token_shards):
        part = np.load(part_path, mmap_mode="r")
        if len(part) != len(shard):
            raise ValueError(
                f"trajectory part {part_path} has {len(part)} rows for {len(shard)} tokens"
            )
        if merged is None:
            merged = np.empty((len(expected_tokens), *part.shape[1:]), dtype=part.dtype)
        elif part.shape[1:] != merged.shape[1:]:
            raise ValueError(
                f"trajectory shape mismatch in {part_path}: {part.shape[1:]}"
            )
        for token, poses in zip(shard, part):
            if token in seen:
                raise ValueError(f"duplicate trajectory token: {token}")
            if token not in expected_order:
                raise ValueError(f"unexpected trajectory token: {token}")
            merged[expected_order[token]] = poses
            seen.add(token)

    missing = [token for token in expected_tokens if token not in seen]
    if missing:
        raise ValueError(f"missing trajectory tokens: {missing[:5]}")
    assert merged is not None
    write_numpy_atomic(output_path, merged)
    return output_path


def _initialize_pdm_worker(
    config: EpdmsEvaluationConfig,
    trajectory_file: Path,
    trajectory_tokens: Sequence[str],
    cpu_ids: Sequence[int],
    threads_per_worker: int,
) -> None:
    """Initialize process-local NAVSIM scoring state once per CPU worker."""
    global _PDM_WORKER_STATE

    identity = multiprocessing.current_process()._identity
    worker_identity = identity[0] if identity else 1
    cpu_group = cpu_group_for_worker(worker_identity, cpu_ids, threads_per_worker)
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(cpu_group))

    import torch
    from nuplan.planning.simulation.trajectory.trajectory_sampling import (
        TrajectorySampling,
    )

    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
        PDMScorer,
        PDMScorerConfig,
    )
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
        PDMSimulator,
    )
    from navsim.traffic_agents_policies.log_replay_traffic_agents import (
        LogReplayTrafficAgents,
    )

    torch.set_num_threads(threads_per_worker)
    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    _PDM_WORKER_STATE = {
        "cache": _MetricCacheStore(config.metric_cache_path),
        "trajectories": np.load(trajectory_file, mmap_mode="r"),
        "token_to_index": {
            token: index for index, token in enumerate(trajectory_tokens)
        },
        "sampling": sampling,
        "trajectory_sampling": _latent_wam_model_config(config).trajectory_sampling,
        "simulator": PDMSimulator(proposal_sampling=sampling),
        "scorer": PDMScorer(
            proposal_sampling=sampling,
            config=PDMScorerConfig(human_penalty_filter=config.human_penalty_filter),
        ),
        "traffic_agents_policy": LogReplayTrafficAgents(sampling),
        "cpu_group": cpu_group,
    }


def _score_token_chunk(tokens: Sequence[str]) -> pd.DataFrame:
    """Score one dynamic task using process-local PDM state."""
    if _PDM_WORKER_STATE is None:
        raise RuntimeError("PDM worker has not been initialized")

    from nuplan.common.actor_state.state_representation import StateSE2
    from nuplan.common.geometry.convert import relative_to_absolute_poses

    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score

    state = _PDM_WORKER_STATE
    rows: list[pd.DataFrame] = []
    for token in tokens:
        metric_cache = state["cache"].get_from_token(token)
        poses = np.asarray(state["trajectories"][state["token_to_index"][token]]).copy()
        trajectory = Trajectory(poses, state["trajectory_sampling"])
        score_row, ego_simulated_states = pdm_score(
            metric_cache=metric_cache,
            model_trajectory=trajectory,
            future_sampling=state["sampling"],
            simulator=state["simulator"],
            scorer=state["scorer"],
            traffic_agents_policy=state["traffic_agents_policy"],
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
        absolute_endpoint = relative_to_absolute_poses(
            metric_cache.ego_state.rear_axle, [end_pose]
        )[0]
        score_row["endpoint_x"] = absolute_endpoint.x
        score_row["endpoint_y"] = absolute_endpoint.y
        score_row["start_point_x"] = metric_cache.ego_state.rear_axle.x
        score_row["start_point_y"] = metric_cache.ego_state.rear_axle.y
        score_row["ego_simulated_states"] = [ego_simulated_states]
        score_row["token"] = token
        rows.append(score_row)
    return pd.concat(rows, ignore_index=True)


def run_pdm_pool(
    config: EpdmsEvaluationConfig,
    trajectory_file: Path,
    eval_tokens: Sequence[str],
    worker_count: int,
    chunk_size: int,
    threads_per_worker: int,
) -> pd.DataFrame:
    """Score trajectories through a dynamically scheduled CPU process pool."""
    from tqdm import tqdm

    token_chunks = chunk_tokens(eval_tokens, chunk_size)
    cpu_ids = available_cpu_ids()
    thread_variables = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "CUDA_VISIBLE_DEVICES",
    )
    previous_environment = {name: os.environ.get(name) for name in thread_variables}
    for name in thread_variables[:-1]:
        os.environ[name] = str(threads_per_worker)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    frames: list[pd.DataFrame] = []
    context = multiprocessing.get_context("spawn")
    try:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
            initializer=_initialize_pdm_worker,
            initargs=(
                config,
                trajectory_file,
                list(eval_tokens),
                cpu_ids,
                threads_per_worker,
            ),
        ) as executor:
            futures = {
                executor.submit(_score_token_chunk, token_chunk): len(token_chunk)
                for token_chunk in token_chunks
            }
            with tqdm(total=len(eval_tokens), desc="Parallel PDM scoring") as progress:
                for future in as_completed(futures):
                    frames.append(future.result())
                    progress.update(futures[future])
    finally:
        for name, value in previous_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    return order_score_rows(pd.concat(frames, ignore_index=True), eval_tokens)


def finalize_scores(
    raw_score_rows: pd.DataFrame,
    eval_tokens: Sequence[str],
    target_tokens: Sequence[str],
) -> pd.DataFrame:
    """Apply global adjacent-frame aggregation and produce the standard CSV schema."""
    from navsim.common.dataclasses import PDMResults
    from navsim.planning.script.run_pdm_score_one_stage import (
        compute_final_scores,
        create_scene_aggregators,
        infer_start_adjacent_mapping,
    )

    score_df = order_score_rows(raw_score_rows, eval_tokens)
    score_df = create_scene_aggregators(
        infer_start_adjacent_mapping(score_df), score_df, _trajectory_sampling()
    )
    score_df = compute_final_scores(score_df)
    score_df = score_df[score_df["token"].isin(target_tokens)].copy()
    score_df = order_score_rows(score_df, target_tokens)
    score_columns = [
        column
        for column in score_df.columns
        if (
            (
                any(result_field.name in column for result_field in fields(PDMResults))
                or column == "two_frame_extended_comfort"
            )
            and column != "pdm_score"
        )
    ]
    if "score" in score_df.columns:
        score_columns.append("score")
    average_row = score_df[score_columns].mean(skipna=True)
    average_row["token"] = "average_all_frames"
    average_row["valid"] = bool(score_df["valid"].all())
    score_df = score_df[["token", "valid"] + score_columns]
    score_df.loc[len(score_df)] = average_row
    return score_df.reset_index(drop=True)


def _trajectory_sampling():
    from nuplan.planning.simulation.trajectory.trajectory_sampling import (
        TrajectorySampling,
    )

    return TrajectorySampling(num_poses=40, interval_length=0.1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse coordinator and internal inference-worker arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--navsim-log-path", type=Path, required=True)
    parser.add_argument("--sensor-path", type=Path, required=True)
    parser.add_argument("--metric-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vision-model", default=None)
    parser.add_argument("--split", default="navtest")
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument(
        "--devices",
        default="0",
        help="Comma-separated physical device IDs (default: 0).",
    )
    parser.add_argument(
        "--inference-workers-per-device",
        "--workers-per-device",
        dest="inference_workers_per_device",
        type=int,
        default=1,
        help="Accelerator inference processes per device (default: 1).",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=8,
        help="Trajectory inference batch size in each accelerator worker (default: 8).",
    )
    parser.add_argument(
        "--pdm-workers",
        type=int,
        default=0,
        help="CPU PDM workers; 0 selects min(8, available CPUs) (default: 0).",
    )
    parser.add_argument(
        "--pdm-chunk-size",
        type=int,
        default=64,
        help="Tokens per dynamically scheduled PDM task (default: 64).",
    )
    parser.add_argument(
        "--pdm-threads-per-worker",
        type=int,
        default=1,
        help="CPU threads and affinity width per PDM worker (default: 1).",
    )
    parser.add_argument(
        "--no-human-penalty-filter",
        action="store_false",
        dest="human_penalty_filter",
        help="Disable the NAVSIM human-penalty exemption.",
    )
    parser.add_argument(
        "--worker-token-file", type=Path, default=None, help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    args.devices = parse_devices(args.devices)
    if args.max_scenarios is not None and args.max_scenarios < 1:
        raise ValueError("--max-scenarios must be positive")
    if args.inference_workers_per_device < 1:
        raise ValueError("--inference-workers-per-device must be positive")
    if args.inference_batch_size < 1:
        raise ValueError("--inference-batch-size must be positive")
    if args.pdm_workers < 0:
        raise ValueError("--pdm-workers must be non-negative")
    if args.pdm_chunk_size < 1:
        raise ValueError("--pdm-chunk-size must be positive")
    if args.pdm_threads_per_worker < 1:
        raise ValueError("--pdm-threads-per-worker must be positive")
    return args


def _evaluation_config(args: argparse.Namespace) -> EpdmsEvaluationConfig:
    return EpdmsEvaluationConfig(
        checkpoint_path=args.checkpoint,
        navsim_log_path=args.navsim_log_path,
        sensor_path=args.sensor_path,
        metric_cache_path=args.metric_cache,
        vision_model=args.vision_model,
        split=args.split,
        max_scenarios=args.max_scenarios,
        human_penalty_filter=args.human_penalty_filter,
    )


def _select_evaluation_tokens(
    args: argparse.Namespace,
) -> tuple[list[str], list[str]]:
    scene_filter = load_scene_filter(args.split)
    scene_loader = SceneLoader(
        original_sensor_path=None,
        data_path=args.navsim_log_path,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_loader = _MetricCacheStore(args.metric_cache)
    return _select_tokens(scene_loader, metric_cache_loader, _evaluation_config(args))


def _run_inference_worker(args: argparse.Namespace) -> None:
    tokens = json.loads(args.worker_token_file.read_text(encoding="utf-8"))
    if not isinstance(tokens, list) or not all(
        isinstance(token, str) for token in tokens
    ):
        raise ValueError(f"invalid worker token file: {args.worker_token_file}")
    trajectories = infer_trajectories_batched(
        _evaluation_config(args), tokens, args.inference_batch_size
    )
    write_numpy_atomic(args.output, trajectories)
    print(
        json.dumps({"output": str(args.output), "trajectories": len(tokens)}, indent=2),
        flush=True,
    )


def _inference_worker_command(
    args: argparse.Namespace,
    token_file: Path,
    trajectory_output: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--checkpoint",
        str(args.checkpoint),
        "--navsim-log-path",
        str(args.navsim_log_path),
        "--sensor-path",
        str(args.sensor_path),
        "--metric-cache",
        str(args.metric_cache),
        "--output",
        str(trajectory_output),
        "--split",
        args.split,
        "--devices",
        "0",
        "--worker-token-file",
        str(token_file),
        "--inference-batch-size",
        str(args.inference_batch_size),
    ]
    if args.vision_model is not None:
        command.extend(["--vision-model", str(args.vision_model)])
    if not args.human_penalty_filter:
        command.append("--no-human-penalty-filter")
    return command


def _read_log_tail(path: Path, max_lines: int = 30) -> str:
    try:
        return "\n".join(
            path.read_text(encoding="utf-8", errors="replace").splitlines()[-max_lines:]
        )
    except OSError:
        return f"<unable to read worker log {path}>"


def _run_inference_processes(
    args: argparse.Namespace,
    token_shards: Sequence[Sequence[str]],
    device_assignments: Sequence[str],
    run_dir: Path,
) -> list[Path]:
    processes: list[subprocess.Popen[bytes]] = []
    log_handles = []
    trajectory_parts: list[Path] = []
    try:
        for index, (device, shard) in enumerate(zip(device_assignments, token_shards)):
            token_file = run_dir / f"inference-{index:03d}.tokens.json"
            trajectory_output = run_dir / f"inference-{index:03d}.npy"
            log_path = run_dir / f"inference-{index:03d}.log"
            token_file.write_text(json.dumps(shard) + "\n", encoding="utf-8")
            log_handle = log_path.open("wb")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = device
            process = subprocess.Popen(
                _inference_worker_command(args, token_file, trajectory_output),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=environment,
            )
            processes.append(process)
            log_handles.append(log_handle)
            trajectory_parts.append(trajectory_output)
            print(
                f"inference_worker={index} pid={process.pid} device={device} "
                f"scenarios={len(shard)} log={log_path}",
                flush=True,
            )

        failures = []
        for index, process in enumerate(processes):
            return_code = process.wait()
            log_handles[index].close()
            print(
                f"inference_worker={index} pid={process.pid} exit={return_code}",
                flush=True,
            )
            if return_code != 0:
                failures.append(
                    (index, return_code, run_dir / f"inference-{index:03d}.log")
                )
        if failures:
            details = "\n\n".join(
                f"inference worker {index} exited {return_code}:\n{_read_log_tail(log_path)}"
                for index, return_code, log_path in failures
            )
            raise RuntimeError(
                f"parallel trajectory inference failed; artifacts retained at {run_dir}\n{details}"
            )
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for handle in log_handles:
            if not handle.closed:
                handle.close()
    return trajectory_parts


def _run_coordinator(args: argparse.Namespace) -> None:
    started_at = time.monotonic()
    target_tokens, eval_tokens = _select_evaluation_tokens(args)
    inference_slots = len(args.devices) * args.inference_workers_per_device
    token_shards = partition_tokens(eval_tokens, inference_slots)
    device_assignments = [
        args.devices[index % len(args.devices)] for index in range(len(token_shards))
    ]
    cpu_ids = available_cpu_ids()
    pdm_worker_count = resolve_pdm_worker_count(
        args.pdm_workers, len(eval_tokens), len(cpu_ids)
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.output.parent / f"{args.output.stem}_parts_{timestamp}_{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    print(
        f"Launching {len(token_shards)} inference workers for {len(eval_tokens)} trajectories "
        f"on devices {device_assignments}; pdm_workers={pdm_worker_count}; parts={run_dir}",
        flush=True,
    )

    trajectory_parts = _run_inference_processes(
        args, token_shards, device_assignments, run_dir
    )
    trajectory_file = merge_trajectory_parts(
        trajectory_parts,
        token_shards,
        eval_tokens,
        run_dir / "trajectories.npy",
    )
    raw_scores = run_pdm_pool(
        _evaluation_config(args),
        trajectory_file,
        eval_tokens,
        pdm_worker_count,
        args.pdm_chunk_size,
        args.pdm_threads_per_worker,
    )
    merged = finalize_scores(raw_scores, eval_tokens, target_tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.parent / f".{args.output.name}.{os.getpid()}.tmp"
    merged.to_csv(temporary_output, index=False)
    os.replace(temporary_output, args.output)
    average_row = merged.iloc[-1]
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "parts": str(run_dir.resolve()),
                "inference_workers": len(token_shards),
                "devices": device_assignments,
                "pdm_workers": pdm_worker_count,
                "pdm_chunk_size": args.pdm_chunk_size,
                "scenarios": len(target_tokens),
                "valid": bool(merged.iloc[:-1]["valid"].all()),
                "epdms": float(average_row["score"]),
                "elapsed_seconds": time.monotonic() - started_at,
            },
            indent=2,
        ),
        flush=True,
    )


def main(argv: list[str] | None = None) -> None:
    """Run one inference shard or coordinate the two-stage evaluation."""
    args = parse_args(argv)
    if args.worker_token_file is not None:
        _run_inference_worker(args)
    else:
        _run_coordinator(args)


if __name__ == "__main__":
    main()
