#!/usr/bin/env python3
"""Replay the first CSV tokens through the current Latent WAM inference path."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch

from navsim.agents.latent_wam.agent import LatentWAMAgent
from navsim.agents.latent_wam.config import LatentWAMConfig, VisionConfig
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--navsim-log-path", type=Path, required=True)
    parser.add_argument("--sensor-path", type=Path, required=True)
    parser.add_argument("--vision-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    return parser.parse_args()


def _read_tokens(path: Path, limit: int) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, skipinitialspace=True))
    if not rows:
        raise ValueError(f"CSV contains no rows: {path}")
    token_key = next((key for key in rows[0] if key.strip() == "token"), None)
    if token_key is None:
        raise ValueError(f"CSV has no token column: {path}")
    tokens = [str(row[token_key]).strip() for row in rows[:limit]]
    if any(not token for token in tokens):
        raise ValueError("CSV contains an empty token in the requested prefix")
    if len(set(tokens)) != len(tokens):
        raise ValueError("CSV contains duplicate tokens in the requested prefix")
    return tokens


def _error_record(error: BaseException) -> dict[str, Any]:
    return {"status": "error", "error_type": type(error).__name__, "error": str(error)}


def main() -> None:
    args = _arguments()
    if args.limit < 1:
        raise ValueError("--limit must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("token replay requires a CUDA-compatible accelerator")

    tokens = _read_tokens(args.csv, args.limit)
    config = LatentWAMConfig(vision=VisionConfig(model_id=str(args.vision_model)))
    agent = LatentWAMAgent(config=config, checkpoint_path=str(args.checkpoint))
    agent.initialize()
    agent.eval()

    scene_filter = SceneFilter(
        num_history_frames=4,
        num_future_frames=10,
        # Match the released navtest scene filter. The CSV contains every
        # scene-start token from that split, not a stride-14 subset.
        frame_interval=1,
        has_route=True,
        tokens=tokens,
    )
    scene_loader = SceneLoader(
        original_sensor_path=args.sensor_path,
        data_path=args.navsim_log_path,
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    available_tokens = set(scene_loader.tokens)
    results: list[dict[str, Any]] = []
    for index, token in enumerate(tokens):
        if token not in available_tokens:
            results.append({"token": token, "status": "missing_from_scene_loader"})
            continue
        start = time.perf_counter()
        try:
            agent_input = scene_loader.get_agent_input_from_token(token)
            with torch.inference_mode():
                trajectory = agent.compute_trajectory(agent_input)
            poses = trajectory.poses
            if poses.shape != (8, 3):
                raise RuntimeError(f"unexpected trajectory shape: {poses.shape}")
            if not torch.isfinite(torch.as_tensor(poses)).all():
                raise RuntimeError("trajectory contains non-finite values")
            results.append(
                {
                    "token": token,
                    "status": "ok",
                    "trajectory_shape": list(poses.shape),
                    "trajectory_finite": True,
                    "elapsed_seconds": time.perf_counter() - start,
                }
            )
        except Exception as error:  # noqa: BLE001 - preserve per-token replay failures
            results.append(
                {
                    "token": token,
                    "elapsed_seconds": time.perf_counter() - start,
                    **_error_record(error),
                }
            )
        print(f"replay={index + 1}/{len(tokens)} token={token} status={results[-1]['status']}", flush=True)

    summary = {
        "csv": str(args.csv.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "requested_tokens": len(tokens),
        "scene_loader_tokens": len(available_tokens),
        "ok": sum(result["status"] == "ok" for result in results),
        "missing_from_scene_loader": sum(result["status"] == "missing_from_scene_loader" for result in results),
        "errors": sum(result["status"] == "error" for result in results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))
    if summary["ok"] != len(tokens):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
