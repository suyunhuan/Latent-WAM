#!/usr/bin/env python3
"""Benchmark Latent WAM cache loading and full optimization steps."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Dict, Iterator, Sequence, Tuple

import psutil
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from navsim.agents.latent_wam.agent import LatentWAMAgent
from navsim.agents.latent_wam.config import LatentWAMConfig, VisionConfig
from navsim.agents.latent_wam.dataset import LatentWAMCacheDataset
from navsim.agents.latent_wam.sampler import LocalityAwareDistributedSampler

Batch = Tuple[Dict[str, object], Dict[str, torch.Tensor]]


class _TrainingStepModule(nn.Module):
    """Expose the training loss through ``forward`` so DDP tracks the graph."""

    def __init__(self, agent: LatentWAMAgent) -> None:
        super().__init__()
        self.model = agent.model

    def forward(self, features: Dict[str, object], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        predictions = self.model.forward_train(features)
        losses = self.model.compute_loss(targets, predictions)
        return torch.stack(tuple(losses.values())).sum()


class RepeatedDataset(Dataset[Batch]):
    """Repeat a non-empty cache dataset to a requested benchmark length."""

    def __init__(
        self,
        dataset: Dataset[Batch],
        length: int,
        sample_locality: Sequence[int] | None = None,
    ) -> None:
        if len(dataset) == 0:
            raise ValueError("benchmark cache dataset must not be empty")
        if length < 1:
            raise ValueError("benchmark length must be positive")
        self.dataset = dataset
        self.length = length
        self.sample_locality = None if sample_locality is None else [
            int(sample_locality[index % len(sample_locality)]) for index in range(length)
        ]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Batch:
        return self.dataset[index % len(self.dataset)]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("data", "train"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--vision-model", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--measure-steps", type=int, default=100)
    parser.add_argument("--log-every-steps", type=int, default=0)
    parser.add_argument("--sampler-mode", choices=("sequential", "locality"), default="sequential")
    parser.add_argument("--precision", choices=("fp32", "bf16-mixed"), default="fp32")
    parser.add_argument(
        "--sample-token",
        help="Repeat one real cache sample with this token instead of walking the full cache.",
    )
    return parser.parse_args()


def select_sample(dataset: Dataset[Batch], token: str) -> Dataset[Batch]:
    """Select one cache sample by token without loading every tensor payload."""
    samples = getattr(dataset, "samples", None)
    if samples is None:
        raise ValueError("cache dataset does not expose sample metadata")
    for index, sample in enumerate(samples):
        if isinstance(sample, dict):
            sample_token = sample.get("token")
        elif isinstance(sample, tuple) and sample:
            sample_token = Path(sample[0]).parent.name
        else:
            sample_token = None
        if sample_token == token:
            return Subset(dataset, [index])
    raise ValueError(f"sample token is not present in cache: {token}")


def _sample_locality(dataset: Dataset[Batch]) -> Sequence[int] | None:
    """Return sample-to-shard locality metadata through simple dataset wrappers."""
    locality = getattr(dataset, "sample_locality", None)
    if locality is not None:
        return locality
    if isinstance(dataset, Subset):
        parent_locality = _sample_locality(dataset.dataset)
        if parent_locality is not None:
            return [parent_locality[index] for index in dataset.indices]
    return None


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty duration sequence")
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_seconds(values: Sequence[float]) -> Dict[str, float]:
    """Summarize benchmark durations without external statistics packages."""
    return {
        "mean": statistics.fmean(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "min": min(values),
        "max": max(values),
    }


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _next_timed(iterator: Iterator[Batch]) -> Tuple[Batch, float]:
    start = time.perf_counter()
    batch = next(iterator)
    return batch, time.perf_counter() - start


def _build_loader(args: argparse.Namespace) -> DataLoader[Batch]:
    source = LatentWAMCacheDataset(args.cache_path)
    if args.sample_token:
        source = select_sample(source, args.sample_token)
    total_samples = (args.warmup_steps + args.measure_steps) * args.batch_size
    if args.sampler_mode == "locality" and args.sample_token:
        raise ValueError("locality sampler mode cannot be combined with --sample-token")
    source_locality = _sample_locality(source)
    if args.sampler_mode == "locality":
        if source_locality is None:
            raise ValueError("locality sampler mode requires sample_locality metadata")
        dataset: Dataset[Batch] = source
    else:
        dataset = RepeatedDataset(source, total_samples, sample_locality=source_locality)
    parameters: Dict[str, object] = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "shuffle": False,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.sampler_mode == "locality":
        parameters["sampler"] = LocalityAwareDistributedSampler(
            source_locality,
            mode="locality",
            shuffle=True,
            seed=0,
            rank=int(os.environ.get("RANK", "0")),
            world_size=int(os.environ.get("WORLD_SIZE", "1")),
        )
    if args.num_workers:
        parameters["prefetch_factor"] = 1
        parameters["persistent_workers"] = True
    return DataLoader(dataset, **parameters)


def _build_agent(
    args: argparse.Namespace,
) -> Tuple[LatentWAMAgent, nn.Module, torch.optim.Optimizer]:
    if args.vision_model is None:
        raise ValueError("train mode requires --vision-model")
    if not torch.cuda.is_available():
        raise RuntimeError("train mode requires a CUDA-compatible accelerator")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    config = LatentWAMConfig(vision=VisionConfig(model_id=str(args.vision_model)))
    agent = LatentWAMAgent(
        config=config,
        checkpoint_path=str(args.checkpoint) if args.checkpoint is not None else None,
        build_target_encoder=True,
    )
    agent.initialize()
    agent.to(device).train()
    training_module: nn.Module = _TrainingStepModule(agent).to(device)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        training_module = torch.nn.parallel.DistributedDataParallel(
            training_module,
            device_ids=[local_rank],
            find_unused_parameters=True,
        )
    optimizer = torch.optim.Adam(
        (parameter for parameter in training_module.parameters() if parameter.requires_grad),
        lr=config.training.learning_rate,
    )
    return agent, training_module, optimizer


def _optimization_step(
    agent: LatentWAMAgent,
    training_module: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Batch,
    precision: str,
) -> Dict[str, float]:
    features, targets = batch
    device = next(agent.model.parameters()).device
    device_targets = {key: value.to(device, non_blocking=True) for key, value in targets.items()}
    optimizer.zero_grad(set_to_none=True)
    forward_start = time.perf_counter()
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if precision == "bf16-mixed"
        else nullcontext()
    )
    with autocast_context:
        loss = training_module(features, device_targets)
    _synchronize()
    forward_seconds = time.perf_counter() - forward_start
    backward_start = time.perf_counter()
    loss.backward()
    _synchronize()
    backward_seconds = time.perf_counter() - backward_start
    optimizer_start = time.perf_counter()
    optimizer.step()
    agent.update_target_encoder()
    _synchronize()
    return {
        "forward": forward_seconds,
        "backward": backward_seconds,
        "optimizer": time.perf_counter() - optimizer_start,
    }


def _process_io_and_memory() -> Tuple[int, int]:
    process = psutil.Process()
    processes = [process, *process.children(recursive=True)]
    rss_bytes = 0
    read_bytes = 0
    for item in processes:
        try:
            rss_bytes += item.memory_info().rss
            read_bytes += item.io_counters().read_bytes
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    return rss_bytes, read_bytes


def _run(args: argparse.Namespace) -> Dict[str, object]:
    loader = _build_loader(args)
    iterator = iter(loader)
    agent = training_module = optimizer = None
    if args.mode == "train":
        agent, training_module, optimizer = _build_agent(args)

    data_seconds = []
    step_seconds = []
    phase_seconds: Dict[str, list[float]] = {"forward": [], "backward": [], "optimizer": []}
    measured_rss_start = measured_read_start = 0
    for step in range(args.warmup_steps + args.measure_steps):
        if step == args.warmup_steps:
            measured_rss_start, measured_read_start = _process_io_and_memory()
        batch, data_duration = _next_timed(iterator)
        if args.mode == "train":
            assert agent is not None and training_module is not None and optimizer is not None
            _synchronize()
            step_start = time.perf_counter()
            phases = _optimization_step(agent, training_module, optimizer, batch, args.precision)
            _synchronize()
            step_duration = time.perf_counter() - step_start
        else:
            step_duration = data_duration
        if step >= args.warmup_steps:
            data_seconds.append(data_duration)
            step_seconds.append(step_duration)
            if args.mode == "train":
                for name, duration in phases.items():
                    phase_seconds[name].append(duration)
        completed = step + 1
        if args.log_every_steps and completed % args.log_every_steps == 0:
            phase = "warmup" if completed <= args.warmup_steps else "measure"
            measured = len(step_seconds)
            throughput = measured * args.batch_size / sum(step_seconds) if measured else 0.0
            print(
                f"progress={completed}/{args.warmup_steps + args.measure_steps} phase={phase} "
                f"measured_steps={measured} samples_per_second={throughput:.3f}",
                flush=True,
            )

    total_step_seconds = sum(step_seconds)
    measured_rss_end, measured_read_end = _process_io_and_memory()
    source_dataset = loader.dataset
    if isinstance(source_dataset, RepeatedDataset):
        source_dataset = source_dataset.dataset
    report = {
        "mode": args.mode,
        "cache_path": str(args.cache_path.resolve()),
        "source_samples": len(source_dataset),
        "repeated_source": isinstance(loader.dataset, RepeatedDataset),
        "sample_token": args.sample_token,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "sampler_mode": args.sampler_mode,
        "precision": args.precision,
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "data_wait_seconds": summarize_seconds(data_seconds),
        "step_seconds": summarize_seconds(step_seconds),
        "samples_per_second": args.measure_steps * args.batch_size / total_step_seconds,
        "process_metrics": {
            "rss_bytes_after_warmup": measured_rss_start,
            "rss_bytes_end": measured_rss_end,
            "read_bytes_measured": max(0, measured_read_end - measured_read_start),
            "accelerator_utilization_percent": None,
            "accelerator_utilization_note": "not exposed by the installed PPU PyTorch runtime",
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device_count": torch.cuda.device_count(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "rank": int(os.environ.get("RANK", "0")),
        },
    }
    if args.mode == "train":
        report["phase_seconds"] = {name: summarize_seconds(values) for name, values in phase_seconds.items()}
    return report


def main() -> None:
    args = _arguments()
    if (
        args.batch_size < 1
        or args.num_workers < 0
        or args.warmup_steps < 0
        or args.measure_steps < 1
        or args.log_every_steps < 0
    ):
        raise ValueError("invalid benchmark iteration or dataloader setting")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    owns_process_group = world_size > 1 and not dist.is_initialized()
    if owns_process_group:
        dist.init_process_group("nccl")
    try:
        report = _run(args)
        reports = [report]
        if world_size > 1:
            reports = [None] * world_size
            dist.all_gather_object(reports, report)
        rank = int(os.environ.get("RANK", "0"))
        if rank == 0:
            output: Dict[str, object]
            if world_size == 1:
                output = report
            else:
                per_rank_seconds = [
                    args.measure_steps * args.batch_size / rank_report["samples_per_second"] for rank_report in reports
                ]
                output = {
                    "mode": args.mode,
                    "world_size": world_size,
                    "aggregate_samples_per_second": world_size
                    * args.measure_steps
                    * args.batch_size
                    / max(per_rank_seconds),
                    "ranks": reports,
                }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(output, indent=2))
    finally:
        if owns_process_group:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
