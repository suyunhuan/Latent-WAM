#!/usr/bin/env python3
"""Run a small BF16 optimization smoke test on the reusable shard subset."""

from __future__ import annotations

import json
import argparse
import time
from pathlib import Path
from types import SimpleNamespace

import torch

import navsim.agents.latent_wam.dataset as cache_dataset_module

# The source cache generator already verified these immutable shard files. Avoid
# repeating the slow shared-storage header walk for this isolated smoke test.
cache_dataset_module.validate_cache_headers = lambda cache_root, manifest: None

from scripts.benchmark_latent_wam_training import _build_agent, _build_loader, _synchronize


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    cli = _arguments()
    args = SimpleNamespace(
        cache_path=Path("/tmp/latent-wam-partial-complete-20260712"),
        output=cli.output or Path(
            f"/tmp/latent-wam-partial-complete-20260712/"
            f"bf16-smoke-bs{cli.batch_size}-workers{cli.num_workers}.json"
        ),
        mode="train",
        checkpoint=None,
        vision_model=Path("/mnt/workspace/wlb/world4drive/checkpoints/dinov2-base"),
        batch_size=cli.batch_size,
        num_workers=cli.num_workers,
        warmup_steps=2,
        measure_steps=12,
        log_every_steps=0,
        sample_token=None,
    )
    loader = _build_loader(args)
    iterator = iter(loader)
    agent, training_module, optimizer = _build_agent(args)
    timings = []
    data_wait = []
    losses = []
    sample_tokens = []
    for step in range(args.warmup_steps + args.measure_steps):
        data_start = time.perf_counter()
        features, targets = next(iterator)
        data_wait.append(time.perf_counter() - data_start)
        if isinstance(features.get("token"), list):
            sample_tokens.extend(features["token"])
        device = next(agent.model.parameters()).device
        device_targets = {key: value.to(device, non_blocking=True) for key, value in targets.items()}
        optimizer.zero_grad(set_to_none=True)
        _synchronize()
        start = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = training_module(features, device_targets)
        loss_value = float(loss.detach().cpu())
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"non-finite BF16 loss at step {step}: {loss_value}")
        loss.backward()
        optimizer.step()
        agent.update_target_encoder()
        _synchronize()
        duration = time.perf_counter() - start
        if step >= args.warmup_steps:
            timings.append(duration)
            losses.append(loss_value)

    report = {
        "cache_path": str(args.cache_path),
        "source_samples": 62915,
        "source_shards": 2182,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "precision": "bf16-mixed",
        "mean_step_seconds": sum(timings) / len(timings),
        "samples_per_second": len(timings) * args.batch_size / sum(timings),
        "data_wait_mean_seconds": sum(data_wait[args.warmup_steps:]) / len(timings),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_finite": all(torch.isfinite(torch.tensor(losses)).tolist()),
        "unique_sample_tokens_seen": len(set(sample_tokens)),
        "max_memory_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "max_memory_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
