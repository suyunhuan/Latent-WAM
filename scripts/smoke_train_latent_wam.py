#!/usr/bin/env python3
"""Run one real-cache Latent WAM optimization step and reload the result."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from navsim.agents.latent_wam.agent import LatentWAMAgent
from navsim.agents.latent_wam.config import LatentWAMConfig, VisionConfig
from navsim.agents.latent_wam.dataset import LatentWAMCacheDataset


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vision-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _inference_state(agent: LatentWAMAgent) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone().contiguous()
        for key, value in agent.model.state_dict().items()
        if not key.startswith("target_dino_encoder.")
    }


def main() -> None:
    args = _arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("the training smoke test requires a CUDA-compatible device")
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    try:
        torch.cuda.reset_peak_memory_stats(0)
        memory_stats_available = True
    except RuntimeError:
        memory_stats_available = False

    config = LatentWAMConfig(vision=VisionConfig(model_id=str(args.vision_model)))
    agent = LatentWAMAgent(
        config=config,
        checkpoint_path=str(args.checkpoint),
        build_target_encoder=True,
    )
    agent.initialize()
    agent.to(device).train()

    cached_features, cached_targets = LatentWAMCacheDataset(args.cache_path)[0]
    features = {
        key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else value
        for key, value in cached_features.items()
        if key != "token"
    }
    targets = {key: value.unsqueeze(0).to(device) for key, value in cached_targets.items()}
    optimizer = torch.optim.Adam(
        (parameter for parameter in agent.model.parameters() if parameter.requires_grad),
        lr=config.training.learning_rate,
    )

    optimizer.zero_grad(set_to_none=True)
    predictions = agent.forward_train(features)
    losses = agent.compute_loss(features, targets, predictions)
    total_loss = torch.stack(tuple(losses.values())).sum()
    if not torch.isfinite(total_loss):
        raise RuntimeError(f"training loss is not finite: {total_loss.item()}")
    total_loss.backward()

    online_parameters = dict(agent.model.vision_encoder.named_parameters())
    target_parameters = dict(agent.model.target_dino_encoder.named_parameters())
    gradient_name = next(
        (
            name
            for name, parameter in online_parameters.items()
            if parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().max() > 0
        ),
        None,
    )
    if gradient_name is None:
        raise RuntimeError("the online DINO encoder received no finite non-zero gradient")
    target_before = target_parameters[gradient_name].detach().clone()
    gradient_max = float(online_parameters[gradient_name].grad.detach().abs().max().cpu())

    all_gradients = [parameter.grad for parameter in agent.model.parameters() if parameter.grad is not None]
    if not all(torch.isfinite(gradient).all() for gradient in all_gradients):
        raise RuntimeError("at least one model gradient is not finite")
    optimizer.step()
    agent.update_target_encoder()
    target_delta = float((target_parameters[gradient_name] - target_before).abs().max().cpu())
    if target_delta <= 0:
        raise RuntimeError("the EMA target encoder did not update")

    loss_values = {name: float(value.detach().cpu()) for name, value in losses.items()}
    total_loss_value = float(total_loss.detach().cpu())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    weights_path = args.output.with_suffix(".safetensors")
    save_file(_inference_state(agent), str(weights_path))
    peak_memory_gib = torch.cuda.max_memory_allocated(0) / 1024**3 if memory_stats_available else None

    del (
        optimizer,
        predictions,
        losses,
        total_loss,
        all_gradients,
        online_parameters,
        target_parameters,
        target_before,
        agent,
    )
    torch.cuda.empty_cache()
    reloaded = LatentWAMAgent(config=config, checkpoint_path=str(weights_path))
    reloaded.initialize()
    reloaded.to(device).eval()
    current_frame_index = config.data.time_offsets_s.index(0.0)
    with torch.inference_mode():
        inference = reloaded.forward(
            {
                "images": features["images"][:, current_frame_index].to(device),
                "ego_status": features["ego_status"][:, current_frame_index].to(device),
            }
        )
    trajectory = inference["trajectory"]
    if trajectory.shape != (1, config.trajectory.num_poses, 3) or not torch.isfinite(trajectory).all():
        raise RuntimeError("reloaded inference did not produce a finite [1,8,3] trajectory")

    report = {
        "losses": loss_values,
        "total_loss": total_loss_value,
        "gradient_parameter": gradient_name,
        "gradient_max": gradient_max,
        "ema_target_max_delta": target_delta,
        "peak_memory_gib": peak_memory_gib,
        "weights_path": str(weights_path),
        "trajectory_shape": list(trajectory.shape),
        "trajectory_finite": True,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
