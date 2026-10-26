#!/usr/bin/env python3
"""Run one Latent WAM optimization step and reload the result.

Use a real Latent WAM cache, or ``--synthetic`` for Apple Silicon / device smokes
without maps, sensors, WorldMirror, or the full nuplan stack.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from safetensors.torch import load_file, save_file
from transformers import Dinov2Config, Dinov2Model

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from navsim.agents.latent_wam.config import LatentWAMConfig, VisionConfig  # noqa: E402
from navsim.agents.latent_wam.device import (  # noqa: E402
    describe_device,
    empty_cache,
    max_memory_allocated_gib,
    resolve_torch_device,
)
from navsim.agents.latent_wam.model import LatentWAMModel  # noqa: E402


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--vision-model",
        default=None,
        help="Local DINOv2 snapshot or HF id for the cache/agent path.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic batch tensors instead of a Latent WAM cache (MPS/CPU smoke).",
    )
    parser.add_argument("--device", default=None, help="auto|cuda|mps|cpu")
    args = parser.parse_args()
    if not args.synthetic and args.cache_path is None:
        parser.error("--cache-path is required unless --synthetic is set")
    return args


def _synthetic_batch(device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Build one teacher-forcing batch with released-schema shapes."""
    generator = torch.Generator().manual_seed(0)
    ego_status = torch.zeros(1, 4, 8, dtype=torch.float32)
    ego_status[..., 0] = 1.0
    features = {
        "images": torch.randint(0, 256, (1, 4, 3, 224, 448, 3), dtype=torch.uint8, generator=generator),
        "geometry": torch.randn(1, 4, 3, 519, 2048, generator=generator),
        "ego_status": ego_status,
    }
    targets = {"trajectory": torch.zeros(1, 8, 3, dtype=torch.float32, device=device)}
    return features, targets


def _build_released_skeleton_encoder() -> Dinov2Model:
    """Match the released DINOv2-base topology without downloading weights."""
    return Dinov2Model(
        Dinov2Config(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            patch_size=14,
            image_size=518,
        )
    )


def _load_model(checkpoint: Path, device: torch.device, vision_model: str | None) -> LatentWAMModel:
    config = LatentWAMConfig(vision=VisionConfig(model_id=vision_model or "facebook/dinov2-base"))
    if device.type != "cuda":
        config.training.learning_rate = min(float(config.training.learning_rate), 1e-5)
    model = LatentWAMModel(
        config,
        vision_encoder=_build_released_skeleton_encoder(),
        build_target_encoder=True,
    ).to(device)
    state = load_file(str(checkpoint), device=str(device))
    incompatible = model.load_state_dict(state, strict=False)
    if model.target_dino_encoder is not None:
        missing_target = [key for key in incompatible.missing_keys if key.startswith("target_dino_encoder.")]
        if missing_target:
            model.target_dino_encoder.load_state_dict(model.vision_encoder.state_dict())
    model.train()
    return model


def _inference_state(model: LatentWAMModel) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone().contiguous()
        for key, value in model.state_dict().items()
        if not key.startswith("target_dino_encoder.")
    }


def _optimize_and_reload(
    *,
    model: LatentWAMModel,
    config: LatentWAMConfig,
    features: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    device: torch.device,
    output: Path,
    synthetic: bool,
    update_target: Callable[[], None],
) -> dict:
    memory_stats_available = False
    if device.type == "cuda":
        try:
            torch.cuda.reset_peak_memory_stats(device.index or 0)
            memory_stats_available = True
        except RuntimeError:
            memory_stats_available = False

    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.training.learning_rate,
    )
    optimizer.zero_grad(set_to_none=True)
    predictions = model.forward_train(features)
    losses = model.compute_loss(targets, predictions)
    total_loss = torch.stack(tuple(losses.values())).sum()
    if not torch.isfinite(total_loss):
        raise RuntimeError(f"training loss is not finite: {total_loss.item()}")
    total_loss.backward()

    def _grad_finite_nonzero(grad: torch.Tensor) -> bool:
        # MPS aborts on reductions over rank>4; flatten before finite/max checks.
        flat = grad.detach().reshape(-1)
        return bool(torch.isfinite(flat).all().item()) and float(flat.abs().max().cpu()) > 0

    online_parameters = dict(model.vision_encoder.named_parameters())
    target_parameters = dict(model.target_dino_encoder.named_parameters())
    gradient_name = next(
        (
            name
            for name, parameter in online_parameters.items()
            if parameter.grad is not None and _grad_finite_nonzero(parameter.grad)
        ),
        None,
    )
    if gradient_name is None:
        raise RuntimeError("the online DINO encoder received no finite non-zero gradient")
    target_before = target_parameters[gradient_name].detach().clone()
    gradient_max = float(online_parameters[gradient_name].grad.detach().reshape(-1).abs().max().cpu())

    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        if not bool(torch.isfinite(parameter.grad.detach().reshape(-1)).all().item()):
            raise RuntimeError("at least one model gradient is not finite")
    optimizer.step()
    update_target()
    target_delta = float((target_parameters[gradient_name] - target_before).detach().reshape(-1).abs().max().cpu())
    if target_delta <= 0:
        raise RuntimeError("the EMA target encoder did not update")

    loss_values = {name: float(value.detach().cpu()) for name, value in losses.items()}
    total_loss_value = float(total_loss.detach().cpu())
    output.parent.mkdir(parents=True, exist_ok=True)
    weights_path = output.with_suffix(".safetensors")
    save_file(_inference_state(model), str(weights_path))
    peak_memory_gib = max_memory_allocated_gib(device) if memory_stats_available else None

    empty_cache(device)

    reloaded = LatentWAMModel(
        config,
        vision_encoder=_build_released_skeleton_encoder(),
        build_target_encoder=False,
    ).to(device)
    reloaded.load_state_dict(load_file(str(weights_path), device=str(device)), strict=True)
    reloaded.eval()
    current_frame_index = config.data.time_offsets_s.index(0.0)
    with torch.inference_mode():
        inference = reloaded.predict(
            features["images"][:, current_frame_index].to(device),
            features["ego_status"][:, current_frame_index].to(device),
        )
    trajectory = inference["trajectory"]
    if trajectory.shape != (1, config.trajectory.num_poses, 3) or not torch.isfinite(trajectory).all():
        raise RuntimeError("reloaded inference did not produce a finite [1,8,3] trajectory")

    return {
        "device": describe_device(device),
        "synthetic": bool(synthetic),
        "learning_rate": float(config.training.learning_rate),
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


def _run_with_agent(args: argparse.Namespace, device: torch.device) -> dict:
    """Full-stack path that needs nuplan + a real Latent WAM cache."""
    from navsim.agents.latent_wam.agent import LatentWAMAgent
    from navsim.agents.latent_wam.dataset import LatentWAMCacheDataset

    vision_model = str(args.vision_model) if args.vision_model is not None else "facebook/dinov2-base"
    config = LatentWAMConfig(vision=VisionConfig(model_id=vision_model))
    if device.type != "cuda":
        config.training.learning_rate = min(float(config.training.learning_rate), 1e-5)
    agent = LatentWAMAgent(
        config=config,
        checkpoint_path=str(args.checkpoint),
        build_target_encoder=True,
    )
    agent.initialize(device=device)
    agent.to(device).train()
    cached_features, cached_targets = LatentWAMCacheDataset(args.cache_path)[0]
    features = {
        key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else value
        for key, value in cached_features.items()
        if key != "token"
    }
    targets = {key: value.unsqueeze(0).to(device) for key, value in cached_targets.items()}
    return _optimize_and_reload(
        model=agent.model,
        config=config,
        features=features,
        targets=targets,
        device=device,
        output=args.output,
        synthetic=False,
        update_target=agent.update_target_encoder,
    )


def main() -> None:
    args = _arguments()
    device = resolve_torch_device(args.device, require_accelerator=True)
    print(f"training smoke device: {describe_device(device)}")
    torch.manual_seed(0)

    if args.synthetic:
        model = _load_model(args.checkpoint, device, args.vision_model)
        features, targets = _synthetic_batch(device)
        momentum = float(model.config.training.target_momentum)

        def _update_target() -> None:
            model.update_target_encoder(momentum)

        report = _optimize_and_reload(
            model=model,
            config=model.config,
            features=features,
            targets=targets,
            device=device,
            output=args.output,
            synthetic=True,
            update_target=_update_target,
        )
    else:
        report = _run_with_agent(args, device)

    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
