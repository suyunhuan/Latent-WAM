#!/usr/bin/env python3
"""Strictly load released weights and run a one-sample trajectory smoke test."""

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import BitImageProcessor, Dinov2Config, Dinov2Model

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from navsim.agents.latent_wam.config import LatentWAMConfig  # noqa: E402
from navsim.agents.latent_wam.model import LatentWAMModel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision_encoder = Dinov2Model(
        Dinov2Config(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            patch_size=14,
            image_size=518,
        )
    )
    processor = BitImageProcessor(
        do_resize=True,
        size={"height": 224, "width": 448},
        do_center_crop=True,
        crop_size={"height": 224, "width": 448},
    )
    model = LatentWAMModel(
        LatentWAMConfig(),
        vision_encoder=vision_encoder,
        image_processor=processor,
        build_target_encoder=False,
    ).to(device)
    model.load_state_dict(load_file(str(args.weights), device=str(device)), strict=True)
    images = torch.zeros((1, 3, 224, 448, 3), dtype=torch.uint8)
    ego_status = torch.tensor([[0, 1, 0, 0, 0, 0, 0, 0]], dtype=torch.float32, device=device)
    with torch.inference_mode():
        output = model.predict(images, ego_status)
    if output["trajectory"].shape != (1, 8, 3):
        raise RuntimeError(f"unexpected trajectory shape: {output['trajectory'].shape}")
    if not torch.isfinite(output["trajectory"]).all():
        raise RuntimeError("trajectory contains non-finite values")
    print("checkpoint and inference smoke test passed")


if __name__ == "__main__":
    main()
