#!/usr/bin/env python3
"""Convert the original Lightning checkpoint into Latent WAM safetensors."""

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from navsim.agents.latent_wam.checkpoint import export_safetensors  # noqa: E402
from navsim.agents.latent_wam.config import LatentWAMConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="trusted epoch84-style .ckpt file")
    parser.add_argument("output", type=Path, help="output model.safetensors path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = export_safetensors(args.checkpoint, args.output, LatentWAMConfig())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
