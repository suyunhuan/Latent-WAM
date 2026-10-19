#!/usr/bin/env python3
"""Convert a Latent WAM Lightning .ckpt to model-only .safetensors on CPU."""

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def main() -> None:
    """Parse conversion options and publish a standalone model weights file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="source Lightning .ckpt")
    parser.add_argument("--output", type=Path, help="default: replace the input suffix with .safetensors")
    parser.add_argument("--inference-only", action="store_true", help="omit EMA target encoder weights")
    parser.add_argument("--overwrite", action="store_true", help="allow replacing an existing output")
    args = parser.parse_args()

    from navsim.agents.latent_wam.checkpoint import convert_checkpoint_to_safetensors

    output = args.output or args.checkpoint.with_suffix(".safetensors")
    try:
        metadata = convert_checkpoint_to_safetensors(
            args.checkpoint, output, inference_only=args.inference_only, overwrite=args.overwrite
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps({"source": str(args.checkpoint.resolve()), "output": str(output.resolve()), "metadata": metadata}))


if __name__ == "__main__":
    main()
