#!/usr/bin/env python3
"""Run the public Latent WAM single-stage ePDMS evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from navsim.evaluate.latent_wam_epdms import EpdmsEvaluationConfig, evaluate_epdms


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the formal evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--navsim-log-path", type=Path, required=True)
    parser.add_argument("--sensor-path", type=Path, required=True)
    parser.add_argument("--metric-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--vision-model",
        default=None,
        help="Local DINOv2 snapshot or Hugging Face model ID; defaults to facebook/dinov2-base.",
    )
    parser.add_argument(
        "--split",
        default="navtest",
        help="NAVSIM scene-filter split (default: navtest).",
    )
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=None,
        help="Optional smoke-test limit; omit it for the complete split.",
    )
    parser.add_argument(
        "--no-human-penalty-filter",
        action="store_false",
        dest="human_penalty_filter",
        help="Disable the NAVSIM human-penalty exemption (not recommended for reported ePDMS).",
    )
    parser.add_argument(
        "--mode-select",
        default="command",
        choices=["command", "mode_logits", "hybrid"],
        help="Inference trajectory mode selection (default: command).",
    )
    parser.add_argument(
        "--hybrid-margin",
        type=float,
        default=0.0,
        help="For --mode-select=hybrid: override command when logit margin exceeds this.",
    )
    args = parser.parse_args(argv)
    if args.max_scenarios is not None and args.max_scenarios < 1:
        raise ValueError("--max-scenarios must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    """Run evaluation and write the standard NAVSIM score CSV."""
    args = parse_args(argv)
    score_df = evaluate_epdms(
        EpdmsEvaluationConfig(
            checkpoint_path=args.checkpoint,
            navsim_log_path=args.navsim_log_path,
            sensor_path=args.sensor_path,
            metric_cache_path=args.metric_cache,
            vision_model=args.vision_model,
            split=args.split,
            max_scenarios=args.max_scenarios,
            human_penalty_filter=args.human_penalty_filter,
            mode_select=args.mode_select,
            hybrid_margin=args.hybrid_margin,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    score_df.to_csv(args.output, index=False)
    average_row = score_df.iloc[-1]
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "scenarios": len(score_df) - 1,
                "valid": bool(score_df.iloc[:-1]["valid"].all()),
                "epdms": float(average_row["score"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
