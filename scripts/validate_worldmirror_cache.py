#!/usr/bin/env python3
"""Compare camera-conditioned WorldMirror extraction with one trusted legacy frame."""

import argparse
import gzip
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from navsim.agents.latent_wam.config import GeometryConfig, VisionConfig
from navsim.agents.latent_wam.features import CURRENT_FRAME_INDEX, TRAINING_FRAME_INDICES, build_camera_sensor_config
from navsim.agents.latent_wam.geometry import WorldMirrorFeatureExtractor, WorldMirrorInputs
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.script.run_latent_wam_caching import build_world_mirror_inputs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-frame-cache", type=Path, required=True)
    parser.add_argument("--legacy-geometry", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-split", default="trainval")
    parser.add_argument("--calibration-log", required=True)
    parser.add_argument("--calibration-token", required=True)
    parser.add_argument("--worldmirror-model", required=True)
    parser.add_argument("--vision-model", required=True)
    parser.add_argument("--minimum-cosine", type=float, default=0.999)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run the legacy WorldMirror cache comparison and write a JSON report."""
    args = _arguments()
    with gzip.open(args.legacy_frame_cache, "rb") as file:
        legacy_frame: dict[str, Any] = pickle.load(file)
    if "dino_feature" not in legacy_frame:
        raise ValueError("legacy frame cache has no dino_feature")
    legacy_images = torch.as_tensor(legacy_frame["dino_feature"])
    legacy_geometry = torch.from_numpy(np.array(np.load(args.legacy_geometry, mmap_mode="r"), copy=True))

    split_root = args.data_root / "meta_datas" / args.data_split
    sensor_root = args.data_root / "sensor_blobs" / args.data_split
    scene_filter = SceneFilter(
        frame_interval=1,
        max_scenes=1,
        log_names=[args.calibration_log],
        tokens=[args.calibration_token],
    )
    loader = SceneLoader(
        data_path=split_root,
        original_sensor_path=sensor_root,
        scene_filter=scene_filter,
        sensor_config=build_camera_sensor_config(list(TRAINING_FRAME_INDICES)),
    )
    if args.calibration_token not in loader.tokens:
        raise ValueError(f"calibration token is not available: {args.calibration_token}")
    calibrated = build_world_mirror_inputs(loader.get_scene_from_token(args.calibration_token))
    current_offset = TRAINING_FRAME_INDICES.index(CURRENT_FRAME_INDEX)
    inputs = WorldMirrorInputs(
        images=legacy_images[None],
        intrinsics=calibrated.intrinsics[current_offset : current_offset + 1],
        extrinsics=calibrated.extrinsics[current_offset : current_offset + 1],
    )
    extractor = WorldMirrorFeatureExtractor.from_pretrained(
        GeometryConfig(model_id=args.worldmirror_model),
        VisionConfig(model_id=args.vision_model),
    )
    current_geometry = extractor(inputs)
    if current_geometry.shape != legacy_geometry.shape:
        raise ValueError(
            f"geometry shape mismatch: current {tuple(current_geometry.shape)}, "
            f"legacy {tuple(legacy_geometry.shape)}"
        )
    delta = current_geometry - legacy_geometry
    feature_dim = current_geometry.shape[-1]
    mean_cosine = F.cosine_similarity(
        current_geometry.reshape(-1, feature_dim),
        legacy_geometry.reshape(-1, feature_dim),
        dim=-1,
    ).mean()
    report = {
        "legacy_frame_cache_sha256": _sha256(args.legacy_frame_cache),
        "legacy_geometry_sha256": _sha256(args.legacy_geometry),
        "calibration_log": args.calibration_log,
        "calibration_token": args.calibration_token,
        "shape": list(current_geometry.shape),
        "minimum_cosine": args.minimum_cosine,
        "mean_cosine": mean_cosine.item(),
        "mean_absolute_error": delta.abs().mean().item(),
        "maximum_absolute_error": delta.abs().max().item(),
        "rmse": delta.square().mean().sqrt().item(),
        "passed": mean_cosine.item() >= args.minimum_cosine,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
