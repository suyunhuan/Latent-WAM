"""Checkpoint migration utilities for released Latent WAM weights."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Mapping, Tuple

import torch

from navsim.agents.latent_wam.config import LatentWAMConfig

LEGACY_MODEL_PREFIX = "agent._transfuser_model."
DROP_PREFIXES: Tuple[str, ...] = (
    "dino_encoder.",
    "target_dino_encoder.",
    "refine_block.",
    "refine_traj_head.",
    "geometry_decoder.",
)
DROP_KEYS = {"geometry_query"}
RENAME_PREFIXES: Tuple[Tuple[str, str], ...] = (
    ("lora_dino_encoder.", "vision_encoder."),
    ("_trajectory_head.", "trajectory_head."),
    ("_status_encoding.", "status_encoder."),
    ("_mode_embedding.", "mode_encoder."),
    ("_keyval_embedding.", "keyval_embedding."),
    ("_query_embedding.", "trajectory_queries."),
    ("_wm_query_embedding.", "world_queries."),
    ("_tf_decoder.", "trajectory_decoder."),
)


def _strip_legacy_prefix(key: str) -> str:
    if key.startswith(LEGACY_MODEL_PREFIX):
        return key[len(LEGACY_MODEL_PREFIX) :]
    return key


def migrate_key(key: str) -> str | None:
    """Map one legacy Lightning state key to the public model schema."""
    key = _strip_legacy_prefix(key)
    if key in DROP_KEYS or key.startswith(DROP_PREFIXES):
        return None
    for source, destination in RENAME_PREFIXES:
        if key.startswith(source):
            return destination + key[len(source) :]
    return key


def migrate_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Tuple[str, ...]]:
    """Migrate a legacy state dict and return the explicitly dropped keys."""
    migrated: Dict[str, torch.Tensor] = {}
    dropped = []
    for old_key, tensor in state_dict.items():
        new_key = migrate_key(old_key)
        if new_key is None:
            dropped.append(old_key)
            continue
        if new_key in migrated:
            raise ValueError(f"checkpoint key collision after migration: {new_key}")
        migrated[new_key] = tensor
    return migrated, tuple(dropped)


def load_legacy_checkpoint(path: Path) -> Mapping[str, torch.Tensor]:
    """Load tensors from a trusted legacy Lightning checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if "state_dict" not in checkpoint:
        raise ValueError("legacy checkpoint does not contain a state_dict")
    return checkpoint["state_dict"]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Compute the SHA-256 checksum of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def export_safetensors(
    checkpoint_path: Path,
    output_path: Path,
    config: LatentWAMConfig,
) -> Dict[str, object]:
    """Convert a trusted Lightning checkpoint to public safetensors weights."""
    try:
        from safetensors.torch import save_file
    except ImportError as error:
        raise RuntimeError("install safetensors to export public weights") from error

    state_dict = load_legacy_checkpoint(checkpoint_path)
    migrated, dropped = migrate_state_dict(state_dict)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # The released temporal decoder registered one shared layer four times. A
    # safetensors file cannot encode storage aliases, so materialize independent
    # tensors while retaining the identical values expected by the public model.
    tensors = {key: tensor.detach().clone().contiguous() for key, tensor in migrated.items()}
    save_file(tensors, str(output_path))

    config_path = output_path.with_name("config.json")
    config_path.write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
    report = {
        "format_version": 1,
        "source_file": checkpoint_path.name,
        "source_sha256": sha256_file(checkpoint_path),
        "source_keys": len(state_dict),
        "exported_keys": len(migrated),
        "dropped_keys": len(dropped),
        "weights_sha256": sha256_file(output_path),
    }
    report_path = output_path.with_name("conversion_report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
