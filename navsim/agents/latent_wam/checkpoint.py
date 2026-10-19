"""Model-weight serialization and paired Lightning checkpoint exports."""

import os
import pickle
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor, nn

TEMPORAL_ATTENTION_MASK_CHECKPOINT_KEY = "latent_wam_temporal_attention_mask"
LIGHTNING_MODEL_PREFIX = "agent.model."


def _read_lightning_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as error:
        raise ValueError("checkpoint is not readable by the tensor-only loader; export model weights first") from error
    if not isinstance(checkpoint, Mapping) or "state_dict" not in checkpoint:
        raise ValueError("Lightning checkpoint must contain a state_dict mapping")
    return checkpoint


def _normalize_model_weights(weights: Any, *, lightning_checkpoint: bool) -> Dict[str, Tensor]:
    if not isinstance(weights, Mapping) or not weights:
        raise ValueError("model state_dict must be a non-empty mapping")
    if any(not isinstance(key, str) or not isinstance(value, Tensor) for key, value in weights.items()):
        raise ValueError("model state_dict must contain string keys and tensor values only")
    if lightning_checkpoint and any(key.startswith(LIGHTNING_MODEL_PREFIX) for key in weights):
        if not all(key.startswith(LIGHTNING_MODEL_PREFIX) for key in weights):
            raise ValueError("mixed checkpoint namespaces; all model keys must start with agent.model.")
        return {key[len(LIGHTNING_MODEL_PREFIX) :]: value for key, value in weights.items()}
    return dict(weights)


def load_model_weights(path: Path) -> Dict[str, Tensor]:
    """Read tensors only, without restoring any Lightning training state.

    Args:
        path: Bare-model safetensors file or Lightning checkpoint. A checkpoint
            state_dict may use bare model keys or the uniform ``agent.model.`` prefix.

    Returns:
        Model tensors on CPU with bare model parameter names.

    Raises:
        ValueError: If the format, checkpoint structure, or key namespace is invalid.
    """
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        weights = load_file(str(path), device="cpu")
    elif path.suffix == ".ckpt":
        weights = _read_lightning_checkpoint(path)["state_dict"]
    else:
        raise ValueError("Latent WAM weights must use .ckpt or .safetensors format")
    return _normalize_model_weights(weights, lightning_checkpoint=path.suffix == ".ckpt")


def save_model_safetensors(model: nn.Module, path: Path, metadata: Optional[Dict[str, str]] = None) -> None:
    """Atomically export all model weights, including EMA when present.

    Args:
        model: Model whose tensor state is exported, without Lightning wrapper keys.
        path: Destination on a local or mounted filesystem.
        metadata: Optional text metadata describing the source training checkpoint.
    """
    _save_weight_tensors(model.state_dict(), path, metadata, overwrite=True)


def _save_weight_tensors(
    weights: Mapping[str, Tensor], path: Path, metadata: Optional[Dict[str, str]], *, overwrite: bool
) -> None:
    from safetensors.torch import save_file

    if not overwrite and (path.exists() or path.is_symlink()):
        raise FileExistsError(f"output already exists: {path}; pass --overwrite to replace it")
    weights = {key: value.detach().cpu().clone().contiguous() for key, value in weights.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        save_file(weights, str(temporary), metadata=metadata)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # Publish a complete file without a check-then-replace overwrite race.
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def convert_checkpoint_to_safetensors(
    source: Path, destination: Path, *, inference_only: bool = False, overwrite: bool = False
) -> Dict[str, str]:
    """Convert a Lightning checkpoint to model weights without building a model.

    Args:
        source: Input ``.ckpt`` file, which is never modified.
        destination: Output ``.safetensors`` on a local or mounted filesystem.
        inference_only: Explicitly omit ``target_dino_encoder.*`` tensors.
        overwrite: Allow replacing an existing output file.

    Returns:
        Exported provenance metadata; optimizer and scheduler state are excluded.

    Raises:
        ValueError: If paths, tensor namespaces, or source mask metadata are invalid.
        FileExistsError: If the destination exists and overwrite is disabled.
    """
    if source.suffix != ".ckpt" or destination.suffix != ".safetensors":
        raise ValueError("conversion requires a .ckpt input and a .safetensors output")
    if source.resolve() == destination.resolve() or (
        source.exists() and destination.exists() and source.samefile(destination)
    ):
        raise ValueError("input and output must be different files")
    if not overwrite and (destination.exists() or destination.is_symlink()):
        raise FileExistsError(f"output already exists: {destination}; pass --overwrite to replace it")
    checkpoint = _read_lightning_checkpoint(source)
    weights = _normalize_model_weights(checkpoint["state_dict"], lightning_checkpoint=True)
    mode = checkpoint.get(TEMPORAL_ATTENTION_MASK_CHECKPOINT_KEY, "block_diagonal")
    if mode not in ("block_diagonal", "block_causal"):
        raise ValueError(f"invalid source temporal attention mask: {mode!r}")
    has_target = any(key.startswith("target_dino_encoder.") for key in weights)
    if inference_only:
        weights = {key: value for key, value in weights.items() if not key.startswith("target_dino_encoder.")}
        if not weights:
            raise ValueError("no model weights remain after omitting the target encoder")
    metadata = {
        "latent_wam_format": "latent_wam.model_weights.v1",
        TEMPORAL_ATTENTION_MASK_CHECKPOINT_KEY: mode,
        "source_checkpoint": source.name,
        "ema_target_encoder": "excluded" if inference_only and has_target else "included" if has_target else "absent",
    }
    for key in ("epoch", "global_step"):
        if key in checkpoint:
            metadata[key] = str(checkpoint[key])
    _save_weight_tensors(weights, destination, metadata, overwrite=overwrite)
    return metadata


class LatentWAMModelCheckpoint(ModelCheckpoint):
    """Keep full Lightning checkpoints and optionally export matching model weights."""

    def __init__(self, *, export_safetensors: bool = False, **kwargs: Any) -> None:
        if kwargs.get("save_weights_only", False):
            raise ValueError("Latent WAM training checkpoints must retain full training state")
        super().__init__(**kwargs)
        self.export_safetensors = export_safetensors

    @property
    def state_key(self) -> str:
        """Preserve the original ModelCheckpoint callback identity for exact resume."""
        return super().state_key.replace(type(self).__qualname__, ModelCheckpoint.__qualname__, 1)

    def _save_checkpoint(self, trainer, filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)
        if self.export_safetensors and trainer.is_global_zero:
            agent = trainer.lightning_module.agent
            metadata = {
                "latent_wam_format": "latent_wam.model_weights.v1",
                TEMPORAL_ATTENTION_MASK_CHECKPOINT_KEY: agent.config.training.temporal_attention_mask,
                "global_step": str(trainer.global_step),
                "epoch": str(trainer.current_epoch),
                "source_checkpoint": Path(filepath).name,
            }
            save_model_safetensors(agent.model, Path(filepath).with_suffix(".safetensors"), metadata)

    def _remove_checkpoint(self, trainer, filepath: str) -> None:
        super()._remove_checkpoint(trainer, filepath)
        if self.export_safetensors and trainer.is_global_zero:
            Path(filepath).with_suffix(".safetensors").unlink(missing_ok=True)
