"""WorldMirror feature extraction used to create Latent WAM training caches."""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from navsim.agents.latent_wam.config import GeometryConfig, VisionConfig

WORLDMIRROR_CONDITION_FLAGS = (1, 0, 1)


def _build_camera_priors(
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    image_height: int,
    image_width: int,
) -> tuple[None, torch.Tensor, torch.Tensor]:
    from worldmirror.src.models.utils.camera_utils import extrinsics_to_vector
    from worldmirror.src.models.utils.priors import normalize_poses

    normalized_extrinsics = normalize_poses(extrinsics[..., :3, :])
    poses = extrinsics_to_vector(normalized_extrinsics)
    rays = torch.stack(
        [
            intrinsics[..., 0, 0] / image_width,
            intrinsics[..., 1, 1] / image_height,
            intrinsics[..., 0, 2] / image_width,
            intrinsics[..., 1, 2] / image_height,
        ],
        dim=-1,
    )
    return None, rays, poses


@dataclass(frozen=True)
class WorldMirrorInputs:
    """Images and calibrated cameras for WorldMirror feature extraction."""

    images: torch.Tensor
    intrinsics: torch.Tensor
    extrinsics: torch.Tensor

    def validate(self) -> None:
        """Validate the temporal/view dimensions and camera calibration."""
        if self.images.ndim != 5 or self.images.shape[-1] != 3:
            raise ValueError("images must have shape [T,V,H,W,C]")
        expected_prefix = self.images.shape[:2]
        if self.intrinsics.shape != (*expected_prefix, 3, 3):
            raise ValueError("intrinsics must have shape [T,V,3,3] and align with images")
        if self.extrinsics.shape != (*expected_prefix, 4, 4):
            raise ValueError("extrinsics must have shape [T,V,4,4] and align with images")
        if not torch.isfinite(self.intrinsics).all() or not torch.isfinite(self.extrinsics).all():
            raise ValueError("camera calibration must contain finite values")


class WorldMirrorFeatureExtractor:
    """Extract the 2048-D token representation used as geometry supervision."""

    def __init__(
        self,
        model: torch.nn.Module,
        processor: Any,
        device: torch.device,
        expected_patch_start_index: int = 7,
    ) -> None:
        self.model = model.eval().to(device)
        self.processor = processor
        self.device = device
        self.expected_patch_start_index = expected_patch_start_index

    @staticmethod
    def _resolve_model_directory(model_id: str) -> Path:
        model_path = Path(model_id).expanduser()
        if model_path.is_dir():
            return model_path.resolve()
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise RuntimeError("install huggingface-hub to download WorldMirror") from error
        return Path(
            snapshot_download(
                repo_id=model_id,
                allow_patterns=("config.json", "model.safetensors"),
            )
        )

    @staticmethod
    def _load_visual_encoder(model_directory: Path, device: torch.device) -> torch.nn.Module:
        """Load only the WorldMirror encoder used by geometry-cache extraction."""
        try:
            from lightning_utilities.core.rank_zero import rank_zero_only
            from safetensors import safe_open
            from src.models.models.visual_transformer import VisualGeometryTransformer
        except ImportError as error:
            raise RuntimeError("install the WorldMirror inference dependencies") from error

        # WorldMirror's logging adapter normally receives this value from a
        # Lightning trainer. Standalone cache extraction has no trainer.
        if getattr(rank_zero_only, "rank", None) is None:
            rank_zero_only.rank = 0

        config_path = model_directory / "config.json"
        weights_path = model_directory / "model.safetensors"
        if not config_path.is_file() or not weights_path.is_file():
            raise ValueError(f"WorldMirror checkpoint is incomplete: {model_directory}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        encoder = VisualGeometryTransformer(
            img_size=config["img_size"],
            patch_size=config["patch_size"],
            embed_dim=config["embed_dim"],
            enable_cond=config["enable_cond"],
            sampling_strategy=config["sampling_strategy"],
            patch_embed=config["patch_embed"],
            fixed_patch_embed=config["fixed_patch_embed"],
            condition_strategy=config["condition_strategy"],
        )

        prefix = "visual_geometry_transformer."
        state = encoder.state_dict()
        with safe_open(weights_path, framework="pt", device="cpu") as checkpoint:
            checkpoint_keys = set(checkpoint.keys())
            missing = sorted(prefix + key for key in state if prefix + key not in checkpoint_keys)
            if missing:
                raise ValueError(f"WorldMirror checkpoint is missing encoder key: {missing[0]}")
            with torch.no_grad():
                for key, destination in state.items():
                    source = checkpoint.get_tensor(prefix + key)
                    if source.shape != destination.shape:
                        raise ValueError(
                            f"WorldMirror encoder shape mismatch for {key}: "
                            f"expected {tuple(destination.shape)}, got {tuple(source.shape)}"
                        )
                    destination.copy_(source)
        return encoder.to(device)

    @classmethod
    def from_pretrained(
        cls,
        geometry: GeometryConfig,
        vision: VisionConfig,
        device: Optional[torch.device] = None,
        worldmirror_root: Path = Path("worldmirror"),
    ) -> "WorldMirrorFeatureExtractor":
        """Load the WorldMirror visual encoder and deterministic preprocessor.

        Args:
            geometry: WorldMirror checkpoint and token-layout configuration.
            vision: Image preprocessing dimensions and model identifier.
            device: Optional execution device; defaults via ``resolve_torch_device``.
            worldmirror_root: Root of the checked-out WorldMirror submodule.

        Returns:
            A ready-to-run visual geometry feature extractor.

        Raises:
            ValueError: If the submodule or checkpoint files are missing.
            RuntimeError: If required inference dependencies are unavailable.
        """
        root = worldmirror_root.resolve()
        if not root.is_dir():
            raise ValueError(f"WorldMirror submodule is missing: {root}")
        root_string = str(root)
        if root_string not in sys.path:
            sys.path.insert(0, root_string)
        try:
            from transformers import AutoImageProcessor
        except ImportError as error:
            raise RuntimeError("install transformers for WorldMirror preprocessing") from error

        if device is None:
            from navsim.agents.latent_wam.device import resolve_torch_device

            target_device = resolve_torch_device()
        else:
            target_device = device
        model_directory = cls._resolve_model_directory(geometry.model_id)
        model = cls._load_visual_encoder(model_directory, target_device)
        processor = AutoImageProcessor.from_pretrained(vision.model_id)
        processor.do_resize = True
        processor.size = {"shortest_edge": 256}
        processor.resample = 3
        processor.do_center_crop = True
        processor.do_rescale = True
        processor.rescale_factor = 1 / 255
        processor.do_normalize = False
        processor.crop_size = {"height": vision.crop_height, "width": vision.crop_width}
        return cls(
            model=model,
            processor=processor,
            device=target_device,
            expected_patch_start_index=geometry.patch_start_index,
        )

    @torch.inference_mode()
    def __call__(self, inputs: WorldMirrorInputs) -> torch.Tensor:
        """Extract camera-conditioned WorldMirror tokens.

        Args:
            inputs: Images and calibrated cameras aligned as ``[T,V,...]``.

        Returns:
            Float32 CPU tokens shaped ``[T,V,519,2048]``.

        Raises:
            ValueError: If input or output shapes do not match the cache schema.
        """
        inputs.validate()
        images = inputs.images
        time_steps, views, height, width, channels = images.shape
        processed = self.processor(
            images=images.reshape(time_steps * views, height, width, channels),
            return_tensors="pt",
        ).pixel_values
        processed = processed.reshape(time_steps, views, channels, *processed.shape[-2:]).to(self.device)
        processed_height, processed_width = processed.shape[-2:]
        intrinsics = inputs.intrinsics.to(self.device)
        priors = _build_camera_priors(
            intrinsics=intrinsics,
            extrinsics=inputs.extrinsics.to(self.device),
            image_height=processed_height,
            image_width=processed_width,
        )
        use_bfloat16 = self.device.type == "cuda"
        with torch.amp.autocast(self.device.type, enabled=use_bfloat16, dtype=torch.bfloat16):
            token_list, patch_start_index = self.model(
                processed,
                priors,
                cond_flags=list(WORLDMIRROR_CONDITION_FLAGS),
            )
        tokens = token_list[-1]
        if tokens.shape[-1] != 2048:
            raise ValueError(f"expected 2048-D WorldMirror tokens, got {tokens.shape[-1]}")
        if patch_start_index != self.expected_patch_start_index:
            raise ValueError(f"expected patch start index {self.expected_patch_start_index}, got {patch_start_index}")
        if patch_start_index >= tokens.shape[-2]:
            raise ValueError("WorldMirror patch start index is outside the token sequence")
        return tokens.detach().float().cpu()
