"""Strongly typed configuration for Latent WAM."""

from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple


@dataclass
class VisionConfig:
    """Vision encoder and preprocessing settings."""

    model_id: str = "facebook/dinov2-base"
    feature_dim: int = 768
    crop_height: int = 224
    crop_width: int = 448


@dataclass
class GeometryConfig:
    """WorldMirror supervision settings."""

    model_id: str = "tencent/HunyuanWorld-Mirror"
    feature_dim: int = 2048
    patch_start_index: int = 7


@dataclass
class TransformerConfig:
    """Shared transformer dimensions."""

    model_dim: int = 256
    feedforward_dim: int = 1024
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.0

    def validate(self) -> None:
        """Validate transformer dimensions."""
        if self.model_dim % self.num_heads != 0:
            raise ValueError("transformer.model_dim must be divisible by transformer.num_heads")
        if self.num_layers < 1:
            raise ValueError("transformer.num_layers must be positive")


@dataclass
class TrajectoryConfig:
    """Trajectory prediction settings."""

    num_modes: int = 4
    time_horizon: float = 4.0
    interval_length: float = 0.5

    @property
    def num_poses(self) -> int:
        """Number of predicted poses per trajectory."""
        return int(self.time_horizon / self.interval_length)

    def validate(self) -> None:
        """Validate trajectory sampling."""
        if self.num_modes < 1:
            raise ValueError("trajectory.num_modes must be positive")
        if self.time_horizon <= 0 or self.interval_length <= 0:
            raise ValueError("trajectory sampling values must be positive")
        if self.time_horizon % self.interval_length != 0:
            raise ValueError("trajectory.time_horizon must be divisible by trajectory.interval_length")


@dataclass
class LossConfig:
    """Loss weights used to train the released model."""

    trajectory: float = 1.0
    world_model: float = 0.2
    geometry: float = 0.1
    ego_status: float = 0.2

    def validate(self) -> None:
        """Validate non-negative loss weights."""
        values = (self.trajectory, self.world_model, self.geometry, self.ego_status)
        if any(value < 0 for value in values):
            raise ValueError("loss weights must be non-negative")


@dataclass
class DataConfig:
    """Temporal feature schema settings."""

    time_offsets_s: Tuple[float, ...] = (-1.5, 0.0, 2.0, 4.0)
    num_views: int = 3
    scene_tokens_per_view: int = 16
    schema_version: int = 2

    def __post_init__(self) -> None:
        self.time_offsets_s = tuple(self.time_offsets_s)
        self.validate()

    def validate(self) -> None:
        """Validate the released temporal schema."""
        if self.time_offsets_s != (-1.5, 0.0, 2.0, 4.0):
            raise ValueError("released checkpoints require time offsets (-1.5, 0.0, 2.0, 4.0)")
        if self.num_views != 3:
            raise ValueError("released checkpoints require exactly three camera views")
        if self.scene_tokens_per_view < 1:
            raise ValueError("data.scene_tokens_per_view must be positive")
        if self.schema_version != 2:
            raise ValueError("released caches require schema version 2")


@dataclass
class RuntimeConfig:
    """Runtime-only loading behavior."""

    strict_checkpoint: bool = True


@dataclass
class TrainingConfig:
    """Optimizer and target-encoder settings."""

    learning_rate: float = 2e-4
    target_momentum: float = 0.999

    def validate(self) -> None:
        """Validate training hyperparameters."""
        if self.learning_rate <= 0:
            raise ValueError("training.learning_rate must be positive")
        if not 0.0 <= self.target_momentum <= 1.0:
            raise ValueError("training.target_momentum must be in [0, 1]")


@dataclass
class LatentWAMConfig:
    """Top-level Latent WAM configuration.

    The defaults are inferred from the released epoch-84 checkpoint and are
    validated during construction to prevent silent architecture mismatches.
    """

    vision: VisionConfig = field(default_factory=VisionConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    losses: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self) -> None:
        self.transformer.validate()
        self.trajectory.validate()
        self.losses.validate()
        self.data.validate()
        self.training.validate()

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "LatentWAMConfig":
        """Construct a validated config from a JSON-compatible mapping."""
        data_values = dict(values.get("data", {}))
        if "time_offsets_s" in data_values:
            data_values["time_offsets_s"] = tuple(data_values["time_offsets_s"])
        return cls(
            vision=VisionConfig(**values.get("vision", {})),
            geometry=GeometryConfig(**values.get("geometry", {})),
            transformer=TransformerConfig(**values.get("transformer", {})),
            trajectory=TrajectoryConfig(**values.get("trajectory", {})),
            losses=LossConfig(**values.get("losses", {})),
            data=DataConfig(**data_values),
            runtime=RuntimeConfig(**values.get("runtime", {})),
            training=TrainingConfig(**values.get("training", {})),
        )

    @property
    def trajectory_sampling(self) -> Any:
        """NAVSIM trajectory sampling represented by this configuration."""
        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

        return TrajectorySampling(
            time_horizon=self.trajectory.time_horizon,
            interval_length=self.trajectory.interval_length,
        )
