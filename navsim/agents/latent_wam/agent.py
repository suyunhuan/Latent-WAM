"""NAVSIM agent wrapper for Latent WAM."""

from pathlib import Path
from typing import Dict, List, Mapping, Optional

import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.latent_wam.config import LatentWAMConfig
from navsim.agents.latent_wam.features import (
    CURRENT_FRAME_INDEX,
    LatentWAMFeatureBuilder,
    LatentWAMTargetBuilder,
    build_camera_sensor_config,
)
from navsim.agents.latent_wam.model import LatentWAMModel
from navsim.agents.latent_wam.outputs import LossDict, TrainingOutput, TrajectoryPrediction
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder


class LatentWAMAgent(AbstractAgent):
    """The released Latent WAM planning agent."""

    def __init__(
        self,
        config: LatentWAMConfig,
        checkpoint_path: Optional[str] = None,
        build_target_encoder: bool = False,
    ) -> None:
        super().__init__(config.trajectory_sampling)
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.build_target_encoder = build_target_encoder
        self.model = LatentWAMModel(config, build_target_encoder=build_target_encoder)
        self.trainer: Optional[pl.Trainer] = None

    def name(self) -> str:
        """Return the public agent name."""
        return "LatentWAMAgent"

    def get_sensor_config(self) -> SensorConfig:
        """Return the current-frame three-camera sensor configuration."""
        return build_camera_sensor_config([CURRENT_FRAME_INDEX])

    def initialize(self) -> None:
        """Strictly load a released safetensors checkpoint when configured."""
        if not self.checkpoint_path:
            return
        checkpoint_path = Path(self.checkpoint_path)
        if checkpoint_path.suffix != ".safetensors":
            raise ValueError("Latent WAM checkpoints must use the released .safetensors format")
        try:
            from safetensors.torch import load_file
        except ImportError as error:
            raise RuntimeError("install safetensors to load released weights") from error
        state_dict = load_file(str(checkpoint_path), device="cpu")
        incompatible = self.model.load_state_dict(state_dict, strict=False)
        missing_keys = list(incompatible.missing_keys)
        if self.model.target_dino_encoder is not None:
            target_missing = [key for key in missing_keys if key.startswith("target_dino_encoder.")]
            if target_missing:
                self.model.target_dino_encoder.load_state_dict(self.model.vision_encoder.state_dict())
                missing_keys = [key for key in missing_keys if key not in target_missing]
        if self.config.runtime.strict_checkpoint and (missing_keys or incompatible.unexpected_keys):
            raise RuntimeError(
                "checkpoint is incompatible with Latent WAM: "
                f"missing={missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
        self.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def forward(self, features: Mapping[str, torch.Tensor]) -> TrajectoryPrediction:
        """Predict trajectories from current-frame features.

        Args:
            features: Batched image and ego-status tensors.

        Returns:
            Selected trajectory, all modal trajectories, and mode logits.
        """
        return self.model.predict(features["images"], features["ego_status"])

    def forward_train(self, features: Mapping[str, torch.Tensor]) -> TrainingOutput:
        """Run the released teacher-forcing training path.

        Args:
            features: Batched temporal images, geometry, and ego status.

        Returns:
            Trajectory and auxiliary world-model predictions.
        """
        return self.model.forward_train(features)

    def compute_loss(
        self,
        features: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
        predictions: TrainingOutput,
        **_: object,
    ) -> LossDict:
        """Compute the released multi-task training losses.

        Args:
            features: Inputs retained for the ``AbstractAgent`` contract.
            targets: Ground-truth trajectory tensors.
            predictions: Outputs returned by :meth:`forward_train`.

        Returns:
            Named weighted loss tensors.
        """
        return self.model.compute_loss(targets, predictions)

    def update_target_encoder(self, momentum: Optional[float] = None) -> None:
        """Update the target DINO encoder with exponential moving average.

        Args:
            momentum: Optional override for the configured EMA momentum.
        """
        value = self.config.training.target_momentum if momentum is None else momentum
        self.model.update_target_encoder(value)

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Return the NAVSIM inference feature builder."""
        return [LatentWAMFeatureBuilder(self.config)]

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Return the NAVSIM trajectory target builder."""
        return [LatentWAMTargetBuilder(self.config)]

    def get_optimizers(self) -> Dict[str, object]:
        """Build the optimizer and warmup/cosine scheduler.

        Returns:
            A Lightning optimizer configuration.

        Raises:
            RuntimeError: If no Lightning trainer is attached.
        """
        if self.trainer is None:
            raise RuntimeError("trainer must be attached before configuring optimizers")
        trainable_parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.Adam(trainable_parameters, lr=self.config.training.learning_rate)
        max_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = max(1, int(0.1 * max_steps))
        cosine_steps = max(1, max_steps - warmup_steps)
        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=1e-6)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def get_training_callbacks(self) -> List[pl.Callback]:
        """Return checkpoint and learning-rate logging callbacks."""
        return [
            pl.callbacks.ModelCheckpoint(every_n_epochs=5, save_top_k=-1),
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
        ]

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        """Predict one local-coordinate trajectory from a NAVSIM input.

        Args:
            agent_input: Four-slot NAVSIM history with current camera images.

        Returns:
            Eight local ``(x, y, heading)`` poses sampled at 0.5 seconds.
        """
        self.eval()
        features = self.get_feature_builders()[0].compute_features(agent_input)
        features = {key: value.unsqueeze(0).to(self.model.scene_embeds.device) for key, value in features.items()}
        with torch.no_grad():
            trajectory = self.forward(features)["trajectory"].squeeze(0).cpu().numpy()
        return Trajectory(trajectory, self.config.trajectory_sampling)
