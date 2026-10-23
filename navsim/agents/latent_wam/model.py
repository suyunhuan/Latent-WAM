"""Latent WAM model implementation."""

import copy
from typing import Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.models.dinov2 import Dinov2Model

from navsim.agents.latent_wam.cache import LatentWAMFeatures
from navsim.agents.latent_wam.config import LatentWAMConfig
from navsim.agents.latent_wam.mode_select import (
    apply_mode_selection,
    resolve_hybrid_margin,
    resolve_mode_select,
)
from navsim.agents.latent_wam.modules.temporal_world_model import TemporalWorldModel
from navsim.agents.latent_wam.outputs import LossDict, TrainingOutput, TrajectoryPrediction
from navsim.agents.latent_wam.preprocessing import DINO_CROP_SIZE, normalize_dino_images
from navsim.common.enums import StateSE2Index


class EgoStatusHead(nn.Module):
    """Predict command, velocity, and acceleration from a world token."""

    def __init__(self, config: LatentWAMConfig) -> None:
        super().__init__()
        self.cmd_decoder = self._make_head(config, 4)
        self.velocity_decoder = self._make_head(config, 2)
        self.acceleration_decoder = self._make_head(config, 2)

    @staticmethod
    def _make_head(config: LatentWAMConfig, output_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(config.transformer.model_dim, config.transformer.feedforward_dim),
            nn.ReLU(),
            nn.Linear(config.transformer.feedforward_dim, output_dim),
        )

    def forward(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.cmd_decoder(latent),
            self.velocity_decoder(latent),
            self.acceleration_decoder(latent),
        )


class TrajectoryHead(nn.Module):
    """Predict one trajectory per command-conditioned mode."""

    def __init__(self, num_poses: int, d_ffn: int, d_model: int, num_modes: int) -> None:
        super().__init__()
        self.num_poses = num_poses
        self.num_modes = num_modes
        self.d_model = d_model
        self._mlp = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, StateSE2Index.size()),
        )
        self._cls_mlp = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, 1),
        )

    def forward(self, object_queries: torch.Tensor, command: torch.Tensor) -> TrajectoryPrediction:
        poses = self._mlp(object_queries).reshape(-1, self.num_modes, self.num_poses, StateSE2Index.size())
        poses[..., StateSE2Index.HEADING] = poses[..., StateSE2Index.HEADING].tanh() * np.pi
        mode_features = object_queries.reshape(-1, self.num_modes, self.num_poses, self.d_model).mean(dim=2)
        mode_logits = self._cls_mlp(mode_features).squeeze(-1)
        batch_indices = torch.arange(poses.shape[0], device=poses.device)
        trajectory = poses[batch_indices, command.long()]
        return {
            "trajectory": trajectory,
            "all_trajectories": poses,
            "mode_logits": mode_logits,
        }


class LatentWAMModel(nn.Module):
    """Released Latent WAM architecture.

    Args:
        config: Validated model and data configuration.
        vision_encoder: Optional injected DINOv2 encoder for offline tests.
        build_target_encoder: Build the EMA target encoder required for training.
    """

    def __init__(
        self,
        config: LatentWAMConfig,
        vision_encoder: Optional[Dinov2Model] = None,
        build_target_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_poses = config.trajectory.num_poses
        self.num_modes = config.trajectory.num_modes
        self.num_keyval = config.data.num_views * config.data.scene_tokens_per_view + 1

        self.scene_embeds = nn.Parameter(
            torch.randn(
                1,
                len(config.data.time_offsets_s),
                config.data.num_views,
                config.data.scene_tokens_per_view,
                config.vision.feature_dim,
            )
            * 1e-6
        )
        self.vision_encoder = vision_encoder or AutoModel.from_pretrained(config.vision.model_id)
        self.target_dino_encoder: Optional[Dinov2Model]
        if build_target_encoder:
            self.target_dino_encoder = copy.deepcopy(self.vision_encoder)
            self.target_dino_encoder.requires_grad_(False)
        else:
            self.target_dino_encoder = None

        self.dino_projector = nn.Sequential(
            nn.Linear(config.vision.feature_dim, config.transformer.feedforward_dim),
            nn.LayerNorm(config.transformer.feedforward_dim),
            nn.GELU(),
            nn.Dropout(config.transformer.dropout),
            nn.Linear(config.transformer.feedforward_dim, config.transformer.model_dim),
            nn.LayerNorm(config.transformer.model_dim),
        )
        self.geometry_projector = nn.Sequential(
            nn.Linear(config.transformer.model_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, config.geometry.feature_dim),
        )
        self.status_encoder = nn.Linear(8, config.transformer.model_dim)
        self.waypoint_mode_ref = nn.Embedding(self.num_modes * self.num_poses, config.transformer.model_dim)
        self.mode_encoder = nn.Sequential(
            nn.Linear(config.transformer.model_dim, config.transformer.feedforward_dim),
            nn.ReLU(),
            nn.Linear(config.transformer.feedforward_dim, config.transformer.model_dim),
        )
        self.keyval_embedding = nn.Embedding(self.num_keyval, config.transformer.model_dim)
        self.trajectory_queries = nn.Embedding(self.num_modes * self.num_poses, config.transformer.model_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.transformer.model_dim,
            nhead=config.transformer.num_heads,
            dim_feedforward=config.transformer.feedforward_dim,
            dropout=config.transformer.dropout,
            batch_first=True,
        )
        self.trajectory_decoder = nn.TransformerDecoder(decoder_layer, config.transformer.num_layers)
        self.trajectory_head = TrajectoryHead(
            self.num_poses,
            config.transformer.feedforward_dim,
            config.transformer.model_dim,
            self.num_modes,
        )

        num_world_queries = (len(config.data.time_offsets_s) - 1) * self.num_keyval
        self.world_queries = nn.Embedding(num_world_queries, config.transformer.model_dim)
        self.temporal_world_model = TemporalWorldModel(config, use_4d_rope=False)
        self.ego_status_decoder = EgoStatusHead(config)

    def _preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """Transfer cached uint8 images and normalize them on the model device."""
        if images.shape[-3:-1] != DINO_CROP_SIZE:
            raise ValueError(f"model images must already be cropped to {DINO_CROP_SIZE}")
        images = images.to(self.scene_embeds.device, non_blocking=True)
        return normalize_dino_images(images)

    def _encode_with_scene_queries(
        self,
        model: Dinov2Model,
        pixel_values: torch.Tensor,
        scene_queries: torch.Tensor,
    ) -> BaseModelOutputWithPooling:
        embedding_output = model.embeddings(pixel_values, None)
        embedding_output = torch.cat([scene_queries, embedding_output], dim=1)
        encoder_outputs: BaseModelOutput = model.encoder(embedding_output, None, None)
        sequence_output = model.layernorm(encoder_outputs.last_hidden_state)
        sequence_output = self.dino_projector(sequence_output)
        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
        )

    def _encode_frames(
        self,
        pixel_values: torch.Tensor,
        ego_status: torch.Tensor,
        encoder: Dinov2Model,
        scene_start_index: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_frames, num_views = pixel_values.shape[:3]
        scene_queries = self.scene_embeds[:, scene_start_index : scene_start_index + num_frames].repeat(
            batch_size, 1, 1, 1, 1
        )
        scene_queries = scene_queries.reshape(
            batch_size * num_frames * num_views,
            self.config.data.scene_tokens_per_view,
            self.config.vision.feature_dim,
        )
        encoded = self._encode_with_scene_queries(
            encoder,
            pixel_values.reshape(-1, *pixel_values.shape[-3:]),
            scene_queries,
        ).last_hidden_state
        encoded = encoded.reshape(batch_size, num_frames, num_views, -1, self.config.transformer.model_dim)
        scene_tokens = encoded[..., : self.config.data.scene_tokens_per_view, :].reshape(
            batch_size, num_frames, -1, self.config.transformer.model_dim
        )
        status_tokens = self.status_encoder(ego_status)
        keyval = torch.cat([status_tokens[:, :, None], scene_tokens], dim=-2)
        keyval = keyval + self.keyval_embedding.weight[None, None]
        return encoded, keyval, scene_queries

    def _decode_trajectories(
        self,
        keyval: torch.Tensor,
        commands: torch.Tensor,
    ) -> TrajectoryPrediction:
        batch_size, num_frames = keyval.shape[:2]
        queries = self.trajectory_queries.weight[None, None].repeat(batch_size, num_frames, 1, 1)
        mode_reference = self.waypoint_mode_ref.weight.reshape(
            self.num_modes, self.num_poses, self.config.transformer.model_dim
        )
        mode_reference = mode_reference[None, None].repeat(batch_size, num_frames, 1, 1, 1)
        queries = queries + self.mode_encoder(mode_reference).reshape_as(queries)
        decoded = self.trajectory_decoder(
            queries.reshape(-1, *queries.shape[-2:]),
            keyval.reshape(-1, *keyval.shape[-2:]),
        ).reshape(batch_size, num_frames, -1, self.config.transformer.model_dim)
        current_idx = self.config.data.time_offsets_s.index(0.0) if num_frames > 1 else 0
        return self.trajectory_head(decoded[:, current_idx], commands[:, current_idx])

    def predict(self, images: torch.Tensor, ego_status: torch.Tensor) -> TrajectoryPrediction:
        """Predict trajectories from current images and current ego status only.

        Args:
            images: Tensor with shape ``[B, V, H, W, C]``.
            ego_status: Tensor with shape ``[B, 8]``.
        """
        if images.ndim != 5 or ego_status.ndim != 2:
            raise ValueError("predict expects images [B,V,H,W,C] and ego_status [B,8]")
        if images.shape[1] != self.config.data.num_views or images.shape[-1] != 3:
            raise ValueError("predict expects exactly three RGB camera views")
        if ego_status.shape[-1] != 8 or ego_status.shape[0] != images.shape[0]:
            raise ValueError("ego_status must have shape [B,8] and match the image batch")
        pixel_values = self._preprocess_images(images[:, None])
        _, keyval, _ = self._encode_frames(
            pixel_values,
            ego_status[:, None].to(self.scene_embeds.device),
            self.vision_encoder,
            scene_start_index=self.config.data.time_offsets_s.index(0.0),
        )
        commands = ego_status[:, None, :4].argmax(dim=-1).to(self.scene_embeds.device)
        prediction = self._decode_trajectories(keyval, commands)
        strategy = resolve_mode_select(self.config.runtime.mode_select)
        margin = resolve_hybrid_margin(self.config.runtime.hybrid_margin)
        prediction, _disagree = apply_mode_selection(
            prediction, commands[:, 0], strategy, hybrid_margin=margin
        )
        return prediction

    def forward_train(
        self,
        features: LatentWAMFeatures | Mapping[str, torch.Tensor],
    ) -> TrainingOutput:
        """Run the epoch84 teacher-forcing training path."""
        if self.target_dino_encoder is None:
            raise RuntimeError("forward_train requires build_target_encoder=True")
        if isinstance(features, LatentWAMFeatures):
            images = features.images
            geometry = features.geometry
            ego_status = features.ego_status
        else:
            images = features["images"]
            geometry = features["geometry"]
            ego_status = features["ego_status"]
        if images.ndim == 5:
            images, geometry, ego_status = images[None], geometry[None], ego_status[None]
        if images.shape[1:3] != (len(self.config.data.time_offsets_s), self.config.data.num_views):
            raise ValueError("training images must have shape [B,4,3,H,W,C]")
        if geometry.shape[:3] != images.shape[:3] or geometry.shape[-1] != self.config.geometry.feature_dim:
            raise ValueError("geometry features must align with [B,4,3] and use the configured feature dimension")
        if ego_status.shape != (images.shape[0], len(self.config.data.time_offsets_s), 8):
            raise ValueError("training ego_status must have shape [B,4,8]")
        device = self.scene_embeds.device
        ego_status = ego_status.to(device)
        geometry = geometry.to(device)
        pixel_values = self._preprocess_images(images)
        encoded, keyval, scene_queries = self._encode_frames(pixel_values, ego_status, self.vision_encoder)
        commands = ego_status[..., :4].argmax(dim=-1)
        output: TrainingOutput = dict(self._decode_trajectories(keyval, commands))
        current_idx = self.config.data.time_offsets_s.index(0.0)
        output["mode_command_gt"] = commands[:, current_idx]

        with torch.no_grad():
            target_encoded = self._encode_with_scene_queries(
                self.target_dino_encoder,
                pixel_values.reshape(-1, *pixel_values.shape[-3:]),
                scene_queries.detach(),
            ).last_hidden_state
            batch_size, num_frames, num_views = pixel_values.shape[:3]
            target_encoded = target_encoded.reshape(
                batch_size, num_frames, num_views, -1, self.config.transformer.model_dim
            )
            target_scene_tokens = target_encoded[..., : self.config.data.scene_tokens_per_view, :].reshape(
                batch_size, num_frames, -1, self.config.transformer.model_dim
            )
            target_status = self.status_encoder(ego_status).detach()
            target_keyval = torch.cat([target_status[:, :, None], target_scene_tokens], dim=-2)
            target_keyval = target_keyval + self.keyval_embedding.weight[None, None]

        dino_tokens = encoded[..., self.config.data.scene_tokens_per_view + 1 :, :]
        output["geometry_feature_align"] = self.geometry_projector(dino_tokens)
        output["geometry_feature_gt"] = geometry[..., self.config.geometry.patch_start_index :, :]

        batch_size, num_frames = keyval.shape[:2]
        world_keyval = keyval[:, :-1].reshape(batch_size, -1, self.config.transformer.model_dim)
        world_queries = self.world_queries.weight[None].repeat(batch_size, 1, 1)
        token_count = (num_frames - 1) * self.num_keyval
        attention_mask = self.temporal_world_model.get_attention_mask(token_count, self.num_keyval).to(device)
        world_latent = self.temporal_world_model(
            query=world_queries,
            keyval=world_keyval,
            time_frames=num_frames - 1,
            height=self.config.data.num_views,
            width=self.config.data.scene_tokens_per_view,
            tgt_mask=attention_mask,
            memory_mask=attention_mask,
        ).reshape(batch_size, num_frames - 1, self.num_keyval, self.config.transformer.model_dim)
        output["wm_next_latent"] = world_latent
        output["gt_next_latent"] = target_keyval[:, 1:]
        status_latent = world_latent[..., 0, :]
        cmd_pred, velocity_pred, acceleration_pred = self.ego_status_decoder(status_latent)
        output.update(
            {
                "cmd_pred": cmd_pred,
                "velocity_pred": velocity_pred,
                "acceleration_pred": acceleration_pred,
                "cmd_gt": commands[:, 1:],
                "velocity_gt": ego_status[:, 1:, 4:6],
                "acceleration_gt": ego_status[:, 1:, 6:8],
            }
        )
        return output

    def compute_loss(
        self,
        targets: Mapping[str, torch.Tensor],
        predictions: TrainingOutput,
    ) -> LossDict:
        """Compute the loss used by the released epoch84 training path.

        Optional extensions (disabled by default weights):
        - ``losses.mode_classification``: CE on ``mode_logits`` toward WTA ``best_mode``
          and/or GT command (``losses.mode_ce_target``).
        - ``losses.trajectory_lateral_weight``: scale the lateral (y) residual inside
          the trajectory L1 (``1.0`` matches the released isotropic L1).
        """
        ground_truth = targets["trajectory"]
        all_trajectories = predictions["all_trajectories"]
        trajectory_distance = torch.norm(all_trajectories[..., :2] - ground_truth[:, None, :, :2], p=1, dim=-1)
        best_mode = trajectory_distance[:, :, -1].argmin(dim=-1)
        batch_indices = torch.arange(ground_truth.shape[0], device=ground_truth.device)
        best_trajectory = all_trajectories[batch_indices, best_mode]
        residual = (best_trajectory - ground_truth).abs()
        lateral_weight = float(self.config.losses.trajectory_lateral_weight)
        if lateral_weight == 1.0:
            trajectory_loss = residual.mean()
        else:
            axis_weights = residual.new_tensor([1.0, lateral_weight, 1.0])
            trajectory_loss = (residual * axis_weights).mean()
        losses: LossDict = {
            "trajectory_loss": trajectory_loss * self.config.losses.trajectory,
            "wm_loss": F.mse_loss(predictions["wm_next_latent"], predictions["gt_next_latent"])
            * self.config.losses.world_model,
        }
        geometry_align = F.normalize(predictions["geometry_feature_align"], p=2, dim=-1)
        geometry_gt = F.normalize(predictions["geometry_feature_gt"], p=2, dim=-1)
        losses["geometry_loss"] = (
            2 - 2 * F.cosine_similarity(geometry_align, geometry_gt).mean()
        ) * self.config.losses.geometry
        cmd_loss = F.cross_entropy(predictions["cmd_pred"].reshape(-1, 4), predictions["cmd_gt"].reshape(-1))
        velocity_loss = F.mse_loss(predictions["velocity_pred"], predictions["velocity_gt"])
        acceleration_loss = F.mse_loss(predictions["acceleration_pred"], predictions["acceleration_gt"])
        losses["ego_status_loss"] = (cmd_loss + velocity_loss + acceleration_loss) * self.config.losses.ego_status

        mode_ce_weight = float(self.config.losses.mode_classification)
        if mode_ce_weight > 0:
            mode_logits = predictions["mode_logits"]
            ce_target = str(self.config.losses.mode_ce_target)
            if ce_target == "best_mode":
                mode_ce = F.cross_entropy(mode_logits, best_mode.long())
            elif ce_target == "command":
                mode_command = predictions.get("mode_command_gt")
                if mode_command is None:
                    raise ValueError("mode_ce_target='command' requires predictions['mode_command_gt']")
                mode_ce = F.cross_entropy(mode_logits, mode_command.long())
            else:
                mode_command = predictions.get("mode_command_gt")
                if mode_command is None:
                    raise ValueError("mode_ce_target='both' requires predictions['mode_command_gt']")
                mode_ce = 0.5 * (
                    F.cross_entropy(mode_logits, best_mode.long())
                    + F.cross_entropy(mode_logits, mode_command.long())
                )
            losses["mode_classification_loss"] = mode_ce * mode_ce_weight
        return losses

    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        """Update the training-only target encoder using EMA."""
        if self.target_dino_encoder is None:
            return
        for target_param, online_param in zip(self.target_dino_encoder.parameters(), self.vision_encoder.parameters()):
            target_param.data.mul_(momentum).add_(online_param.data, alpha=1.0 - momentum)
