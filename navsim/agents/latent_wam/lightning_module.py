"""PyTorch Lightning training wrapper for Latent WAM."""

import logging
from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch
from torch import Tensor

from navsim.agents.latent_wam.agent import LatentWAMAgent
from navsim.agents.latent_wam.checkpoint import TEMPORAL_ATTENTION_MASK_CHECKPOINT_KEY

logger = logging.getLogger(__name__)


class LatentWAMLightningModule(pl.LightningModule):
    """Run the single teacher-forcing training path used by the released model."""

    def __init__(self, agent: LatentWAMAgent) -> None:
        super().__init__()
        if not agent.build_target_encoder:
            raise ValueError("Latent WAM training requires build_target_encoder=True")
        self.agent = agent
        logger.info("Latent WAM temporal attention mask: %s", agent.config.training.temporal_attention_mask)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Record training behavior without adding keys to the model state dict."""
        checkpoint[TEMPORAL_ATTENTION_MASK_CHECKPOINT_KEY] = self.agent.config.training.temporal_attention_mask

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Reject exact resumes across temporal attention modes, including legacy runs."""
        stored_mode = checkpoint.get(TEMPORAL_ATTENTION_MASK_CHECKPOINT_KEY, "block_diagonal")
        requested_mode = self.agent.config.training.temporal_attention_mask
        if stored_mode != requested_mode:
            raise ValueError(
                "cannot resume Latent WAM with a different temporal attention mask: "
                f"checkpoint={stored_mode!r}, requested={requested_mode!r}. "
                "Start a new run with weights-only initialization instead of resume_ckpt_path."
            )

    def _step(
        self,
        batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
        prefix: str,
    ) -> Tensor:
        features, targets = batch
        predictions = self.agent.forward_train(features)
        losses = self.agent.compute_loss(features, targets, predictions)
        for name, value in losses.items():
            self.log(
                f"{prefix}/{name}",
                value,
                on_step=prefix == "train",
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
        total_loss = torch.stack(tuple(losses.values())).sum()
        self.log(
            f"{prefix}/loss",
            total_loss,
            on_step=prefix == "train",
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return total_loss

    def training_step(
        self,
        batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
        batch_idx: int,
    ) -> Tensor:
        del batch_idx
        return self._step(batch, "train")

    def validation_step(
        self,
        batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
        batch_idx: int,
    ) -> Tensor:
        del batch_idx
        return self._step(batch, "val")

    def on_train_batch_end(self, outputs: object, batch: object, batch_idx: int) -> None:
        del outputs, batch, batch_idx
        self.agent.update_target_encoder()

    def configure_optimizers(self) -> Dict[str, object]:
        self.agent.trainer = self.trainer
        return self.agent.get_optimizers()
