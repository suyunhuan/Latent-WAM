import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.latent_wam.agent import LatentWAMAgent
from navsim.agents.latent_wam.cache import stage_cache
from navsim.agents.latent_wam.dataset import LatentWAMCacheDataset
from navsim.agents.latent_wam.lightning_module import LatentWAMLightningModule
from navsim.agents.latent_wam.sampler import LocalityAwareDistributedSampler
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.script.training_runtime import configure_rank_logging, distributed_rank, pin_experiment_uid
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def build_lightning_module(agent: AbstractAgent, resume_ckpt_path: Optional[str] = None) -> pl.LightningModule:
    """Initialize training weights or defer exact resume to Lightning.

    Args:
        agent: Instantiated planning agent.
        resume_ckpt_path: Full Lightning checkpoint for an exact resume, if any.

    Returns:
        The agent's training module, with fresh optimizer state for weights-only startup.

    Raises:
        ValueError: If both weights-only initialization and exact resume are requested.
    """
    if isinstance(agent, LatentWAMAgent):
        if agent.checkpoint_path:
            if resume_ckpt_path:
                raise ValueError(
                    "agent.checkpoint_path and resume_ckpt_path are mutually exclusive: "
                    "use weights-only initialization for a new run, or exact resume for an existing run"
                )
            logger.info("Initializing Latent WAM model weights only from %s on CPU", agent.checkpoint_path)
            agent.initialize(device=torch.device("cpu"))
        return LatentWAMLightningModule(agent=agent)
    return AgentLightningModule(agent=agent)


def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names if log_name in cfg.train_logs
        ]
    else:
        train_scene_filter.log_names = cfg.train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs]
    else:
        val_scene_filter.log_names = cfg.val_logs

    data_path = Path(cfg.navsim_log_path)
    original_sensor_path = Path(cfg.original_sensor_path)

    train_scene_loader = SceneLoader(
        original_sensor_path=original_sensor_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    val_scene_loader = SceneLoader(
        original_sensor_path=original_sensor_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_data, val_data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """
    pin_experiment_uid(str(cfg.experiment_uid))
    rank = distributed_rank()
    configure_rank_logging(Path(cfg.output_dir), rank)
    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)
    if isinstance(agent, LatentWAMAgent) and not cfg.use_cache_without_dataset:
        raise ValueError("Latent WAM training requires caches; set use_cache_without_dataset=true")

    logger.info("Building Lightning Module")
    lightning_module = build_lightning_module(agent, resume_ckpt_path=cfg.get("resume_ckpt_path", None))

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
            cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"
        if isinstance(agent, LatentWAMAgent):
            cache_path = Path(cfg.cache_path)
            if cfg.get("cache_staging_path"):
                cache_path = stage_cache(cache_path, Path(cfg.cache_staging_path))
            shard_cache_capacity = int(cfg.get("cache_shard_lru", 8))
            allowed_logs = set(cfg.train_logs) | set(cfg.val_logs)
            validate_headers = bool(cfg.get("validate_cache_headers", True))
            train_data = LatentWAMCacheDataset(
                cache_path,
                log_names=cfg.train_logs,
                shard_cache_capacity=shard_cache_capacity,
                config=agent.config,
                dataset_split=str(cfg.train_test_split.data_split),
                allowed_logs=allowed_logs,
                validate_headers=validate_headers,
            )
            val_data = LatentWAMCacheDataset(
                cache_path,
                log_names=cfg.val_logs,
                shard_cache_capacity=shard_cache_capacity,
                config=agent.config,
                dataset_split=str(cfg.train_test_split.data_split),
                allowed_logs=allowed_logs,
                validate_headers=validate_headers,
            )
        else:
            train_data = CacheOnlyDataset(
                cache_path=cfg.cache_path,
                feature_builders=agent.get_feature_builders(),
                target_builders=agent.get_target_builders(),
                log_names=cfg.train_logs,
            )
            val_data = CacheOnlyDataset(
                cache_path=cfg.cache_path,
                feature_builders=agent.get_feature_builders(),
                target_builders=agent.get_target_builders(),
                log_names=cfg.val_logs,
            )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    logger.info("Building Datasets")
    dataloader_parameters = dict(cfg.dataloader.params)
    trainer_parameters = dict(cfg.trainer.params)
    if isinstance(agent, LatentWAMAgent):
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        global_rank = int(os.environ.get("RANK", "0"))
        train_sampler = LocalityAwareDistributedSampler(
            train_data.sample_locality,
            mode=str(cfg.get("cache_sampler_mode", "global")),
            shuffle=True,
            seed=int(cfg.seed),
            rank=global_rank,
            world_size=world_size,
        )
        val_sampler = LocalityAwareDistributedSampler(
            val_data.sample_locality,
            mode="global",
            shuffle=False,
            seed=int(cfg.seed),
            rank=global_rank,
            world_size=world_size,
        )
        train_dataloader = DataLoader(train_data, **dataloader_parameters, sampler=train_sampler, shuffle=False)
        val_dataloader = DataLoader(val_data, **dataloader_parameters, sampler=val_sampler, shuffle=False)
        trainer_parameters["use_distributed_sampler"] = False
    else:
        train_dataloader = DataLoader(train_data, **dataloader_parameters, shuffle=True)
        val_dataloader = DataLoader(val_data, **dataloader_parameters, shuffle=False)
    logger.info("Num training samples: %d", len(train_data))
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")
    trainer = pl.Trainer(**trainer_parameters, callbacks=agent.get_training_callbacks())

    logger.info("Starting Training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=cfg.get("resume_ckpt_path", None),
    )


if __name__ == "__main__":
    main()
