"""Weights-only training startup must stay separate from exact Lightning resume."""

import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from transformers import Dinov2Config, Dinov2Model

from navsim.agents.latent_wam.agent import LatentWAMAgent
from navsim.agents.latent_wam.checkpoint import LatentWAMModelCheckpoint, save_model_safetensors
from navsim.agents.latent_wam.config import (
    GeometryConfig,
    LatentWAMConfig,
    TrainingConfig,
    TransformerConfig,
    VisionConfig,
)
from navsim.agents.latent_wam.lightning_module import LatentWAMLightningModule
from navsim.planning.script import run_training


@pytest.fixture(autouse=True)
def _local_vision_encoder(monkeypatch):
    def build(*args, **kwargs):
        return Dinov2Model(
            Dinov2Config(
                hidden_size=32,
                num_hidden_layers=1,
                num_attention_heads=4,
                intermediate_size=64,
                image_size=224,
                patch_size=14,
            )
        )

    monkeypatch.setattr("navsim.agents.latent_wam.model.AutoModel.from_pretrained", build)


def _agent(mode, checkpoint_path=None, build_target_encoder=True, learning_rate=2e-4):
    return LatentWAMAgent(
        LatentWAMConfig(
            vision=VisionConfig(feature_dim=32),
            geometry=GeometryConfig(feature_dim=8),
            transformer=TransformerConfig(model_dim=32, feedforward_dim=64, num_heads=1, num_layers=1),
            training=TrainingConfig(temporal_attention_mask=mode, learning_rate=learning_rate),
        ),
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        build_target_encoder=build_target_encoder,
    )


def _optimizers(agent):
    agent.trainer = SimpleNamespace(estimated_stepping_batches=20)
    configured = agent.get_optimizers()
    return configured["optimizer"], configured["lr_scheduler"]["scheduler"]


def _save_weights(agent, path, optimizer=None, scheduler=None):
    weights = {key: value.detach().clone() for key, value in agent.model.state_dict().items()}
    if path.suffix == ".safetensors":
        save_file(weights, path)
    else:
        torch.save(
            {
                "state_dict": {"agent.model." + key: value for key, value in weights.items()},
                "optimizer_states": [optimizer.state_dict()] if optimizer else [],
                "lr_schedulers": [scheduler.state_dict()] if scheduler else [],
                "global_step": 17,
                "epoch": 6,
                "latent_wam_temporal_attention_mask": agent.config.training.temporal_attention_mask,
            },
            path,
        )


@pytest.mark.parametrize(
    "source_mode,target_mode", [("block_diagonal", "block_causal"), ("block_causal", "block_diagonal")]
)
@pytest.mark.parametrize("include_target", [False, True])
@pytest.mark.parametrize("suffix", [".ckpt", ".safetensors"])
def test_cross_mode_initialization_loads_weights_with_fresh_optimizer_and_scheduler(
    tmp_path, source_mode, target_mode, include_target, suffix
):
    source = _agent(source_mode, build_target_encoder=include_target, learning_rate=0.01)
    if include_target:
        with torch.no_grad():
            next(source.model.target_dino_encoder.parameters()).add_(0.25)
    old_optimizer, old_scheduler = _optimizers(source)
    source.model.scene_embeds.square().sum().backward()
    old_optimizer.step()
    old_scheduler.step()
    assert old_optimizer.state
    old_scheduler_state = copy.deepcopy(old_scheduler.state_dict())
    path = tmp_path / ("old_weights" + suffix)
    _save_weights(source, path, old_optimizer, old_scheduler)

    target = _agent(target_mode, checkpoint_path=path)
    reference_optimizer, reference_scheduler = _optimizers(target)
    reference_optimizer_state = copy.deepcopy(reference_optimizer.state_dict())
    reference_scheduler_state = copy.deepcopy(reference_scheduler.state_dict())
    module = run_training.build_lightning_module(target, resume_ckpt_path=None)
    assert isinstance(module, LatentWAMLightningModule)
    assert module.global_step == 0
    for key, expected in source.model.state_dict().items():
        torch.testing.assert_close(target.model.state_dict()[key], expected, rtol=0, atol=0)
    if not include_target:
        for key, expected in target.model.vision_encoder.state_dict().items():
            torch.testing.assert_close(target.model.target_dino_encoder.state_dict()[key], expected, rtol=0, atol=0)
    assert {parameter.device.type for parameter in target.parameters()} == {"cpu"}
    assert target.config.training.temporal_attention_mask == target_mode
    new_optimizer, new_scheduler = _optimizers(target)
    assert not new_optimizer.state
    assert new_optimizer.state_dict() == reference_optimizer_state
    assert new_scheduler.state_dict() == reference_scheduler_state
    assert new_scheduler.state_dict() != old_scheduler_state


@pytest.mark.parametrize("resume_path", [None, "old_run.ckpt"])
def test_scratch_and_exact_resume_skip_weights_initialization(monkeypatch, resume_path):
    agent = _agent("block_diagonal")
    initialize = Mock(side_effect=AssertionError("unexpected weights initialization"))
    monkeypatch.setattr(agent, "initialize", initialize)
    assert isinstance(run_training.build_lightning_module(agent, resume_path), LatentWAMLightningModule)
    initialize.assert_not_called()


def test_weights_initialization_and_exact_resume_are_mutually_exclusive(monkeypatch):
    agent = _agent("block_causal", checkpoint_path="old_weights.safetensors")
    initialize = Mock(side_effect=AssertionError("must reject before reading weights"))
    monkeypatch.setattr(agent, "initialize", initialize)
    with pytest.raises(ValueError, match="mutually exclusive"):
        run_training.build_lightning_module(agent, resume_ckpt_path="old_run.ckpt")
    initialize.assert_not_called()


@pytest.mark.parametrize("placement", [None, torch.device("cpu")])
def test_initialize_preserves_default_inference_placement_and_can_force_cpu(tmp_path, monkeypatch, placement):
    source = _agent("block_diagonal", build_target_encoder=False)
    path = tmp_path / "weights.safetensors"
    save_file({key: value.detach().clone() for key, value in source.model.state_dict().items()}, path)
    target = _agent("block_causal", checkpoint_path=path)
    move = Mock(return_value=target)
    monkeypatch.setattr(target, "to", move)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    if placement is None:
        target.initialize()
    else:
        target.initialize(device=placement)
    move.assert_called_once_with(torch.device("cuda") if placement is None else placement)


@pytest.mark.parametrize("resume_path", [None, "old_run.ckpt"])
@pytest.mark.parametrize("suffix", [".ckpt", ".safetensors"])
def test_training_entry_point_selects_weights_initialization_or_exact_resume(
    tmp_path, monkeypatch, resume_path, suffix
):
    source = _agent("block_diagonal", build_target_encoder=False)
    path = tmp_path / ("old_weights" + suffix)
    expected = {key: value.detach().clone() for key, value in source.model.state_dict().items()}
    _save_weights(source, path)
    target = _agent("block_causal", checkpoint_path=path if resume_path is None else None)
    initialize = Mock(wraps=target.initialize)
    monkeypatch.setattr(target, "initialize", initialize)
    fits = []

    class Trainer:
        def __init__(self, **kwargs):
            pass

        def fit(self, model, train_dataloaders, val_dataloaders, ckpt_path):
            assert ckpt_path == resume_path
            assert model.global_step == 0
            if resume_path is None:
                for key, value in expected.items():
                    torch.testing.assert_close(model.agent.model.state_dict()[key], value, rtol=0, atol=0)
            optimizer, scheduler = _optimizers(model.agent)
            assert not optimizer.state
            assert scheduler.last_epoch == 0
            fits.append(model)

    monkeypatch.setattr(run_training, "instantiate", lambda cfg: target)
    monkeypatch.setattr(run_training, "pin_experiment_uid", lambda uid: None)
    monkeypatch.setattr(run_training, "configure_rank_logging", lambda *args: None)
    monkeypatch.setattr(run_training, "distributed_rank", lambda: 0)

    # Keep dataset/sampler mechanics out of this startup test while exercising the real main function.
    class Dataset(list):
        sample_locality = [(0, 0)]

    monkeypatch.setattr(run_training, "LatentWAMCacheDataset", lambda *args, **kwargs: Dataset([0]))
    monkeypatch.setattr(run_training, "LocalityAwareDistributedSampler", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_training, "DataLoader", lambda data, **kwargs: data)
    monkeypatch.setattr(run_training.pl, "Trainer", Trainer)
    cfg = OmegaConf.create(
        dict(
            experiment_uid="new_block_causal_run",
            output_dir=str(tmp_path / "new_run"),
            seed=0,
            agent={},
            use_cache_without_dataset=True,
            force_cache_computation=False,
            cache_path=str(tmp_path / "cache"),
            train_logs=["train"],
            val_logs=["val"],
            train_test_split={"data_split": "trainval"},
            dataloader={"params": {}},
            trainer={"params": {}},
            resume_ckpt_path=resume_path,
        )
    )
    run_training.main.__wrapped__(cfg)
    assert len(fits) == 1
    if resume_path is None:
        initialize.assert_called_once_with(device=torch.device("cpu"))
    else:
        initialize.assert_not_called()


def test_other_agents_keep_their_existing_training_wrapper():
    agent = torch.nn.Linear(1, 1)
    agent.initialize = Mock(side_effect=AssertionError("must not change other agent startup"))
    module = run_training.build_lightning_module(agent)
    assert isinstance(module, run_training.AgentLightningModule)
    agent.initialize.assert_not_called()


@pytest.mark.parametrize("suffix", [".ckpt", ".safetensors"])
def test_invalid_weights_fail_before_training_module_is_built(tmp_path, suffix):
    source = _agent("block_diagonal", build_target_encoder=False)
    weights = {key: value.detach().clone() for key, value in source.model.state_dict().items()}
    del weights["world_queries.weight"]
    path = tmp_path / ("incompatible" + suffix)
    if suffix == ".ckpt":
        torch.save({"state_dict": {"agent.model." + key: value for key, value in weights.items()}}, path)
    else:
        save_file(weights, path)
    with pytest.raises(RuntimeError, match="checkpoint is incompatible"):
        run_training.build_lightning_module(_agent("block_causal", checkpoint_path=path), None)


@pytest.mark.parametrize("suffix", [".ckpt", ".safetensors"])
def test_real_lightning_weights_only_run_starts_at_step_zero(tmp_path, suffix):
    source = _agent("block_diagonal", build_target_encoder=False)
    path = tmp_path / ("old_weights" + suffix)
    _save_weights(source, path)
    module = run_training.build_lightning_module(_agent("block_causal", checkpoint_path=path))
    observed = []

    class FreshState(pl.Callback):
        def on_train_start(self, trainer, pl_module):
            assert trainer.global_step == 0
            assert not trainer.optimizers[0].state
            assert trainer.lr_scheduler_configs[0].scheduler.last_epoch == 0
            assert pl_module.agent.config.training.temporal_attention_mask == "block_causal"
            observed.append(True)

    features = {
        "images": torch.zeros(1, 4, 3, 224, 448, 3, dtype=torch.uint8),
        "geometry": torch.randn(1, 4, 3, 519, 8),
        "ego_status": torch.zeros(1, 4, 8),
    }
    features["ego_status"][..., 0] = 1
    loader = DataLoader([(features, {"trajectory": torch.zeros(1, 8, 3)})], batch_size=None)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[FreshState()],
        default_root_dir=tmp_path / "new_run",
    )
    trainer.fit(module, train_dataloaders=loader, ckpt_path=None)
    assert observed == [True]
    assert trainer.global_step == 1
    checkpoint_path = tmp_path / "new_run.ckpt"
    trainer.save_checkpoint(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert checkpoint["latent_wam_temporal_attention_mask"] == "block_causal"
    assert checkpoint["global_step"] == 1


@pytest.mark.parametrize("suffix", [".ckpt", ".safetensors"])
def test_inference_ignores_only_valid_complete_ema_weights(tmp_path, suffix):
    source = _agent("block_diagonal")
    path = tmp_path / ("weights" + suffix)
    _save_weights(source, path)
    target = _agent("block_causal", checkpoint_path=path, build_target_encoder=False)
    target.initialize(device=torch.device("cpu"))
    for key, value in target.model.state_dict().items():
        torch.testing.assert_close(value, source.model.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("training", [False, True])
def test_partial_ema_is_rejected_instead_of_silently_reinitialized(tmp_path, training):
    source = _agent("block_diagonal")
    weights = {key: value.detach().clone() for key, value in source.model.state_dict().items()}
    del weights[next(key for key in weights if key.startswith("target_dino_encoder."))]
    path = tmp_path / "partial_ema.safetensors"
    save_file(weights, path)
    target = _agent("block_causal", checkpoint_path=path, build_target_encoder=training)
    with pytest.raises(RuntimeError, match="incomplete or incompatible target_dino_encoder"):
        target.initialize(device=torch.device("cpu"))


def test_export_preserves_ema_for_cross_mode_training_initialization(tmp_path):
    source = _agent("block_diagonal")
    with torch.no_grad():
        next(source.model.target_dino_encoder.parameters()).add_(0.25)
    path = tmp_path / "training_weights.safetensors"
    save_model_safetensors(source.model, path)
    target = _agent("block_causal", checkpoint_path=path)
    run_training.build_lightning_module(target)
    for key, value in source.model.state_dict().items():
        torch.testing.assert_close(target.model.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("enabled", [False, True])
def test_agent_export_config_controls_checkpoint_callback(enabled):
    agent = _agent("block_diagonal")
    agent.config.training.export_safetensors = enabled
    callback = agent.get_training_callbacks()[0]
    assert isinstance(callback, LatentWAMModelCheckpoint)
    assert callback.export_safetensors is enabled
    assert callback.save_weights_only is False
