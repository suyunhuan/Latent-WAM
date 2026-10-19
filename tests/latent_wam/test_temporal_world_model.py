"""Causal dependency and released-behavior regression tests, without model downloads."""

import pytest
import torch

from navsim.agents.latent_wam.config import LatentWAMConfig, TrainingConfig, TransformerConfig
from navsim.agents.latent_wam.modules.temporal_world_model import TemporalWorldModel

BLOCK_SIZE = 49
MODES = ("block_diagonal", "block_causal")


def _model(mode: str) -> TemporalWorldModel:
    torch.manual_seed(17)
    config = LatentWAMConfig(
        transformer=TransformerConfig(model_dim=32, feedforward_dim=64, num_heads=1, num_layers=3),
        training=TrainingConfig(temporal_attention_mask=mode),
    )
    return TemporalWorldModel(config).eval()


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(23)
    return tuple(torch.randn(2, 3 * BLOCK_SIZE, 32, generator=generator) for _ in range(2))


def _forward(model: TemporalWorldModel, query: torch.Tensor, keyval: torch.Tensor) -> torch.Tensor:
    mask = model.get_attention_mask(query.shape[1], BLOCK_SIZE)
    return model(query, keyval, time_frames=3, height=3, width=16, tgt_mask=mask, memory_mask=mask)


def _released_mask(seq_len: int, block_size: int) -> torch.Tensor:
    """Exact pre-change implementation retained solely as a regression oracle."""
    mask = torch.zeros((seq_len, seq_len))
    for i in range(seq_len // block_size):
        start, end = i * block_size, (i + 1) * block_size
        mask[start:end, start:end] = 1
    return mask.masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("blocks", [1, 3])
def test_temporal_mask_allows_exactly_the_expected_blocks(mode: str, blocks: int) -> None:
    mask = _model(mode).get_attention_mask(blocks * BLOCK_SIZE, BLOCK_SIZE)
    expected_blocks = torch.eye(blocks, dtype=torch.bool)
    if mode == "block_causal":
        expected_blocks = torch.tril(torch.ones(blocks, blocks, dtype=torch.bool))
    expected = expected_blocks.repeat_interleave(BLOCK_SIZE, 0).repeat_interleave(BLOCK_SIZE, 1)

    assert mask.shape == (blocks * BLOCK_SIZE, blocks * BLOCK_SIZE)
    assert mask.dtype == torch.float32
    assert torch.equal(mask == 0, expected)
    assert torch.equal(torch.isneginf(mask), ~expected)
    assert torch.isfinite(mask).any(dim=-1).all()


def test_diagonal_mode_matches_released_mask_and_multilayer_output() -> None:
    model = _model("block_diagonal")
    query, keyval = _inputs()
    old_mask = _released_mask(query.shape[1], BLOCK_SIZE)
    assert torch.equal(model.get_attention_mask(query.shape[1], BLOCK_SIZE), old_mask)
    with torch.no_grad():
        old_output = model(query, keyval, 3, 3, 16, tgt_mask=old_mask, memory_mask=old_mask)
        new_output = _forward(model, query, keyval)
    torch.testing.assert_close(new_output, old_output, rtol=0, atol=0)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("future_block", [1, 2])
@pytest.mark.parametrize("changed_input", ["query", "keyval"])
def test_future_inputs_cannot_change_earlier_predictions(mode: str, future_block: int, changed_input: str) -> None:
    model = _model(mode)
    query, keyval = _inputs()
    changed_query, changed_keyval = query.clone(), keyval.clone()
    changed = changed_query if changed_input == "query" else changed_keyval
    begin, end = future_block * BLOCK_SIZE, (future_block + 1) * BLOCK_SIZE
    changed[:, begin:end] += torch.randn_like(changed[:, begin:end]) * 10
    with torch.no_grad():
        baseline = _forward(model, query, keyval)
        perturbed = _forward(model, changed_query, changed_keyval)
    torch.testing.assert_close(perturbed[:, :begin], baseline[:, :begin], rtol=0, atol=0)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("future_block", [1, 2])
def test_earlier_predictions_have_zero_gradient_to_future_inputs(mode: str, future_block: int) -> None:
    model = _model(mode)
    query, keyval = (value.requires_grad_() for value in _inputs())
    end = future_block * BLOCK_SIZE
    output = _forward(model, query, keyval)[:, :end]
    loss = (output * torch.randn_like(output)).sum()
    query_grad, keyval_grad = torch.autograd.grad(loss, (query, keyval))
    assert torch.count_nonzero(query_grad[:, end:]) == 0
    assert torch.count_nonzero(keyval_grad[:, end:]) == 0


@pytest.mark.parametrize("mode", MODES)
def test_only_causal_mode_uses_earlier_observations_for_later_predictions(mode: str) -> None:
    model = _model(mode)
    query, keyval = _inputs()
    changed = keyval.clone()
    changed[:, :BLOCK_SIZE] += torch.randn_like(changed[:, :BLOCK_SIZE]) * 10
    with torch.no_grad():
        baseline = _forward(model, query, keyval)
        perturbed = _forward(model, query, changed)
    for block in (1, 2):
        begin, end = block * BLOCK_SIZE, (block + 1) * BLOCK_SIZE
        if mode == "block_causal":
            assert (perturbed[:, begin:end] - baseline[:, begin:end]).abs().max() > 1e-5
        else:
            torch.testing.assert_close(perturbed[:, begin:end], baseline[:, begin:end], rtol=0, atol=0)


def test_causal_predictions_receive_nonzero_gradients_from_history() -> None:
    model = _model("block_causal")
    query, keyval = _inputs()
    keyval.requires_grad_()
    output = _forward(model, query, keyval)[:, 2 * BLOCK_SIZE :]
    gradient = torch.autograd.grad((output * torch.randn_like(output)).sum(), keyval)[0]
    assert gradient[:, :BLOCK_SIZE].abs().max() > 1e-6


def test_rope_positions_match_ego_view_scene_order_and_strictly_past_sources() -> None:
    model = _model("block_causal")
    getter = model.position_getter
    query_positions = getter(2, 3, 3, 16, torch.device("cpu"), [4, 8, 12]).long()
    key_positions = getter(2, 3, 3, 16, torch.device("cpu"), [1, 4, 8]).long()
    assert query_positions.shape == key_positions.shape == (2, 147, 3)
    for positions, times in ((query_positions, [4, 8, 12]), (key_positions, [1, 4, 8])):
        for block, time in enumerate(times):
            expected = [[time, 0, 0]] + [[time, view, scene] for view in range(1, 4) for scene in range(1, 17)]
            actual = positions[:, block * BLOCK_SIZE : (block + 1) * BLOCK_SIZE]
            assert torch.equal(actual, torch.tensor(expected).expand(2, -1, -1))

    mask = model.get_attention_mask(147, BLOCK_SIZE)
    allowed = mask == 0
    target_times = query_positions[0, :, 0, None]
    source_times = key_positions[0, None, :, 0]
    assert (source_times < target_times)[allowed].all()
    assert torch.isneginf(mask[:BLOCK_SIZE, BLOCK_SIZE : 2 * BLOCK_SIZE]).all()
    assert torch.isneginf(mask[BLOCK_SIZE : 2 * BLOCK_SIZE, 2 * BLOCK_SIZE :]).all()


def test_modes_have_identical_strictly_loadable_parameter_layouts() -> None:
    diagonal, causal = (_model(mode) for mode in MODES)
    old_state = diagonal.state_dict()
    result = causal.load_state_dict(old_state, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    assert old_state.keys() == causal.state_dict().keys()
    assert all(old_state[name].shape == value.shape for name, value in causal.state_dict().items())
