import torch

from navsim.agents.latent_wam.model import TrajectoryHead


def test_trajectory_head_uses_public_output_names() -> None:
    head = TrajectoryHead(num_poses=8, d_ffn=32, d_model=16, num_modes=4)
    queries = torch.zeros((2, 32, 16))
    commands = torch.tensor([0, 3])

    output = head(queries, commands)

    assert set(output) == {"trajectory", "all_trajectories", "mode_logits"}
    assert output["trajectory"].shape == (2, 8, 3)
    assert output["all_trajectories"].shape == (2, 4, 8, 3)
    assert output["mode_logits"].shape == (2, 4)
