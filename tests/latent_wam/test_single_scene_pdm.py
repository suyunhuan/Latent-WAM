import pandas as pd
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.planning.script.run_pdm_score_one_stage import create_scene_aggregators


def test_single_scene_pdm_skips_adjacent_frame_aggregation() -> None:
    score = pd.DataFrame(
        {
            "token": ["scene"],
            "ego_simulated_states": [[object()]],
            "two_frame_extended_comfort": [float("nan")],
        }
    )

    aggregated = create_scene_aggregators(
        {},
        score,
        TrajectorySampling(num_poses=40, interval_length=0.1),
    )

    assert aggregated["token"].tolist() == ["scene"]
    assert "ego_simulated_states" not in aggregated.columns
