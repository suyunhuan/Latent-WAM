import numpy as np
import pandas as pd
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.evaluate.pdm_score import _apply_human_penalty_filter
from navsim.planning.script.run_pdm_score_one_stage import create_scene_aggregators
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex


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


def test_human_penalty_filter_keeps_derived_metrics_consistent() -> None:
    pdm_result = pd.DataFrame(
        {
            "no_at_fault_collisions": [1.0],
            "drivable_area_compliance": [1.0],
            "driving_direction_compliance": [1.0],
            "traffic_light_compliance": [0.0],
            "ego_progress": [0.4],
            "time_to_collision_within_bound": [1.0],
            "lane_keeping": [0.5],
            "history_comfort": [0.0],
            "multiplicative_metrics_prod": [0.0],
            "weighted_metrics": [np.array([0.4, 1.0, 0.5, 0.0, 0.0])],
            "weighted_metrics_array": [np.array([5.0, 5.0, 2.0, 2.0, 2.0])],
            "pdm_score": [0.0],
        }
    )
    human_result = pd.DataFrame(
        {
            "no_at_fault_collisions": [1.0],
            "drivable_area_compliance": [1.0],
            "driving_direction_compliance": [1.0],
            "traffic_light_compliance": [0.0],
            "ego_progress": [1.0],
            "time_to_collision_within_bound": [1.0],
            "lane_keeping": [1.0],
            "history_comfort": [0.0],
            "multiplicative_metrics_prod": [0.0],
            "weighted_metrics": [np.array([1.0, 1.0, 1.0, 0.0, 0.0])],
            "weighted_metrics_array": [np.array([5.0, 5.0, 2.0, 2.0, 2.0])],
            "pdm_score": [0.0],
        }
    )

    filtered = _apply_human_penalty_filter(pdm_result, human_result)

    assert filtered.at[0, "traffic_light_compliance"] == 1.0
    assert filtered.at[0, "history_comfort"] == 1.0
    assert filtered.at[0, "multiplicative_metrics_prod"] == 1.0
    assert filtered.at[0, "weighted_metrics"][WeightedMetricIndex.HISTORY_COMFORT] == 1.0
    assert filtered.at[0, "pdm_score"] == 0.0
