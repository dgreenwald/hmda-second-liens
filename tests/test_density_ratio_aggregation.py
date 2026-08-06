import pandas as pd
import pytest

from hmda_seconds.density_ratio import aggregation


def test_two_stage_means_weight_horizons_equally_and_count_cells():
    cells = pd.DataFrame(
        {
            "candidate": ["a", "a", "a"],
            "horizon": [1, 1, 2],
            "year": [2004, 2005, 2004],
            "score": [0.1, 0.3, 0.8],
        }
    )

    horizons, summary = aggregation.two_stage_horizon_means(
        cells,
        candidate_columns=("candidate",),
        metric_columns=("score",),
        count_column="year",
    )

    assert horizons.set_index("horizon")["score"].to_dict() == pytest.approx(
        {1: 0.2, 2: 0.8}
    )
    assert horizons.set_index("horizon")["n_cells"].to_dict() == {1: 2, 2: 1}
    assert summary["score"].item() == pytest.approx(0.5)
    assert summary["n_horizons"].item() == 2
    assert summary["n_cells"].item() == 3


def test_two_stage_means_reject_incomplete_candidate_horizons():
    cells = pd.DataFrame(
        {
            "candidate": ["a", "a", "b"],
            "horizon": [1, 2, 1],
            "year": [2004, 2004, 2004],
            "score": [0.1, 0.2, 0.3],
        }
    )

    with pytest.raises(ValueError, match="horizon coverage is incomplete"):
        aggregation.two_stage_horizon_means(
            cells,
            candidate_columns=("candidate",),
            metric_columns=("score",),
            count_column="year",
        )


def test_two_stage_means_can_enforce_an_external_horizon_plan():
    cells = pd.DataFrame(
        {"candidate": ["a"], "horizon": [1], "year": [2004], "score": [0.1]}
    )

    with pytest.raises(ValueError, match="horizon coverage is incomplete"):
        aggregation.two_stage_horizon_means(
            cells,
            candidate_columns=("candidate",),
            metric_columns=("score",),
            count_column="year",
            expected_horizons=(1, 2),
        )
