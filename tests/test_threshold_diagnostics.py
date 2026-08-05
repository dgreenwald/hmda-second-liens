import numpy as np
import pandas as pd
import pytest

from hmda_seconds import threshold_diagnostics


def test_threshold_metrics_uses_canonical_inclusive_cutoff():
    y = np.array([False, True, True, False])
    probability = np.array([0.1, 0.5, 0.8, 0.9])

    metrics = threshold_diagnostics.threshold_metrics(y, probability)

    assert metrics["true_positive"] == 2
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 0
    assert metrics["precision_second"] == pytest.approx(2 / 3)
    assert metrics["recall_second"] == 1


def test_fixed_grid_precision_recall_counts_match_direct_classification():
    y = np.array([False, True, True, False, True])
    probability = np.array([0.1, 0.2, 0.5, 0.8, 0.9])
    thresholds = np.array([0.0, 0.5, 1.0])

    curve = threshold_diagnostics.precision_recall_counts(
        y, probability, thresholds
    ).set_index("threshold")

    assert curve.loc[0.5, "true_positive"] == 2
    assert curve.loc[0.5, "false_positive"] == 1
    assert curve.loc[0.5, "false_negative"] == 1
    assert curve.loc[0.5, "precision"] == pytest.approx(2 / 3)
    assert curve.loc[0.5, "recall"] == pytest.approx(2 / 3)
    assert curve.loc[1.0, "precision"] == 1
    assert curve.loc[1.0, "recall"] == 0


def test_subgroup_metrics_include_predeclared_dimensions_and_counts():
    n = 100
    frame = pd.DataFrame(
        {
            "loan_type": np.tile([1, 2, 3, 4], 25),
            "purchaser_type": np.tile(np.arange(10), 10),
            "state_code": np.tile([6, 12, 25, 17], 25),
            "log_lti": np.linspace(-2, 2, n),
            "log_county_value_to_loan": np.linspace(1, 3, n),
        }
    )
    y = np.arange(n) % 3 == 0
    probability = np.linspace(0.01, 0.99, n)

    groups = threshold_diagnostics.subgroup_metrics(frame, y, probability)

    assert set(groups["subgroup_dimension"]) == {
        "loan_type",
        "purchaser_type",
        "region",
        "log_lti_target_decile",
        "log_county_value_to_loan_target_decile",
    }
    for dimension in groups["subgroup_dimension"].unique():
        assert groups.loc[groups["subgroup_dimension"] == dimension, "n"].sum() == n
    assert "average_precision" not in groups


def test_subgroup_aggregation_does_not_require_ranking_metric():
    cells = pd.DataFrame(
        {
            "estimator": ["known_source_prior_mixture"],
            "subgroup_dimension": ["loan_type"],
            "subgroup_value": ["1"],
            "horizon": [1],
            "validation_year": [2004],
            "n": [100],
            **{
                column: [0.5]
                for column in threshold_diagnostics.SUBGROUP_METRIC_COLUMNS
            },
        }
    )

    horizons, summary = threshold_diagnostics.aggregate_subgroup_metrics(cells)

    assert horizons["n_cells"].item() == 1
    assert summary["precision_second"].item() == 0.5
