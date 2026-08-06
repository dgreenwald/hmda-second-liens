import numpy as np
import pandas as pd

from hmda_seconds.density_ratio import diagnostics, evaluation


def test_evaluate_cell_builds_common_outputs_and_extensions():
    y_second = np.array([False, True, False, True])
    probability = np.array([0.1, 0.2, 0.7, 0.8])
    metrics = evaluation.evaluate_sample(y_second, probability)

    cell = diagnostics.evaluate_cell(
        y_second,
        probability,
        metadata={"validation_year": 2004},
        n_bins=2,
        metrics=metrics,
        additional_metrics={"mixture_share": 0.4},
        extension_records={"tails": {"log_ratio_max": 3.0}},
    )

    assert cell.metrics["brier_score"].item() == metrics["brier_score"]
    assert cell.metrics["mixture_share"].item() == 0.4
    assert cell.bins["n"].sum() == 4
    assert cell.extensions["tails"].to_dict("records") == [
        {"validation_year": 2004, "log_ratio_max": 3.0}
    ]


def test_reliability_bins_matches_public_calibration_schema():
    bins = diagnostics.reliability_bins(
        np.array([False, True, False, True]),
        np.array([0.1, 0.2, 0.7, 0.8]),
        n_bins=2,
    )

    assert list(bins) == [
        "probability_bin",
        "n",
        "min_probability",
        "max_probability",
        "mean_predicted_probability",
        "observed_second_share",
    ]
    pd.testing.assert_series_equal(
        bins["n"], pd.Series([2, 2], name="n", dtype=np.int64)
    )
