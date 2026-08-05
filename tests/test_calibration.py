import numpy as np
import pandas as pd
import pytest

from hmda_seconds import calibration


def test_probability_metrics_reports_raw_scores_and_mean_error():
    y = np.array([False, True, False, True])
    probability = np.array([0.1, 0.2, 0.7, 0.8])

    metrics = calibration.probability_metrics(y, probability)

    assert metrics["n"] == 4
    assert metrics["brier_score"] == pytest.approx(0.295)
    assert metrics["observed_second_share"] == pytest.approx(0.5)
    assert metrics["mean_predicted_second_share"] == pytest.approx(0.45)
    assert metrics["calibration_mean_error"] == pytest.approx(-0.05)
    assert np.isfinite(metrics["calibration_intercept"])
    assert np.isfinite(metrics["calibration_slope"])


def test_reliability_bins_preserve_counts_and_order():
    probability = np.linspace(0.001, 0.999, 100)
    y = np.arange(100) % 3 == 0

    bins = calibration.reliability_bins(y, probability, n_bins=10)

    assert len(bins) == 10
    assert bins["n"].sum() == 100
    assert bins["n"].max() - bins["n"].min() <= 1
    assert bins["mean_predicted_probability"].is_monotonic_increasing
    assert (bins["min_probability"] <= bins["max_probability"]).all()


def test_aggregate_reliability_bins_uses_loan_counts():
    bins = pd.DataFrame(
        {
            "horizon": [1, 1],
            "validation_year": [2004, 2005],
            "probability_bin": [1, 1],
            "n": [1, 3],
            "min_probability": [0.1, 0.2],
            "max_probability": [0.1, 0.4],
            "mean_predicted_probability": [0.1, 0.3],
            "observed_second_share": [0.0, 2 / 3],
        }
    )

    pooled = calibration.aggregate_reliability_bins(bins, ["horizon"])

    assert pooled["n"].item() == 4
    assert pooled["n_cells"].item() == 2
    assert pooled["mean_predicted_probability"].item() == pytest.approx(0.25)
    assert pooled["observed_second_share"].item() == pytest.approx(0.5)


def test_reverse_metric_summary_weights_horizons_equally():
    metrics = pd.DataFrame(
        {
            "horizon": [1, 1, 2],
            "validation_year": [2004, 2005, 2004],
            "n": [10, 20, 30],
            **{
                column: [0.1, 0.3, 0.8]
                for column in calibration.METRIC_COLUMNS
            },
        }
    )

    by_horizon, overall = calibration.aggregate_reverse_metrics(metrics)

    assert by_horizon.set_index("horizon").loc[1, "brier_score"] == pytest.approx(
        0.2
    )
    assert overall["brier_score"].item() == pytest.approx(0.5)
    assert overall["n_cells"].item() == 3


def test_probability_metrics_rejects_single_class_sample():
    with pytest.raises(ValueError, match="both outcome classes"):
        calibration.probability_metrics(
            np.zeros(4, dtype=bool), np.full(4, 0.2)
        )
