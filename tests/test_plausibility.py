import numpy as np
import pandas as pd
import pytest

from hmda_seconds import plausibility


def test_annual_prediction_summary_reports_all_release_shares():
    frame = pd.DataFrame(
        {"year": [2004] * 4, "lien_status": [1, 2, 1, 2]}
    )
    raw = np.array([0.1, 0.4, 0.7, 0.8])
    adjusted = np.array([0.05, 0.55, 0.45, 0.95])

    row = plausibility.annual_prediction_summary(
        frame, raw, adjusted, mixture_share=0.5
    )

    assert row["actual_second_share"] == 0.5
    assert row["raw_mean_probability"] == pytest.approx(0.5)
    assert row["raw_hard_share_050"] == 0.5
    assert row["mixture_mean_probability"] == 0.5
    assert row["mixture_hard_share_050"] == 0.5


def test_unlabeled_annual_summary_does_not_invent_actual_share():
    frame = pd.DataFrame({"year": [2003, 2003]})

    row = plausibility.annual_prediction_summary(
        frame, np.array([0.1, 0.2]), np.array([0.2, 0.4]), 0.3
    )

    assert np.isnan(row["actual_second_share"])


def test_continuity_summary_uses_2003_to_2004_difference():
    annual = pd.DataFrame(
        {
            "year": [2003, 2004],
            "actual_second_share": [np.nan, 0.2],
            "raw_mean_probability": [0.1, 0.15],
            "raw_hard_share_050": [0.08, 0.12],
            "mixture_mean_probability": [0.18, 0.21],
            "mixture_hard_share_050": [0.17, 0.19],
        }
    )

    result = plausibility.continuity_summary(annual).set_index("estimator")

    assert result.loc[
        "mixture_mean_probability", "predicted_2004_minus_2003"
    ] == pytest.approx(0.03)
    assert result.loc[
        "mixture_mean_probability", "predicted_2004_minus_actual"
    ] == pytest.approx(0.01)
