import numpy as np
import pandas as pd
import pytest

from hmda_seconds import county_values


def test_annualize_zillow_requires_complete_year():
    dates = pd.date_range("2000-01-31", periods=14, freq="ME")
    df = pd.DataFrame(
        {
            "fips": ["01001"] * len(dates),
            "date": dates,
            "zhvi": np.arange(len(dates), dtype=float) + 100_000,
        }
    )

    out = county_values.annualize_zillow_county(df)

    assert list(out["year"]) == [2000]
    assert out["n_zhvi_months"].item() == 12
    assert out["zhvi"].item() == pytest.approx(df["zhvi"].iloc[:12].mean())


def _scaling_inputs():
    fhfa = pd.DataFrame(
        {
            "fips": [1001, 1001, 1001, 1003],
            "year": [2015, 2016, 2017, 2017],
            "hpi": [100.0, 110.0, 120.0, 200.0],
        }
    )
    zillow = pd.DataFrame(
        {
            "fips": [1001, 1001, 1001, 1003],
            "year": [2015, 2016, 2017, 2017],
            "zhvi": [200_000.0, 220_000.0, 240_000.0, 300_000.0],
            "n_zhvi_months": [12, 12, 12, 12],
        }
    )
    return fhfa, zillow


def test_scale_estimators_recover_exact_proportional_series():
    fhfa, zillow = _scaling_inputs()
    scales, overlap = county_values.estimate_county_scales(fhfa, zillow)
    county = scales.set_index("fips").loc[1001]

    assert len(overlap) == 4
    assert county["n_overlap_years"] == 3
    assert county["scale_geometric"] == pytest.approx(2000.0)
    assert county["scale_ols_origin"] == pytest.approx(2000.0)
    assert county["scale_median_log_ratio"] == pytest.approx(2000.0)
    assert county["scale_anchor_2017"] == pytest.approx(2000.0)
    assert county["log_rmse_geometric"] == pytest.approx(0.0)
    assert county["unrestricted_log_slope"] == pytest.approx(1.0)


def test_build_panel_applies_support_cutoff_and_scale():
    fhfa, zillow = _scaling_inputs()
    scales, _ = county_values.estimate_county_scales(fhfa, zillow)

    panel = county_values.build_county_value_panel(
        fhfa, scales, years=[2015, 2016, 2017], min_overlap_years=3
    )

    assert set(panel["fips"]) == {1001}
    assert list(panel["county_value"]) == pytest.approx(
        [200_000.0, 220_000.0, 240_000.0]
    )


def test_build_panel_rejects_unknown_method():
    fhfa, zillow = _scaling_inputs()
    scales, _ = county_values.estimate_county_scales(fhfa, zillow)

    with pytest.raises(ValueError, match="Unknown scaling method"):
        county_values.build_county_value_panel(
            fhfa, scales, years=[2017], method="not-a-method"
        )


def test_support_summary_reports_each_method():
    fhfa, zillow = _scaling_inputs()
    scales, _ = county_values.estimate_county_scales(fhfa, zillow)

    out = county_values.support_summary(scales, minimums=[1, 3])

    assert list(out["n_counties"]) == [2, 1]
    assert list(out["n_counties_geometric"]) == [2, 1]
    assert list(out["n_counties_anchor_2017"]) == [2, 1]
