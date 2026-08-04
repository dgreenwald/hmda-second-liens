import pandas as pd
import pytest

from hmda_seconds import audit


def _audit_frame():
    return pd.DataFrame(
        {
            "action_taken": [1, 2, 1, 1, 1, 1, 1],
            "loan_purp": [1, 1, 1, 1, 1, 1, 1],
            "occupancy": [1, 1, 1, 1, 1, 1, 1],
            "state_code": [1, 1, 3, 1, 1, 1, 1],
            "county_code": [1, 1, 1, 1, 2, 3, 1],
            "app_income": [100, 100, 100, 0, 100, 100, 100],
            "loan_amt": [200, 200, 200, 200, 200, 200, 200],
            "loan_type": [1, 1, 1, 1, 5, 1, 1],
            "lien_status": [1, 1, 2, 2, 1, 2, pd.NA],
        }
    )


def test_audit_frame_reports_cumulative_attrition():
    stages, classes, coverage = audit.audit_frame(
        _audit_frame(), year=2004, balanced_fips={1003}, yearly_fips={1001, 1003}
    )

    counts = stages.set_index("stage")["n"]
    assert counts.to_dict() == {
        "raw": 7,
        "originated_purchase_owner_occupied": 6,
        "nonmissing_lien_status": 5,
        "valid_geography": 4,
        "positive_income_and_loan_amount": 3,
        "valid_loan_type": 2,
        "matched_fhfa_balanced": 1,
        "finite_model_features": 1,
    }
    assert coverage["n_pre_fhfa"] == 2
    assert coverage["n_matched_balanced"] == 1
    assert coverage["n_matched_year_specific"] == 2

    second = classes.loc[classes["lien_status"] == 2].set_index("stage")["n"]
    assert second.loc["raw"] == 3
    assert second.loc["finite_model_features"] == 1


def test_audit_frame_does_not_require_absent_pre_2004_label():
    df = _audit_frame().drop(columns="lien_status")
    stages, classes, _ = audit.audit_frame(
        df, year=2003, balanced_fips={1003}, yearly_fips={1001, 1003}
    )

    counts = stages.set_index("stage")["n"]
    assert counts.loc["nonmissing_lien_status"] == counts.loc[
        "originated_purchase_owner_occupied"
    ]
    assert classes.empty


def test_fhfa_coverage_sets_distinguish_balanced_from_year_specific():
    hpi = pd.DataFrame(
        {
            "fips": [1001, 1001, 1003, 1003, 1005],
            "year": [2000, 2001, 2000, 2001, 2001],
            "hpi": [100.0, 101.0, 90.0, 92.0, 80.0],
        }
    )

    balanced, yearly = audit.fhfa_coverage_sets(hpi, [2000, 2001])

    assert balanced == {1001, 1003}
    assert yearly == {2000: {1001, 1003}, 2001: {1001, 1003, 1005}}


def test_add_retention_rates():
    df = pd.DataFrame(
        {
            "year": [2000, 2000, 2000],
            "stage": ["raw", "middle", "final"],
            "n": [100, 80, 40],
        }
    )
    out = audit.add_retention_rates(df, ["year"])

    assert list(out["n_dropped"]) == [0, 20, 40]
    assert list(out["retention_from_previous"]) == pytest.approx([1.0, 0.8, 0.5])
    assert list(out["retention_from_raw"]) == pytest.approx([1.0, 0.8, 0.4])


def test_hpi_level_comparability_detects_county_specific_native_scaling():
    hpi = pd.DataFrame(
        {
            "fips": [1001, 1001, 1003, 1003],
            "year": [1990, 1991, 1990, 1991],
            "hpi": [100.0, 110.0, 200.0, 240.0],
            "hpi_1990_base": [100.0, 110.0, 100.0, 120.0],
        }
    )

    out = audit.hpi_level_comparability(hpi, {1001, 1003}, [1990, 1991])

    assert out.loc[out["year"] == 1990, "base_1990_hpi_min"].item() == 100.0
    assert out["native_to_1990_scale_p000"].iloc[0] == pytest.approx(1.0)
    assert out["native_to_1990_scale_p100"].iloc[0] == pytest.approx(2.0)


def test_county_value_coverage_audit_reports_class_rates(monkeypatch):
    panel = pd.DataFrame({"fips": [1001], "year": [2004]})

    def fake_audit_year(year, balanced_fips, yearly_fips, **kwargs):
        assert year == 2004
        assert balanced_fips == yearly_fips == {1001}
        classes = pd.DataFrame(
            {
                "lien_status": [1, 2, 1, 2],
                "stage": [
                    "valid_loan_type",
                    "valid_loan_type",
                    "finite_model_features",
                    "finite_model_features",
                ],
                "n": [80, 20, 76, 19],
            }
        )
        coverage = pd.DataFrame(
            {
                "n_pre_fhfa": [100],
                "n_matched_balanced": [95],
                "loan_match_rate_balanced": [0.95],
                "n_counties_pre_fhfa": [2],
                "n_counties_matched_balanced": [1],
                "county_match_rate_balanced": [0.5],
            }
        )
        return pd.DataFrame(), classes, coverage

    monkeypatch.setattr(audit, "audit_year", fake_audit_year)
    overall, classes = audit.run_county_value_coverage_audit(panel, years=[2004])

    assert overall["loan_match_rate_county_value"].item() == pytest.approx(0.95)
    rates = classes.set_index("lien_status")["loan_match_rate_county_value"]
    assert rates.loc[1] == pytest.approx(0.95)
    assert rates.loc[2] == pytest.approx(0.95)
