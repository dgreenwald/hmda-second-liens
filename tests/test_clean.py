import numpy as np
import pandas as pd
import pytest

from hmda_seconds import clean


def _base_row(**overrides):
    row = {
        "asof_date": 2005,
        "action_taken": 1,
        "loan_purp": 1,
        "occupancy": 1,
        "lien_status": 1,
        "state_code": 6,  # CA
        "county_code": 37,  # -> fips 6037
        "app_income": 100.0,
        "loan_amt": 300.0,
        "loan_type": 1,
        "purchaser_type": 1,
        "edit_status": np.nan,
        "agency_code": 1,
        "resp_id": 111.0,
        "seq_num": 1,
    }
    row.update(overrides)
    return row


FHFA_BALANCED = pd.DataFrame(
    {"fips": pd.array([6037], dtype="Int64"), "year": [2005], "hpi": [200.0]}
)


def test_valid_first_lien_row_passes_and_computes_features():
    df = pd.DataFrame([_base_row()])
    out = clean.clean_frame(df, FHFA_BALANCED)

    assert len(out) == 1
    row = out.iloc[0]
    assert row["fips"] == 6037
    assert row["lien_status"] == 1
    assert row["log_lti"] == pytest.approx(np.log(300.0 / 100.0))
    assert row["log_ltv"] == pytest.approx(np.log(200.0 / 300.0))
    assert not row["has_edit_status"]  # edit_status was NaN
    assert not row["loan_below_10k"]
    assert row["loan_type"] == 1
    assert isinstance(row["loan_type"], (int, np.integer))


def test_edit_status_and_small_loan_flags():
    df = pd.DataFrame([_base_row(lien_status=2, edit_status=5, loan_amt=5.0)])
    out = clean.clean_frame(df, FHFA_BALANCED)

    assert len(out) == 1
    row = out.iloc[0]
    assert row["has_edit_status"]
    assert row["edit_status"] == 5
    assert row["loan_below_10k"]


def test_id_vars_are_retained():
    df = pd.DataFrame([_base_row()])
    out = clean.clean_frame(df, FHFA_BALANCED)
    assert "resp_id" in out.columns
    assert "seq_num" in out.columns


@pytest.mark.parametrize(
    "overrides",
    [
        {"action_taken": 2},
        {"loan_purp": 3},
        {"occupancy": 2},
        {"lien_status": np.nan},
        {"state_code": 3},  # Puerto Rico, excluded territory code
        {"state_code": 0},  # out of 1-56 range
        {"county_code": 0},
        {"app_income": np.nan},
        {"loan_amt": np.nan},
        {"app_income": 0.5},  # below the >=1 floor
        {"loan_type": 5},  # outside 1-4
    ],
)
def test_sample_restrictions_drop_invalid_rows(overrides):
    df = pd.DataFrame([_base_row(**overrides)])
    out = clean.clean_frame(df, FHFA_BALANCED)
    assert len(out) == 0


def test_county_without_fhfa_match_is_dropped():
    df = pd.DataFrame([_base_row(county_code=99)])  # fips 6099, not in FHFA_BALANCED
    out = clean.clean_frame(df, FHFA_BALANCED)
    assert len(out) == 0


def test_build_balanced_fhfa_panel_requires_full_year_coverage():
    df_fhfa = pd.DataFrame(
        {
            "fips": pd.array([1, 1, 2], dtype="Int64"),
            "year": [2004, 2005, 2004],  # county 2 is missing 2005
            "hpi": [100.0, 110.0, 90.0],
        }
    )
    out = clean.build_balanced_fhfa_panel(df_fhfa, years=[2004, 2005])

    assert set(out["fips"].unique()) == {1}
    assert len(out) == 2  # one row per year for the single balanced county


def test_pre_2004_frame_without_lien_status_is_not_filtered_on_it():
    df = pd.DataFrame([_base_row()]).drop(columns=["lien_status"])
    out = clean.clean_frame(df, FHFA_BALANCED)
    assert len(out) == 1
    assert "lien_status" not in out.columns
