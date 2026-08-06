import numpy as np
import pandas as pd
import pytest

from hmda_seconds import clean, config


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


COUNTY_VALUES = pd.DataFrame(
    {
        "fips": pd.array([6037], dtype="Int64"),
        "year": [2005],
        "hpi": [200.0],
        "county_value": [450_000.0],
    }
)


def test_valid_first_lien_row_passes_and_computes_features():
    df = pd.DataFrame([_base_row()])
    out = clean.clean_frame(df, COUNTY_VALUES)

    assert len(out) == 1
    row = out.iloc[0]
    assert row["fips"] == 6037
    assert row["lien_status"] == 1
    assert row["log_lti"] == pytest.approx(np.log(300.0 / 100.0))
    assert row["log_county_value_to_loan"] == pytest.approx(
        np.log(450_000.0 / 300_000.0)
    )
    assert not row["has_edit_status"]  # edit_status was NaN
    assert not row["loan_below_10k"]
    assert row["loan_type"] == 1
    assert isinstance(row["loan_type"], (int, np.integer))


def test_edit_status_and_small_loan_flags():
    df = pd.DataFrame([_base_row(lien_status=2, edit_status=5, loan_amt=5.0)])
    out = clean.clean_frame(df, COUNTY_VALUES)

    assert len(out) == 1
    row = out.iloc[0]
    assert row["has_edit_status"]
    assert row["edit_status"] == 5
    assert row["loan_below_10k"]


def test_id_vars_are_retained():
    df = pd.DataFrame([_base_row()])
    out = clean.clean_frame(df, COUNTY_VALUES)
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
    out = clean.clean_frame(df, COUNTY_VALUES)
    assert len(out) == 0


def test_county_without_fhfa_match_is_dropped():
    df = pd.DataFrame([_base_row(county_code=99)])  # FIPS 6099 is not in panel.
    out = clean.clean_frame(df, COUNTY_VALUES)
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
    out = clean.clean_frame(df, COUNTY_VALUES)
    assert len(out) == 1
    assert "lien_status" not in out.columns


def test_category_columns_are_pinned_to_full_levels_even_when_absent():
    # A one-row slice necessarily has only one value of every categorical
    # column present. Without pinning (config.CATEGORY_LEVELS), dummy
    # encoding this slice would produce one column per variable instead of
    # the full training-time set -- exactly the bug that would silently
    # misalign features when an estimator encodes one year at a time.
    df = pd.DataFrame([_base_row(purchaser_type=3, loan_type=2)])
    out = clean.clean_frame(df, COUNTY_VALUES)

    for var, categories in config.CATEGORY_LEVELS.items():
        assert isinstance(out[var].dtype, pd.CategoricalDtype)
        assert list(out[var].cat.categories) == categories

    dummy_columns = pd.get_dummies(out[config.CATEGORY_VARS]).shape[1]
    expected_dummy_cols = sum(len(v) for v in config.CATEGORY_LEVELS.values())
    assert dummy_columns == expected_dummy_cols


def test_pin_category_levels_is_defensive_and_supports_a_subset():
    frame = pd.DataFrame({"loan_type": [2], "purchaser_type": [3]})

    pinned = clean.pin_category_levels(frame, ["loan_type"])

    assert not isinstance(frame["loan_type"].dtype, pd.CategoricalDtype)
    assert list(pinned["loan_type"].cat.categories) == config.CATEGORY_LEVELS[
        "loan_type"
    ]
    assert not isinstance(pinned["purchaser_type"].dtype, pd.CategoricalDtype)


def test_load_and_clean_year_tolerates_missing_requested_columns_and_drops_label(
    tmp_path,
):
    yearly_dir = tmp_path / "yearly"
    yearly_dir.mkdir()
    pd.DataFrame([_base_row(extra_unused=7)]).to_parquet(
        yearly_dir / "hmda2005.parquet"
    )

    out = clean.load_and_clean_year(
        2005,
        COUNTY_VALUES,
        yearly_dir=yearly_dir,
        columns=[
            "asof_date",
            "action_taken",
            "loan_purp",
            "occupancy",
            *clean.BASE_VAR_LIST,
            "missing_optional",
        ],
        allow_missing_columns=True,
        label_policy="drop",
    )

    assert len(out) == 1
    assert config.LABEL_VAR not in out


def test_load_and_clean_year_can_require_label(tmp_path):
    yearly_dir = tmp_path / "yearly"
    yearly_dir.mkdir()
    raw = pd.DataFrame([_base_row()]).drop(columns=[config.LABEL_VAR])
    raw.to_parquet(yearly_dir / "hmda2005.parquet")

    with pytest.raises(ValueError, match="missing lien_status"):
        clean.load_and_clean_year(
            2005,
            COUNTY_VALUES,
            yearly_dir=yearly_dir,
            label_policy="require",
        )
