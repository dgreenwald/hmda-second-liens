"""HMDA sample restrictions and feature construction.
Ports ``clean_data``/``load_and_clean`` from the original ``classify_seconds.py``
research script. Two changes from the original, both recorded in
MIGRATION_PLAN.md:

- ``resp_id``/``seq_num`` are retained (the original dropped them) so a
  released predicted-lien-status crosswalk can be joined back to a
  same-vintage raw LAR file.
- County FHFA indexes are put on a Zillow dollar scale before forming the
  county-value-to-loan feature; the historical native-HPI ratio was not a
  comparable price-level measure across counties.

The lien_status label was checked against real 2004-2007 data and found to
already be strictly binary ({1, 2}) within this sample restriction, so unlike
the plan's original open question, no extra filter is needed here.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import numpy as np
import pandas as pd
from py_tools.datasets import fhfa, hmda

from . import config, county_values

BASE_VAR_LIST = [
    "year",
    "agency_code",
    "loan_type",
    "loan_amt",
    "app_income",
    "purchaser_type",
    "edit_status",
    "state_code",
    "county_code",
    "lien_status",
    *config.ID_VARS,
]

RAW_COLUMN_ALIASES = {
    "year": "asof_date",
    "resp_id": "respondent_id",
}
RAW_INPUT_COLUMNS = list(
    dict.fromkeys(
        [
            "asof_date",
            "action_taken",
            "loan_purp",
            "occupancy",
            *[
                RAW_COLUMN_ALIASES.get(variable, variable)
                for variable in BASE_VAR_LIST
                if variable != "year"
            ],
        ]
    )
)
NUMERIC_INPUT_COLUMNS = (
    "year",
    "action_taken",
    "loan_purp",
    "occupancy",
    "lien_status",
    "state_code",
    "county_code",
    "app_income",
    "loan_amt",
    "loan_type",
    "purchaser_type",
    "edit_status",
)

# HMDA state codes for territories/non-states to exclude (matches the
# original script): Puerto Rico, Guam, American Samoa, Virgin Islands,
# Northern Mariana Islands.
EXCLUDED_STATE_CODES = [3, 7, 14, 43, 52]


def load_fhfa_county_hpi(data_dir=None) -> pd.DataFrame:
    """Load the FHFA county all-transactions HPI, indexed by (fips, year)."""
    if data_dir is None:
        data_dir = config.FHFA_DATA_DIR
    df_fhfa = fhfa.load("county", data_dir=f"{data_dir}/").reset_index()
    df_fhfa["year"] = df_fhfa["date"].dt.year
    df_fhfa["fips"] = df_fhfa["fips"].astype("Int64")
    return df_fhfa[["fips", "year", "hpi"]]


def build_balanced_fhfa_panel(df_fhfa: pd.DataFrame, years) -> pd.DataFrame:
    """Restrict to counties with a complete HPI panel over ``years``.

    A county missing the HPI for even one year in the application window is
    dropped entirely (from every year), rather than silently interpolated,
    so the sample composition is stable across years.
    """
    years = list(years)
    df_wide = pd.pivot(df_fhfa, index="fips", columns="year", values="hpi")
    ix_balanced = np.all(pd.notnull(df_wide[years]), axis=1)
    df_wide = df_wide.loc[ix_balanced, years].reset_index()
    return pd.melt(
        df_wide, id_vars=["fips"], value_vars=years, var_name="year", value_name="hpi"
    )


def build_county_value_panel(
    years=None,
    method: str = "geometric",
    min_overlap_years: int = config.ZILLOW_MIN_OVERLAP_YEARS,
    zillow_vintage: str = config.ZILLOW_VINTAGE,
) -> pd.DataFrame:
    """Build the year-specific FHFA panel scaled to Zillow dollar levels."""
    if years is None:
        years = config.APPLY_YEARS
    df_fhfa = load_fhfa_county_hpi()
    df_zillow = county_values.load_zillow_county_zhvi(vintage=zillow_vintage)
    df_zillow_annual = county_values.annualize_zillow_county(df_zillow)
    scales, _ = county_values.estimate_county_scales(df_fhfa, df_zillow_annual)
    panel = county_values.build_county_value_panel(
        df_fhfa,
        scales,
        years,
        method=method,
        min_overlap_years=min_overlap_years,
    )
    return panel[["fips", "year", "hpi", "county_value"]]


def clean_frame(df_t: pd.DataFrame, df_county_values: pd.DataFrame) -> pd.DataFrame:
    """Apply sample restrictions and construct model features for one year."""
    df_t = df_t.rename(
        {"asof_date": "year", "respondent_id": "resp_id"}, axis=1
    )
    for variable in NUMERIC_INPUT_COLUMNS:
        if variable in df_t:
            df_t[variable] = pd.to_numeric(df_t[variable], errors="coerce")

    ix = (
        (df_t["action_taken"] == 1)
        & (df_t["loan_purp"] == 1)
        & (df_t["occupancy"] == 1)
    )
    if "lien_status" in df_t:
        ix = ix & pd.notnull(df_t["lien_status"])
    ix = (
        ix
        & df_t["state_code"].between(1, 56)
        & (~df_t["state_code"].isin(EXCLUDED_STATE_CODES))
    )
    ix = ix & (df_t["county_code"] >= 1)
    ix = ix & pd.notnull(df_t["app_income"]) & pd.notnull(df_t["loan_amt"])
    ix = ix & (df_t["app_income"] >= 1) & (df_t["loan_amt"] >= 1)
    ix = ix & df_t["loan_type"].between(1.0, 4.0)

    var_list = [var for var in BASE_VAR_LIST if var in df_t]
    df_t = df_t.loc[ix, var_list].copy()

    df_t["loan_type"] = df_t["loan_type"].astype(int)

    df_t["fips"] = (1000.0 * df_t["state_code"] + df_t["county_code"]).astype("Int64")
    df_t["state_code"] = df_t["state_code"].astype("Int64")
    df_t = pd.merge(df_t, df_county_values, on=["fips", "year"], how="inner")

    df_t["log_lti"] = np.log(df_t["loan_amt"] / df_t["app_income"])
    df_t["log_county_value_to_loan"] = np.log(
        df_t["county_value"] / (1000.0 * df_t["loan_amt"])
    )
    df_t.dropna(subset=config.CONTINUOUS_VARS, inplace=True)

    df_t["has_edit_status"] = pd.notnull(df_t["edit_status"])
    df_t["edit_status"] = df_t["edit_status"].fillna(0).astype(int)

    df_t["loan_below_10k"] = df_t["loan_amt"] < 10

    df_t = pin_category_levels(df_t)

    return df_t.drop(columns=["county_code"])


def pin_category_levels(
    frame: pd.DataFrame, variables: Iterable[str] | None = None
) -> pd.DataFrame:
    """Reapply canonical levels without mutating the caller's frame.

    This remains necessary at encoding boundaries because parquet round trips
    and caller-created subsets may lose pandas categorical metadata.
    """
    if variables is None:
        variables = config.CATEGORY_LEVELS
    pinned = frame.copy(deep=False)
    for variable in variables:
        if variable in config.CATEGORY_LEVELS:
            pinned[variable] = pd.Categorical(
                frame[variable], categories=config.CATEGORY_LEVELS[variable]
            )
    return pinned


def load_and_clean_year(
    year: int,
    df_county_values: pd.DataFrame,
    hmda_data_dir=None,
    *,
    source: str = "auto",
    columns: Iterable[str] | None = None,
    allow_missing_columns: bool = False,
    label_policy: Literal["allow", "drop", "require"] = "allow",
) -> pd.DataFrame:
    """Read one annual extract with an explicit label/schema policy, then clean it."""
    if hmda_data_dir is None:
        hmda_data_dir = config.HMDA_DATA_DIR
    if label_policy not in {"allow", "drop", "require"}:
        raise ValueError(f"Unknown label_policy {label_policy!r}")
    requested = RAW_INPUT_COLUMNS if columns is None else list(dict.fromkeys(columns))
    requested = [RAW_COLUMN_ALIASES.get(column, column) for column in requested]
    if allow_missing_columns:
        requested = [column for column in requested if column in RAW_INPUT_COLUMNS]
    selected_columns = list(dict.fromkeys(requested))
    if label_policy == "drop" or (label_policy == "allow" and year < 2004):
        selected_columns = [
            column for column in selected_columns if column != config.LABEL_VAR
        ]
    df_t = hmda.load(
        year=year,
        source=source,
        data_dir=hmda_data_dir,
        columns=selected_columns,
    )
    if label_policy == "drop":
        df_t = df_t.drop(columns=[config.LABEL_VAR], errors="ignore")
    elif label_policy == "require" and config.LABEL_VAR not in df_t:
        raise ValueError(
            f"HMDA {year} ({source}) is missing {config.LABEL_VAR}"
        )
    return clean_frame(df_t, df_county_values)
