"""Scale FHFA county HPIs to dollar levels using county Zillow ZHVI."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from py_tools.datasets import zillow

from . import config

SCALING_METHODS = {
    "geometric": "scale_geometric",
    "ols_origin": "scale_ols_origin",
    "median_log_ratio": "scale_median_log_ratio",
    "anchor_2017": "scale_anchor_2017",
}


def load_zillow_county_zhvi(
    vintage: str = config.ZILLOW_VINTAGE,
) -> pd.DataFrame:
    """Load the pinned county ZHVI vintage from ``py_tools``."""
    return zillow.load_county(vintage=vintage).reset_index()


def annualize_zillow_county(
    df_zillow: pd.DataFrame,
    required_months: int = 12,
) -> pd.DataFrame:
    """Average monthly smoothed/SA county ZHVI into complete calendar years."""
    work = df_zillow.reset_index() if "date" not in df_zillow else df_zillow.copy()
    work["date"] = pd.to_datetime(work["date"])
    work["year"] = work["date"].dt.year
    work["fips"] = pd.to_numeric(work["fips"], errors="coerce").astype("Int64")
    work["zhvi"] = pd.to_numeric(work["zhvi"], errors="coerce")
    annual = (
        work.dropna(subset=["fips", "zhvi"])
        .groupby(["fips", "year"], as_index=False)
        .agg(zhvi=("zhvi", "mean"), n_zhvi_months=("zhvi", "size"))
    )
    return annual.loc[annual["n_zhvi_months"] >= required_months].reset_index(
        drop=True
    )


def estimate_county_scales(
    df_fhfa: pd.DataFrame,
    df_zillow_annual: pd.DataFrame,
    anchor_year: int = config.ZILLOW_ANCHOR_YEAR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate Zillow-dollar-per-FHFA-point scales for every county.

    Returns one county-level diagnostic row and the annual overlap frame used
    in estimation. The primary scale is the geometric mean of ZHVI/HPI.
    """
    fhfa = df_fhfa[["fips", "year", "hpi"]].copy()
    fhfa["fips"] = pd.to_numeric(fhfa["fips"], errors="coerce").astype("Int64")
    overlap = fhfa.merge(
        df_zillow_annual[["fips", "year", "zhvi", "n_zhvi_months"]],
        on=["fips", "year"],
        how="inner",
        validate="one_to_one",
    )
    overlap = overlap.loc[(overlap["hpi"] > 0) & (overlap["zhvi"] > 0)].copy()
    overlap["log_hpi"] = np.log(overlap["hpi"])
    overlap["log_zhvi"] = np.log(overlap["zhvi"])
    overlap["log_ratio"] = overlap["log_zhvi"] - overlap["log_hpi"]

    rows = [
        _estimate_one_county(group, anchor_year)
        for _, group in overlap.groupby("fips", sort=True, observed=True)
    ]
    scales = pd.DataFrame(rows)
    if not scales.empty:
        scales["fips"] = scales["fips"].astype("Int64")
    return scales, overlap


def build_county_value_panel(
    df_fhfa: pd.DataFrame,
    scales: pd.DataFrame,
    years: Iterable[int],
    method: str = "geometric",
    min_overlap_years: int = config.ZILLOW_MIN_OVERLAP_YEARS,
) -> pd.DataFrame:
    """Apply a county scale to every available annual FHFA observation."""
    try:
        scale_var = SCALING_METHODS[method]
    except KeyError as exc:
        supported = ", ".join(SCALING_METHODS)
        raise ValueError(f"Unknown scaling method {method!r}; choose {supported}") from exc

    years = list(years)
    eligible = scales.loc[
        (scales["n_overlap_years"] >= min_overlap_years)
        & scales[scale_var].notna(),
        ["fips", "n_overlap_years", scale_var],
    ]
    panel = df_fhfa.loc[
        df_fhfa["year"].isin(years) & (df_fhfa["hpi"] > 0),
        ["fips", "year", "hpi"],
    ].merge(eligible, on="fips", how="inner", validate="many_to_one")
    panel["county_value"] = panel["hpi"] * panel[scale_var]
    panel["scaling_method"] = method
    return panel.sort_values(["fips", "year"]).reset_index(drop=True)


def support_summary(
    scales: pd.DataFrame, minimums: Iterable[int] = (1, 3, 5)
) -> pd.DataFrame:
    """Summarize county support and scale availability at candidate cutoffs."""
    rows = []
    for minimum in minimums:
        row = {
            "min_overlap_years": minimum,
            "n_counties": int((scales["n_overlap_years"] >= minimum).sum()),
        }
        for method, scale_var in SCALING_METHODS.items():
            row[f"n_counties_{method}"] = int(
                (
                    (scales["n_overlap_years"] >= minimum)
                    & scales[scale_var].notna()
                ).sum()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _estimate_one_county(group: pd.DataFrame, anchor_year: int) -> dict:
    group = group.sort_values("year")
    hpi = group["hpi"].to_numpy(dtype=float)
    zhvi = group["zhvi"].to_numpy(dtype=float)
    log_hpi = group["log_hpi"].to_numpy(dtype=float)
    log_zhvi = group["log_zhvi"].to_numpy(dtype=float)
    log_ratio = group["log_ratio"].to_numpy(dtype=float)

    mean_log_ratio = float(log_ratio.mean())
    scale_geometric = float(np.exp(mean_log_ratio))
    scale_ols_origin = float(np.dot(hpi, zhvi) / np.dot(hpi, hpi))
    scale_median = float(np.exp(np.median(log_ratio)))
    anchor = group.loc[group["year"] == anchor_year, "log_ratio"]
    scale_anchor = float(np.exp(anchor.iloc[0])) if len(anchor) == 1 else np.nan

    if len(group) >= 2 and np.var(log_hpi) > 0:
        unrestricted_slope = float(
            np.cov(log_hpi, log_zhvi, ddof=0)[0, 1] / np.var(log_hpi)
        )
        ratio_time_trend = float(
            np.cov(group["year"], log_ratio, ddof=0)[0, 1]
            / np.var(group["year"])
        )
    else:
        unrestricted_slope = np.nan
        ratio_time_trend = np.nan

    return {
        "fips": int(group["fips"].iloc[0]),
        "n_overlap_years": len(group),
        "first_overlap_year": int(group["year"].min()),
        "last_overlap_year": int(group["year"].max()),
        "scale_geometric": scale_geometric,
        "scale_ols_origin": scale_ols_origin,
        "scale_median_log_ratio": scale_median,
        "scale_anchor_2017": scale_anchor,
        "log_rmse_geometric": float(
            np.sqrt(np.mean((log_ratio - mean_log_ratio) ** 2))
        ),
        "log_ratio_sd": float(np.std(log_ratio, ddof=0)),
        "log_ratio_time_trend": ratio_time_trend,
        "unrestricted_log_slope": unrestricted_slope,
    }
