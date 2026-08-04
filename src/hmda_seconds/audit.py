"""Sample-attrition and FHFA-coverage audits for the HMDA pipeline."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from py_tools.datasets import fhfa

from . import clean, config

AUDIT_COLUMNS = [
    "action_taken",
    "loan_purp",
    "occupancy",
    "state_code",
    "county_code",
    "app_income",
    "loan_amt",
    "loan_type",
]

STAGES = [
    "raw",
    "originated_purchase_owner_occupied",
    "nonmissing_lien_status",
    "valid_geography",
    "positive_income_and_loan_amount",
    "valid_loan_type",
    "matched_fhfa_balanced",
    "finite_model_features",
]


def fhfa_coverage_sets(
    df_fhfa: pd.DataFrame, years: Iterable[int]
) -> tuple[set[int], dict[int, set[int]]]:
    """Return complete-panel and year-specific sets of covered county FIPS."""
    years = list(years)
    work = df_fhfa.loc[
        df_fhfa["year"].isin(years) & df_fhfa["hpi"].notna(),
        ["fips", "year"],
    ].drop_duplicates()
    counts = work.groupby("fips")["year"].nunique()
    balanced = set(counts.index[counts == len(years)].astype(int))
    yearly = {
        year: set(work.loc[work["year"] == year, "fips"].astype(int))
        for year in years
    }
    return balanced, yearly


def audit_frame(
    df: pd.DataFrame,
    year: int,
    balanced_fips: set[int],
    yearly_fips: set[int],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Audit one frame and return stage, class, and FHFA-coverage counts.

    The stage masks are cumulative and reproduce ``clean.clean_frame``'s final
    sample. The lien-status requirement is a no-op when the pre-2004 input has
    no label column.
    """
    n = len(df)
    all_rows = np.ones(n, dtype=bool)
    originated = _mask(
        (df["action_taken"] == 1)
        & (df["loan_purp"] == 1)
        & (df["occupancy"] == 1)
    )
    if config.LABEL_VAR in df:
        label_nonmissing = _mask(df[config.LABEL_VAR].notna())
    else:
        label_nonmissing = all_rows

    valid_geography = _mask(
        df["state_code"].between(1, 56)
        & ~df["state_code"].isin(clean.EXCLUDED_STATE_CODES)
        & (df["county_code"] >= 1)
    )
    positive_amounts = _mask(
        df["app_income"].notna()
        & df["loan_amt"].notna()
        & (df["app_income"] >= 1)
        & (df["loan_amt"] >= 1)
    )
    valid_loan_type = _mask(df["loan_type"].between(1, 4))

    fips = 1000.0 * df["state_code"] + df["county_code"]
    matched_balanced = _mask(fips.isin(balanced_fips))
    matched_yearly = _mask(fips.isin(yearly_fips))

    masks = {
        "raw": all_rows,
        "originated_purchase_owner_occupied": originated,
        "nonmissing_lien_status": originated & label_nonmissing,
    }
    masks["valid_geography"] = masks["nonmissing_lien_status"] & valid_geography
    masks["positive_income_and_loan_amount"] = (
        masks["valid_geography"] & positive_amounts
    )
    masks["valid_loan_type"] = (
        masks["positive_income_and_loan_amount"] & valid_loan_type
    )
    masks["matched_fhfa_balanced"] = masks["valid_loan_type"] & matched_balanced

    # Positive income/loan amount and a nonmissing, positive county value imply
    # finite model ratios. Keep the stage explicit so the audit mirrors the
    # cleaner and can detect a future change in those assumptions.
    masks["finite_model_features"] = masks["matched_fhfa_balanced"]

    stage_counts = pd.DataFrame(
        {"year": year, "stage": STAGES, "n": [int(masks[s].sum()) for s in STAGES]}
    )

    class_rows = []
    if config.LABEL_VAR in df:
        for lien_status in (config.FIRST_LIEN_CLASS, config.SECOND_LIEN_CLASS):
            is_class = _mask(df[config.LABEL_VAR] == lien_status)
            for stage in STAGES:
                class_rows.append(
                    {
                        "year": year,
                        "lien_status": lien_status,
                        "stage": stage,
                        "n": int((masks[stage] & is_class).sum()),
                    }
                )
    class_counts = pd.DataFrame(
        class_rows, columns=["year", "lien_status", "stage", "n"]
    )

    pre_fhfa = masks["valid_loan_type"]
    coverage = {
        "year": year,
        "n_pre_fhfa": int(pre_fhfa.sum()),
        "n_matched_balanced": int((pre_fhfa & matched_balanced).sum()),
        "n_matched_year_specific": int((pre_fhfa & matched_yearly).sum()),
        "_counties_pre_fhfa": set(fips.loc[pre_fhfa].dropna().astype(int)),
        "_counties_matched_balanced": set(
            fips.loc[pre_fhfa & matched_balanced].dropna().astype(int)
        ),
        "_counties_matched_year_specific": set(
            fips.loc[pre_fhfa & matched_yearly].dropna().astype(int)
        ),
    }
    return stage_counts, class_counts, coverage


def audit_year(
    year: int,
    balanced_fips: set[int],
    yearly_fips: set[int],
    yearly_dir: str | Path | None = None,
    batch_size: int = 1_000_000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stream and audit one yearly parquet without loading the full file."""
    if yearly_dir is None:
        yearly_dir = config.HMDA_YEARLY_DIR
    path = Path(yearly_dir) / f"hmda{year}.parquet"
    parquet = pq.ParquetFile(path)
    columns = list(AUDIT_COLUMNS)
    if config.LABEL_VAR in parquet.schema_arrow.names:
        columns.append(config.LABEL_VAR)

    stage_parts = []
    class_parts = []
    coverage_parts = []
    for batch in parquet.iter_batches(columns=columns, batch_size=batch_size):
        stage, classes, coverage = audit_frame(
            batch.to_pandas(), year, balanced_fips, yearly_fips
        )
        stage_parts.append(stage)
        if not classes.empty:
            class_parts.append(classes)
        coverage_parts.append(coverage)

    stage_counts = (
        pd.concat(stage_parts).groupby(["year", "stage"], as_index=False, sort=False)["n"].sum()
    )
    stage_counts["stage"] = pd.Categorical(
        stage_counts["stage"], categories=STAGES, ordered=True
    )
    stage_counts = stage_counts.sort_values(["year", "stage"]).reset_index(drop=True)
    stage_counts = add_retention_rates(stage_counts, ["year"])

    if class_parts:
        class_counts = (
            pd.concat(class_parts)
            .groupby(["year", "lien_status", "stage"], as_index=False, sort=False)["n"]
            .sum()
        )
        class_counts["stage"] = pd.Categorical(
            class_counts["stage"], categories=STAGES, ordered=True
        )
        class_counts = class_counts.sort_values(
            ["year", "lien_status", "stage"]
        ).reset_index(drop=True)
        class_counts = add_retention_rates(class_counts, ["year", "lien_status"])
    else:
        class_counts = pd.DataFrame(
            columns=[
                "year",
                "lien_status",
                "stage",
                "n",
                "n_dropped",
                "retention_from_previous",
                "retention_from_raw",
            ]
        )

    coverage = pd.DataFrame(
        [
            {
                "year": year,
                "n_pre_fhfa": sum(part["n_pre_fhfa"] for part in coverage_parts),
                "n_matched_balanced": sum(
                    part["n_matched_balanced"] for part in coverage_parts
                ),
                "n_matched_year_specific": sum(
                    part["n_matched_year_specific"] for part in coverage_parts
                ),
                "n_counties_pre_fhfa": len(
                    set().union(
                        *(part["_counties_pre_fhfa"] for part in coverage_parts)
                    )
                ),
                "n_counties_matched_balanced": len(
                    set().union(
                        *(part["_counties_matched_balanced"] for part in coverage_parts)
                    )
                ),
                "n_counties_matched_year_specific": len(
                    set().union(
                        *(
                            part["_counties_matched_year_specific"]
                            for part in coverage_parts
                        )
                    )
                ),
            }
        ]
    )
    for suffix in ("balanced", "year_specific"):
        coverage[f"loan_match_rate_{suffix}"] = (
            coverage[f"n_matched_{suffix}"] / coverage["n_pre_fhfa"]
        )
        coverage[f"county_match_rate_{suffix}"] = (
            coverage[f"n_counties_matched_{suffix}"]
            / coverage["n_counties_pre_fhfa"]
        )
    return stage_counts, class_counts, coverage


def run_sample_audit(
    years: Iterable[int] | None = None,
    yearly_dir: str | Path | None = None,
    batch_size: int = 1_000_000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full streamed audit and return its four release tables."""
    if years is None:
        years = config.APPLY_YEARS
    years = list(years)
    df_fhfa = fhfa.load("county", data_dir=f"{config.FHFA_DATA_DIR}/").reset_index()
    df_fhfa["year"] = df_fhfa["date"].dt.year
    df_fhfa["fips"] = df_fhfa["fips"].astype("Int64")
    balanced_fips, yearly_fips = fhfa_coverage_sets(df_fhfa, years)

    stages = []
    classes = []
    coverage = []
    for year in years:
        stage_t, class_t, coverage_t = audit_year(
            year,
            balanced_fips,
            yearly_fips[year],
            yearly_dir=yearly_dir,
            batch_size=batch_size,
        )
        stages.append(stage_t)
        if not class_t.empty:
            classes.append(class_t)
        coverage.append(coverage_t)

    hpi_summary = hpi_level_comparability(df_fhfa, balanced_fips, years)

    return (
        pd.concat(stages, ignore_index=True),
        pd.concat(classes, ignore_index=True) if classes else pd.DataFrame(),
        pd.concat(coverage, ignore_index=True),
        hpi_summary,
    )


def run_county_value_coverage_audit(
    df_county_values: pd.DataFrame,
    years: Iterable[int] | None = None,
    yearly_dir: str | Path | None = None,
    batch_size: int = 1_000_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure exact HMDA coverage overall and by observed lien status."""
    if years is None:
        years = config.APPLY_YEARS
    years = list(years)
    panel_fips = {
        year: set(
            df_county_values.loc[
                df_county_values["year"] == year, "fips"
            ].astype(int)
        )
        for year in years
    }

    rows = []
    class_rows = []
    for year in years:
        _, classes, coverage = audit_year(
            year,
            panel_fips[year],
            panel_fips[year],
            yearly_dir=yearly_dir,
            batch_size=batch_size,
        )
        row = coverage.iloc[0]
        rows.append(
            {
                "year": year,
                "n_pre_county_value": int(row["n_pre_fhfa"]),
                "n_matched_county_value": int(row["n_matched_balanced"]),
                "loan_match_rate_county_value": row["loan_match_rate_balanced"],
                "n_counties_pre_county_value": int(row["n_counties_pre_fhfa"]),
                "n_counties_matched_county_value": int(
                    row["n_counties_matched_balanced"]
                ),
                "county_match_rate_county_value": row[
                    "county_match_rate_balanced"
                ],
            }
        )
        if not classes.empty:
            pre = classes.loc[
                classes["stage"] == "valid_loan_type",
                ["lien_status", "n"],
            ].rename(columns={"n": "n_pre_county_value"})
            matched = classes.loc[
                classes["stage"] == "finite_model_features",
                ["lien_status", "n"],
            ].rename(columns={"n": "n_matched_county_value"})
            by_class = pre.merge(matched, on="lien_status", validate="one_to_one")
            by_class.insert(0, "year", year)
            by_class["loan_match_rate_county_value"] = (
                by_class["n_matched_county_value"]
                / by_class["n_pre_county_value"]
            )
            class_rows.append(by_class)

    classes = pd.concat(class_rows, ignore_index=True) if class_rows else pd.DataFrame()
    return pd.DataFrame(rows), classes


def hpi_level_comparability(
    df_fhfa: pd.DataFrame, balanced_fips: set[int], years: Iterable[int]
) -> pd.DataFrame:
    """Compare native county HPI levels with the common-1990-base series."""
    years = list(years)
    hpi_balanced = df_fhfa.loc[
        df_fhfa["year"].isin(years) & df_fhfa["fips"].isin(balanced_fips)
    ].copy()
    hpi_balanced["native_to_1990_scale"] = (
        hpi_balanced["hpi"] / hpi_balanced["hpi_1990_base"]
    )
    hpi_summary = (
        hpi_balanced
        .groupby("year")
        .agg(
            n_balanced_counties=("fips", "nunique"),
            native_hpi_min=("hpi", "min"),
            native_hpi_median=("hpi", "median"),
            native_hpi_max=("hpi", "max"),
            base_1990_hpi_min=("hpi_1990_base", "min"),
            base_1990_hpi_median=("hpi_1990_base", "median"),
            base_1990_hpi_max=("hpi_1990_base", "max"),
        )
        .reset_index()
    )
    correlations = hpi_balanced.groupby("year").apply(
        _level_correlation, include_groups=False
    )
    hpi_summary["native_base_1990_correlation"] = hpi_summary["year"].map(
        correlations
    )
    scale = hpi_balanced.groupby("fips")["native_to_1990_scale"].first()
    for quantile in (0.0, 0.1, 0.5, 0.9, 1.0):
        name = f"native_to_1990_scale_p{int(100 * quantile):03d}"
        hpi_summary[name] = scale.quantile(quantile)
    return hpi_summary


def add_retention_rates(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Add sequential and raw-sample retention measures to ordered stages."""
    out = df.copy()
    previous = out.groupby(group_cols, observed=True)["n"].shift()
    raw = out.groupby(group_cols, observed=True)["n"].transform("first")
    out["n_dropped"] = (previous - out["n"]).fillna(0).astype("int64")
    out["retention_from_previous"] = (out["n"] / previous).fillna(1.0)
    out["retention_from_raw"] = out["n"] / raw
    return out


def _mask(values: pd.Series) -> np.ndarray:
    return values.fillna(False).to_numpy(dtype=bool)


def _level_correlation(df: pd.DataFrame) -> float:
    if df["hpi"].nunique() <= 1 or df["hpi_1990_base"].nunique() <= 1:
        return np.nan
    return float(df["hpi"].corr(df["hpi_1990_base"]))
