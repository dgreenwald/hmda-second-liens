"""Step 8 historical application and internal plausibility checks."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import clean, config, mixture, model_selection
from .density_ratio import checkpoints, evaluation

REPORTING_START_YEAR = 2004
HISTORICAL_REQUIRED_COLUMNS = [
    "asof_date",
    "action_taken",
    "loan_purp",
    "occupancy",
    *clean.BASE_VAR_LIST,
]


def annual_prediction_summary(
    frame: pd.DataFrame,
    raw_probability: np.ndarray,
    adjusted_probability: np.ndarray,
    mixture_share: float,
) -> dict[str, float | int]:
    """Summarize one application year without retaining loan-level outputs."""
    raw_probability = np.asarray(raw_probability, dtype=float)
    adjusted_probability = np.asarray(adjusted_probability, dtype=float)
    if len(frame) == 0 or len(raw_probability) != len(frame) or len(adjusted_probability) != len(frame):
        raise ValueError("Frame and probability arrays must be nonempty and aligned")
    row = {
        "year": int(frame["year"].iloc[0]),
        "n_model_sample": len(frame),
        "raw_mean_probability": float(raw_probability.mean()),
        "raw_hard_share_050": float((raw_probability >= 0.5).mean()),
        "mixture_share": float(mixture_share),
        "mixture_mean_probability": float(adjusted_probability.mean()),
        "mixture_hard_share_050": float((adjusted_probability >= 0.5).mean()),
        "mixture_mean_minus_share": float(adjusted_probability.mean() - mixture_share),
    }
    if config.LABEL_VAR in frame and frame[config.LABEL_VAR].notna().any():
        labels = frame[config.LABEL_VAR].to_numpy()
        row["actual_second_share"] = float(
            (labels == config.SECOND_LIEN_CLASS).mean()
        )
    else:
        row["actual_second_share"] = np.nan
    return row


def continuity_summary(annual: pd.DataFrame) -> pd.DataFrame:
    """Report the 2003/2004 jump and labeled-boundary discrepancies."""
    indexed = annual.set_index("year")
    if 2003 not in indexed.index or 2004 not in indexed.index:
        raise ValueError("Annual table must contain 2003 and 2004")
    rows = []
    for estimator, column in (
        ("raw_mean_probability", "raw_mean_probability"),
        ("raw_hard_share_050", "raw_hard_share_050"),
        ("mixture_mean_probability", "mixture_mean_probability"),
        ("mixture_hard_share_050", "mixture_hard_share_050"),
    ):
        predicted_2003 = float(indexed.loc[2003, column])
        predicted_2004 = float(indexed.loc[2004, column])
        actual_2004 = float(indexed.loc[2004, "actual_second_share"])
        rows.append(
            {
                "estimator": estimator,
                "predicted_2003": predicted_2003,
                "predicted_2004": predicted_2004,
                "predicted_2004_minus_2003": predicted_2004 - predicted_2003,
                "actual_2004": actual_2004,
                "predicted_2004_minus_actual": predicted_2004 - actual_2004,
                "predicted_2003_minus_actual_2004": predicted_2003 - actual_2004,
            }
        )
    return pd.DataFrame(rows)


def run_historical_plausibility(
    selection_data_dir: str | Path = config.SELECTION_DATA_DIR,
    yearly_dir: str | Path = config.HMDA_YEARLY_DIR,
    output_dir: str | Path = config.TABLE_DIR,
    figure_dir: str | Path = config.FIGURE_DIR,
    raw_model_file: str | Path = config.SELECTED_LOGISTIC_MODEL_FILE,
    fold_model_dir: str | Path = config.MIXTURE_FOLD_MODEL_DIR,
    source_comparison_file: str | Path = (
        config.PUBLIC_DIR / "step8_external_source_comparison.csv"
    ),
    years=tuple(config.APPLY_YEARS),
) -> dict[str, pd.DataFrame]:
    """Apply the frozen final models and render the 1990--2016 share series."""
    selection_data_dir = Path(selection_data_dir)
    yearly_dir = Path(yearly_dir)
    output_dir = Path(output_dir)
    figure_dir = Path(figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    raw = model_selection.load_selected_model(raw_model_file)
    known_path = mixture.known_source_prior_model_path(
        config.TRAIN_YEARS,
        raw.specification,
        raw.regularization_c,
        fold_model_dir,
    )
    known = mixture.load_known_source_prior_model(known_path)
    if raw.transformer.feature_names_ != known.transformer.feature_names_:
        raise RuntimeError("Final raw and mixture feature columns differ")

    annual_file = output_dir / "step8_annual_plausibility.csv"
    annual = checkpoints.read_csv(annual_file)
    requested = tuple(years)
    missing = [year for year in _application_order(requested) if not _year_complete(annual, year)]
    county_values = None
    for year in missing:
        selection_file = selection_data_dir / f"hmda{year}.parquet"
        if selection_file.exists():
            frame = pd.read_parquet(selection_file)
        else:
            if county_values is None:
                county_values = clean.build_county_value_panel(config.APPLY_YEARS)
            frame = _load_and_clean_application_year(
                year, yearly_dir, county_values
            )
        log_ratio = known.log_ratio(frame)
        evaluated = evaluation.adjust_log_ratio(log_ratio)
        estimate = evaluated.mixture_estimate
        adjusted = evaluated.probability
        raw_probability = raw.predict_proba_second_lien(frame)
        row = annual_prediction_summary(
            frame, raw_probability, adjusted, estimate.share
        )
        row.update(
            {
                "mixture_optimizer_converged": estimate.optimizer_converged,
                "mixture_at_boundary": estimate.at_boundary,
                "mixture_em_difference": estimate.share - estimate.em_share,
            }
        )
        annual = checkpoints.replace_rows(
            annual,
            pd.DataFrame([row]),
            annual_file,
            key_columns=("year",),
        )

    annual = annual.loc[annual["year"].isin(requested)].sort_values("year")
    continuity = continuity_summary(annual)
    continuity.to_csv(output_dir / "step8_boundary_continuity.csv", index=False)
    sources = pd.read_csv(source_comparison_file)
    sources.to_csv(
        output_dir / "step8_external_source_comparison.csv", index=False
    )
    render_annual_shares(
        annual, figure_dir / "step8_predicted_actual_shares_1990_2016.pdf"
    )
    return {
        "annual": annual.reset_index(drop=True),
        "continuity": continuity,
        "external_sources": sources,
    }


def render_annual_shares(annual: pd.DataFrame, output_file: str | Path) -> None:
    """Plot adjusted and benchmark annual shares around the reporting boundary."""
    annual = annual.sort_values("year")
    fig, axis = plt.subplots(figsize=(8.0, 4.8))
    axis.plot(
        annual["year"],
        annual["mixture_mean_probability"],
        color="C0",
        linewidth=2.0,
        label="Mixture mean probability",
    )
    axis.plot(
        annual["year"],
        annual["mixture_hard_share_050"],
        color="C0",
        linestyle="--",
        linewidth=1.4,
        label="Mixture hard share (0.5)",
    )
    axis.plot(
        annual["year"],
        annual["raw_mean_probability"],
        color="0.55",
        linestyle=":",
        linewidth=1.5,
        label="Raw logistic mean",
    )
    labeled = annual["actual_second_share"].notna()
    axis.plot(
        annual.loc[labeled, "year"],
        annual.loc[labeled, "actual_second_share"],
        color="black",
        marker="o",
        markersize=3.5,
        linewidth=1.5,
        label="Actual labeled share",
    )
    axis.axvline(2003.5, color="0.35", linestyle="-.", linewidth=1.0)
    axis.text(2003.65, axis.get_ylim()[1] * 0.95, "Reporting boundary", va="top")
    axis.set_xlim(int(annual["year"].min()), int(annual["year"].max()))
    axis.set_ylim(bottom=0)
    axis.set_xlabel("Year")
    axis.set_ylabel("Second-lien origination-count share")
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file)
    plt.close(fig)


def _load_and_clean_application_year(
    year: int, yearly_dir: Path, county_values: pd.DataFrame
) -> pd.DataFrame:
    label_policy = "drop" if year < REPORTING_START_YEAR else "allow"
    return clean.load_and_clean_year(
        year,
        county_values,
        yearly_dir=yearly_dir,
        columns=HISTORICAL_REQUIRED_COLUMNS,
        allow_missing_columns=True,
        label_policy=label_policy,
    )


def _application_order(years: tuple[int, ...]) -> list[int]:
    priority = [2003, 2004]
    return [year for year in priority if year in years] + [
        year for year in years if year not in priority
    ]


def _year_complete(annual: pd.DataFrame, year: int) -> bool:
    return checkpoints.rows_present(annual, {"year": year})
