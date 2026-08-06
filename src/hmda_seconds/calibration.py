"""Calibration diagnostics for the selected logistic estimator."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config, model_selection
from .density_ratio import evaluation
from .density_ratio import folds as temporal_folds

DEFAULT_N_BINS = 10
METRIC_COLUMNS = [
    "brier_score",
    "log_loss",
    "observed_second_share",
    "mean_predicted_second_share",
    "calibration_mean_error",
    "calibration_intercept",
    "calibration_slope",
]


# Compatibility alias for callers using the former public location.
probability_metrics = evaluation.evaluate_sample


def reliability_bins(
    y_second: np.ndarray,
    probability: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
) -> pd.DataFrame:
    """Build approximate equal-count validation-probability bins."""
    y = np.asarray(y_second, dtype=bool)
    probability = np.asarray(probability, dtype=float)
    _validate_probability_inputs(y, probability)
    if n_bins < 2:
        raise ValueError("n_bins must be at least two")

    quantiles = np.quantile(probability, np.linspace(0.0, 1.0, n_bins + 1))
    interior = np.unique(quantiles[1:-1])
    assignment = np.searchsorted(interior, probability, side="right")
    effective_bins = len(interior) + 1
    count = np.bincount(assignment, minlength=effective_bins)
    probability_sum = np.bincount(
        assignment, weights=probability, minlength=effective_bins
    )
    observed_sum = np.bincount(
        assignment, weights=y.astype(float), minlength=effective_bins
    )
    minimum = np.full(effective_bins, np.inf)
    maximum = np.full(effective_bins, -np.inf)
    np.minimum.at(minimum, assignment, probability)
    np.maximum.at(maximum, assignment, probability)
    keep = count > 0
    return pd.DataFrame(
        {
            "probability_bin": np.arange(1, effective_bins + 1)[keep],
            "n": count[keep].astype(np.int64),
            "min_probability": minimum[keep],
            "max_probability": maximum[keep],
            "mean_predicted_probability": probability_sum[keep] / count[keep],
            "observed_second_share": observed_sum[keep] / count[keep],
        }
    )


def aggregate_reliability_bins(
    bins: pd.DataFrame, group_columns: list[str]
) -> pd.DataFrame:
    """Pool binned counts within requested diagnostic groups."""
    work = bins.copy()
    work["predicted_sum"] = work["mean_predicted_probability"] * work["n"]
    work["observed_sum"] = work["observed_second_share"] * work["n"]
    groups = [*group_columns, "probability_bin"]
    pooled = (
        work.groupby(groups, as_index=False)
        .agg(
            n=("n", "sum"),
            min_probability=("min_probability", "min"),
            max_probability=("max_probability", "max"),
            predicted_sum=("predicted_sum", "sum"),
            observed_sum=("observed_sum", "sum"),
            n_cells=("validation_year", "nunique"),
        )
        .sort_values(groups)
    )
    pooled["mean_predicted_probability"] = pooled["predicted_sum"] / pooled["n"]
    pooled["observed_second_share"] = pooled["observed_sum"] / pooled["n"]
    return pooled.drop(columns=["predicted_sum", "observed_sum"])


def aggregate_reverse_metrics(
    metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average cells within horizon, then horizons with equal weight."""
    by_horizon = (
        metrics.groupby("horizon", as_index=False)
        .agg(
            **{column: (column, "mean") for column in METRIC_COLUMNS},
            n_cells=("validation_year", "size"),
            n_loans=("n", "sum"),
        )
        .sort_values("horizon")
    )
    overall = pd.DataFrame(
        [
            {
                **{column: by_horizon[column].mean() for column in METRIC_COLUMNS},
                "n_horizons": by_horizon["horizon"].nunique(),
                "n_cells": len(metrics),
                "weighting": "equal_within_horizon_then_equal_across_horizons",
            }
        ]
    )
    return by_horizon, overall


def aggregate_forward_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Average forward-validation cells with equal weight across years."""
    return pd.DataFrame(
        [
            {
                **{column: metrics[column].mean() for column in METRIC_COLUMNS},
                "n_cells": len(metrics),
                "weighting": "equal_across_validation_years",
            }
        ]
    )


def run_calibration_diagnostics(
    data_dir: str | Path = config.SELECTION_DATA_DIR,
    output_dir: str | Path = config.TABLE_DIR,
    figure_dir: str | Path = config.FIGURE_DIR,
    model_file: str | Path = config.SELECTED_LOGISTIC_MODEL_FILE,
    n_bins: int = DEFAULT_N_BINS,
) -> dict[str, pd.DataFrame]:
    """Run reverse-fold and forward-regime diagnostics without recalibration."""
    output_dir = Path(output_dir)
    figure_dir = Path(figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    data_by_year = model_selection.load_selection_years(data_dir=data_dir)
    selected = model_selection.load_selected_model(model_file)
    specification = selected.specification
    regularization_c = selected.regularization_c

    reverse_metrics_file = output_dir / "logistic_calibration_reverse_metrics.csv"
    reverse_bins_file = output_dir / "logistic_calibration_reverse_bins.csv"
    reverse_metrics, reverse_bins = _run_reverse_diagnostics(
        data_by_year,
        specification,
        regularization_c,
        reverse_metrics_file,
        reverse_bins_file,
        n_bins,
        config.RAW_LOGISTIC_DIAGNOSTIC_MODEL_DIR,
    )
    reverse_horizons, reverse_summary = aggregate_reverse_metrics(reverse_metrics)
    reverse_horizon_bins = aggregate_reliability_bins(reverse_bins, ["horizon"])

    forward_metrics_file = output_dir / "logistic_calibration_forward_metrics.csv"
    forward_bins_file = output_dir / "logistic_calibration_forward_bins.csv"
    forward_metrics, forward_bins = _run_forward_diagnostics(
        data_by_year,
        selected,
        forward_metrics_file,
        forward_bins_file,
        n_bins,
    )

    outputs = {
        "reverse_metrics": reverse_metrics,
        "reverse_bins": reverse_bins,
        "reverse_horizons": reverse_horizons,
        "reverse_summary": reverse_summary,
        "reverse_horizon_bins": reverse_horizon_bins,
        "forward_metrics": forward_metrics,
        "forward_bins": forward_bins,
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / f"logistic_calibration_{name}.csv", index=False)

    render_reliability_panels(
        reverse_horizon_bins,
        panel="horizon",
        output_file=figure_dir / "logistic_calibration_reverse_horizons.pdf",
    )
    render_reliability_panels(
        forward_bins,
        panel="validation_year",
        output_file=figure_dir / "logistic_calibration_forward_years.pdf",
    )
    render_reliability_panels(
        reverse_horizon_bins,
        panel="horizon",
        output_file=figure_dir / "logistic_calibration_reverse_horizons_log.pdf",
        log_scale=True,
    )
    render_reliability_panels(
        forward_bins,
        panel="validation_year",
        output_file=figure_dir / "logistic_calibration_forward_years_log.pdf",
        log_scale=True,
    )
    return outputs


def _run_reverse_diagnostics(
    data_by_year: dict[int, pd.DataFrame],
    specification: model_selection.FeatureSpecification,
    regularization_c: float,
    metrics_file: Path,
    bins_file: Path,
    n_bins: int,
    model_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = _load_checkpoint(metrics_file, specification.name, regularization_c)
    bins = _load_checkpoint(bins_file, specification.name, regularization_c)
    for fold in temporal_folds.reverse_folds():
        missing = [
            year
            for year in fold.validation_years
            if not (
                _metric_complete(metrics, fold.train_start, year)
                and _bin_complete(bins, fold.train_start, year)
            )
        ]
        if not missing:
            continue
        model_path = model_selection.selected_model_path(
            fold.train_years,
            specification,
            regularization_c,
            model_dir,
        )
        if model_path.exists():
            fitted = model_selection.load_selected_model(model_path)
        else:
            training = pd.concat(
                [data_by_year[year] for year in fold.train_years],
                ignore_index=True,
            )
            fitted = model_selection.fit_selected_model(
                training, specification, regularization_c
            )
            model_selection.save_selected_model(fitted, model_path)
        for validation_year in missing:
            validation = data_by_year[validation_year]
            probability = fitted.predict_proba_second_lien(validation)
            y_second = (
                validation[config.LABEL_VAR].to_numpy()
                == config.SECOND_LIEN_CLASS
            )
            metadata = {
                "evaluation_design": "reverse",
                "specification": specification.name,
                "regularization_c": regularization_c,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "validation_year": validation_year,
                "horizon": fold.horizon_for(validation_year),
            }
            metric_row = pd.DataFrame(
                [
                    evaluation.metric_record(
                        y_second, probability, metadata=metadata
                    )
                ]
            )
            bin_rows = reliability_bins(y_second, probability, n_bins).assign(**metadata)
            if not _bin_complete(bins, fold.train_start, validation_year):
                bins = _append_checkpoint(bins, bin_rows, bins_file)
            if not _metric_complete(
                metrics, fold.train_start, validation_year
            ):
                metrics = _upsert_metric_checkpoint(
                    metrics, metric_row, metrics_file
                )
    return metrics.reset_index(drop=True), bins.reset_index(drop=True)


def _run_forward_diagnostics(
    data_by_year: dict[int, pd.DataFrame],
    selected: model_selection.SelectedLogisticModel,
    metrics_file: Path,
    bins_file: Path,
    n_bins: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    specification = selected.specification.name
    regularization_c = selected.regularization_c
    metrics = _load_checkpoint(metrics_file, specification, regularization_c)
    bins = _load_checkpoint(bins_file, specification, regularization_c)
    fold = temporal_folds.forward_fold(config.TRAIN_YEARS, config.VALIDATE_YEARS)
    for validation_year in fold.target_years:
        if _metric_complete(
            metrics, fold.train_start, validation_year
        ) and _bin_complete(bins, fold.train_start, validation_year):
            continue
        validation = data_by_year[validation_year]
        probability = selected.predict_proba_second_lien(validation)
        y_second = (
            validation[config.LABEL_VAR].to_numpy() == config.SECOND_LIEN_CLASS
        )
        metadata = {
            "evaluation_design": "forward_robustness",
            "specification": specification,
            "regularization_c": regularization_c,
            "train_start": fold.train_start,
            "train_end": fold.train_end,
            "validation_year": validation_year,
            "horizon": fold.horizon_for(validation_year),
        }
        metric_row = pd.DataFrame(
            [
                evaluation.metric_record(
                    y_second, probability, metadata=metadata
                )
            ]
        )
        bin_rows = reliability_bins(y_second, probability, n_bins).assign(**metadata)
        if not _bin_complete(bins, fold.train_start, validation_year):
            bins = _append_checkpoint(bins, bin_rows, bins_file)
        if not _metric_complete(metrics, fold.train_start, validation_year):
            metrics = _upsert_metric_checkpoint(metrics, metric_row, metrics_file)
    return metrics.reset_index(drop=True), bins.reset_index(drop=True)


def render_reliability_panels(
    bins: pd.DataFrame,
    panel: str,
    output_file: str | Path,
    log_scale: bool = False,
) -> None:
    """Render a compact 3-by-3 reliability diagram."""
    values = sorted(bins[panel].unique())
    if len(values) > 9:
        raise ValueError("Reliability panel renderer supports at most nine panels")
    fig, axes = plt.subplots(3, 3, figsize=(9.0, 9.0), squeeze=False)
    for axis, value in zip(axes.flat, values, strict=False):
        sample = bins.loc[bins[panel] == value].sort_values("probability_bin")
        maximum = max(
            0.01,
            float(sample["mean_predicted_probability"].max()),
            float(sample["observed_second_share"].max()),
        )
        limit = min(1.0, maximum * 1.05)
        lower = 0.0
        if log_scale:
            lower = 0.5 * min(
                float(sample["mean_predicted_probability"].min()),
                float(sample["observed_second_share"].min()),
            )
            axis.set_xscale("log")
            axis.set_yscale("log")
        axis.plot([lower, limit], [lower, limit], color="0.65", linestyle="--")
        axis.plot(
            sample["mean_predicted_probability"],
            sample["observed_second_share"],
            marker="o",
            linewidth=1.2,
        )
        axis.set_xlim(lower, limit)
        axis.set_ylim(lower, limit)
        axis.set_title(f"{panel.replace('_', ' ').title()} {value}")
        axis.set_xlabel("Mean predicted probability")
        axis.set_ylabel("Observed second-lien share")
    for axis in axes.flat[len(values) :]:
        axis.set_visible(False)
    fig.tight_layout()
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file)
    plt.close(fig)


def _validate_probability_inputs(y: np.ndarray, probability: np.ndarray) -> None:
    evaluation.validate_probability_inputs(y, probability)


def _load_checkpoint(
    path: Path, specification: str, regularization_c: float
) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    matches = (frame["specification"] == specification) & np.isclose(
        frame["regularization_c"], regularization_c
    )
    if not matches.all():
        raise ValueError(f"Checkpoint {path} belongs to a different selected model")
    return frame


def _append_checkpoint(
    existing: pd.DataFrame, new: pd.DataFrame, path: Path
) -> pd.DataFrame:
    new.to_csv(path, mode="a", header=not path.exists(), index=False)
    if existing.empty:
        return new.copy()
    return pd.concat([existing, new], ignore_index=True)


def _bin_complete(
    bins: pd.DataFrame, train_start: int, validation_year: int
) -> bool:
    return _row_present(bins, train_start, validation_year)


def _metric_complete(
    metrics: pd.DataFrame, train_start: int, validation_year: int
) -> bool:
    if not _row_present(metrics, train_start, validation_year):
        return False
    match = metrics.loc[
        (metrics["train_start"] == train_start)
        & (metrics["validation_year"] == validation_year)
    ]
    return bool(
        np.isfinite(match["calibration_intercept"]).all()
        and np.isfinite(match["calibration_slope"]).all()
    )


def _row_present(frame: pd.DataFrame, train_start: int, validation_year: int) -> bool:
    if frame.empty:
        return False
    return bool(
        (
            (frame["train_start"] == train_start)
            & (frame["validation_year"] == validation_year)
        ).any()
    )


def _upsert_metric_checkpoint(
    existing: pd.DataFrame, new: pd.DataFrame, path: Path
) -> pd.DataFrame:
    row = new.iloc[0]
    if not existing.empty:
        keep = ~(
            (existing["train_start"] == row["train_start"])
            & (existing["validation_year"] == row["validation_year"])
        )
        existing = existing.loc[keep]
    combined = pd.concat([existing, new], ignore_index=True)
    combined.to_csv(path, index=False)
    return combined
