"""Step 6 diagnostics for frozen known-source-prior mixture probabilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import calibration, config, mixture, model_selection
from .density_ratio import folds as temporal_folds

ESTIMATOR = "known_source_prior_mixture"
TAIL_QUANTILES = (0.0, 0.001, 0.01, 0.5, 0.99, 0.999, 1.0)


def run_mixture_calibration_diagnostics(
    data_dir: str | Path = config.SELECTION_DATA_DIR,
    output_dir: str | Path = config.TABLE_DIR,
    figure_dir: str | Path = config.FIGURE_DIR,
    model_file: str | Path = config.SELECTED_LOGISTIC_MODEL_FILE,
    fold_model_dir: str | Path = config.MIXTURE_FOLD_MODEL_DIR,
    n_bins: int = calibration.DEFAULT_N_BINS,
) -> dict[str, pd.DataFrame]:
    """Diagnose frozen mixture-adjusted probabilities backward and forward."""
    output_dir = Path(output_dir)
    figure_dir = Path(figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    data_by_year = model_selection.load_selection_years(data_dir=data_dir)
    selected = model_selection.load_selected_model(model_file)

    reverse_metrics, reverse_bins, reverse_tails = _run_design(
        data_by_year=data_by_year,
        folds=list(reversed(temporal_folds.reverse_folds())),
        specification=selected.specification,
        regularization_c=selected.regularization_c,
        metrics_file=output_dir / "mixture_calibration_reverse_metrics.csv",
        bins_file=output_dir / "mixture_calibration_reverse_bins.csv",
        tails_file=output_dir / "mixture_calibration_reverse_ratio_tails.csv",
        n_bins=n_bins,
        design="reverse",
        fold_model_dir=Path(fold_model_dir),
    )
    forward_fold = temporal_folds.forward_fold(
        train_years=config.TRAIN_YEARS,
        target_years=config.VALIDATE_YEARS,
    )
    forward_metrics, forward_bins, forward_tails = _run_design(
        data_by_year=data_by_year,
        folds=[forward_fold],
        specification=selected.specification,
        regularization_c=selected.regularization_c,
        metrics_file=output_dir / "mixture_calibration_forward_metrics.csv",
        bins_file=output_dir / "mixture_calibration_forward_bins.csv",
        tails_file=output_dir / "mixture_calibration_forward_ratio_tails.csv",
        n_bins=n_bins,
        design="forward_robustness",
        fold_model_dir=Path(fold_model_dir),
    )

    reverse_horizons, reverse_summary = calibration.aggregate_reverse_metrics(
        reverse_metrics
    )
    reverse_horizon_bins = calibration.aggregate_reliability_bins(
        reverse_bins, ["horizon"]
    )
    forward_summary = _simple_summary(forward_metrics)
    outputs = {
        "reverse_metrics": reverse_metrics,
        "reverse_bins": reverse_bins,
        "reverse_ratio_tails": reverse_tails,
        "reverse_horizons": reverse_horizons,
        "reverse_summary": reverse_summary,
        "reverse_horizon_bins": reverse_horizon_bins,
        "forward_metrics": forward_metrics,
        "forward_bins": forward_bins,
        "forward_ratio_tails": forward_tails,
        "forward_summary": forward_summary,
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / f"mixture_calibration_{name}.csv", index=False)

    calibration.render_reliability_panels(
        reverse_horizon_bins,
        panel="horizon",
        output_file=figure_dir / "mixture_calibration_reverse_horizons.pdf",
    )
    calibration.render_reliability_panels(
        reverse_horizon_bins,
        panel="horizon",
        output_file=figure_dir / "mixture_calibration_reverse_horizons_log.pdf",
        log_scale=True,
    )
    calibration.render_reliability_panels(
        forward_bins,
        panel="validation_year",
        output_file=figure_dir / "mixture_calibration_forward_years.pdf",
    )
    calibration.render_reliability_panels(
        forward_bins,
        panel="validation_year",
        output_file=figure_dir / "mixture_calibration_forward_years_log.pdf",
        log_scale=True,
    )
    return outputs


def _run_design(
    data_by_year: dict[int, pd.DataFrame],
    folds: list[temporal_folds.TemporalFold],
    specification: model_selection.FeatureSpecification,
    regularization_c: float,
    metrics_file: Path,
    bins_file: Path,
    tails_file: Path,
    n_bins: int,
    design: str,
    fold_model_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = _read(metrics_file)
    bins = _read(bins_file)
    tails = _read(tails_file)
    for fold in folds:
        missing = [
            year
            for year in fold.validation_years
            if not _cell_complete(metrics, bins, tails, fold.train_start, year)
        ]
        model_path = mixture.known_source_prior_model_path(
            fold.train_years, specification, regularization_c, fold_model_dir
        )
        if model_path.exists():
            fitted = mixture.load_known_source_prior_model(model_path)
        else:
            training = pd.concat(
                [data_by_year[year] for year in fold.train_years],
                ignore_index=True,
            )
            fitted = mixture.fit_known_source_prior_model(
                training,
                specification,
                regularization_c,
                model_file=model_path,
            )
        if not missing:
            continue
        for validation_year in missing:
            target = data_by_year[validation_year]
            log_ratio = fitted.log_ratio(target)
            share = mixture.estimate_mixture_share(log_ratio)
            probability = mixture.adjusted_probability(log_ratio, share.share)
            y_second = (
                target[config.LABEL_VAR].to_numpy() == config.SECOND_LIEN_CLASS
            )
            horizon = fold.horizon_for(validation_year)
            metadata = {
                "evaluation_design": design,
                "estimator": ESTIMATOR,
                "specification": specification.name,
                "regularization_c": regularization_c,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "validation_year": validation_year,
                "horizon": horizon,
            }
            metric_row = pd.DataFrame(
                [
                    {
                        **metadata,
                        **calibration.probability_metrics(y_second, probability),
                        "mixture_share": share.share,
                        "share_optimizer_converged": share.optimizer_converged,
                        "share_at_boundary": share.at_boundary,
                    }
                ]
            )
            bin_rows = calibration.reliability_bins(
                y_second, probability, n_bins
            ).assign(**metadata)
            tail_row = pd.DataFrame(
                [{**metadata, **_tail_metrics(log_ratio)}]
            )
            metrics = _upsert(metrics, metric_row, metrics_file)
            bins = _replace_cell(bins, bin_rows, bins_file)
            tails = _upsert(tails, tail_row, tails_file)
    return metrics.reset_index(drop=True), bins.reset_index(drop=True), tails.reset_index(drop=True)


def _tail_metrics(log_ratio: np.ndarray) -> dict[str, float]:
    quantiles = np.quantile(log_ratio, TAIL_QUANTILES)
    labels = ("min", "p001", "p01", "median", "p99", "p999", "max")
    result = {
        f"log_ratio_{label}": float(value)
        for label, value in zip(labels, quantiles, strict=True)
    }
    result["share_log_ratio_gt_10"] = float(np.mean(log_ratio > 10))
    result["share_log_ratio_lt_minus_10"] = float(np.mean(log_ratio < -10))
    return result


def _simple_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                **{column: metrics[column].mean() for column in calibration.METRIC_COLUMNS},
                "n_cells": len(metrics),
                "weighting": "equal_across_validation_years",
            }
        ]
    )


def _cell_complete(
    metrics: pd.DataFrame,
    bins: pd.DataFrame,
    tails: pd.DataFrame,
    train_start: int,
    validation_year: int,
) -> bool:
    return all(
        not frame.empty
        and bool(
            (
                (frame["train_start"] == train_start)
                & (frame["validation_year"] == validation_year)
            ).any()
        )
        for frame in (metrics, bins, tails)
    )


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _upsert(existing: pd.DataFrame, new: pd.DataFrame, path: Path) -> pd.DataFrame:
    return _replace_cell(existing, new, path)


def _replace_cell(
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
