"""Step 9 histogram-gradient-boosting density-ratio challenger."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import calibration, config, model_selection
from .density_ratio import aggregation, checkpoints, diagnostics, evaluation
from .density_ratio import folds as temporal_folds
from .density_ratio.families.gradient_boosting import (
    BoostingDensityRatioModel,
    BoostingParameters,
    boosting_features,
    boosting_model_path,
    fit_boosting_ratio_model,
    load_boosting_model,
    save_boosting_model,
)
from .density_ratio.pipeline import run_grid
from .density_ratio.protocols import ModelConfiguration

__all__ = [
    "BoostingDensityRatioModel",
    "BoostingParameters",
    "boosting_features",
    "boosting_model_path",
    "fit_boosting_ratio_model",
    "load_boosting_model",
    "save_boosting_model",
]

def structure_grid() -> list[BoostingParameters]:
    """Return the frozen six-candidate structure screen."""
    return [
        BoostingParameters(max_leaf_nodes=leaves, learning_rate=rate)
        for leaves in config.BOOSTING_STRUCTURE_LEAF_NODES
        for rate in config.BOOSTING_STRUCTURE_LEARNING_RATES
    ]


def refinement_grid(best: BoostingParameters) -> list[BoostingParameters]:
    """Vary iterations and L2 one dimension at a time around a winner."""
    candidates = [
        BoostingParameters(
            max_leaf_nodes=best.max_leaf_nodes,
            learning_rate=best.learning_rate,
            max_iter=max_iter,
            l2_regularization=best.l2_regularization,
            min_samples_leaf=best.min_samples_leaf,
        )
        for max_iter in config.BOOSTING_REFINEMENT_MAX_ITER
    ]
    candidates.extend(
        BoostingParameters(
            max_leaf_nodes=best.max_leaf_nodes,
            learning_rate=best.learning_rate,
            max_iter=best.max_iter,
            l2_regularization=l2,
            min_samples_leaf=best.min_samples_leaf,
        )
        for l2 in config.BOOSTING_REFINEMENT_L2
    )
    return sorted(set(candidates) - {best})


def evaluate_target_year(
    fitted: BoostingDensityRatioModel,
    target: pd.DataFrame,
    fold: temporal_folds.TemporalFold,
    fit_diagnostics: dict,
) -> pd.DataFrame:
    """Estimate the target mixture share and score adjusted probabilities."""
    evaluated = evaluation.evaluate_target(
        fitted,
        target,
        fold,
        label_var=config.LABEL_VAR,
        second_lien_class=config.SECOND_LIEN_CLASS,
    )
    result = evaluated.result
    estimate = evaluated.mixture_estimate
    row = {
        "parameter_id": fitted.parameters.identifier,
        **asdict(fitted.parameters),
        "train_start": fold.train_start,
        "train_end": fold.train_end,
        "validation_year": result.target_year,
        "horizon": result.horizon,
        "n_validation": result.n_observations,
        "actual_second_share": result.actual_second_share,
        "mixture_share": result.mixture_share,
        "mixture_share_error": result.mixture_share - result.actual_second_share,
        "mean_adjusted_probability": result.mean_probability,
        "adjusted_brier": result.brier_score,
        "adjusted_log_loss": result.log_loss,
        "adjusted_hard_share_050": result.hard_share_050,
        "fit_seconds": fit_diagnostics["fit_seconds"],
        "prediction_seconds": evaluated.log_ratio_seconds,
        "n_iter_fitted": fit_diagnostics["n_iter_fitted"],
        "optimizer_converged": estimate.optimizer_converged,
        "mixture_at_boundary": estimate.at_boundary,
        "mixture_em_difference": estimate.share - estimate.em_share,
    }
    return pd.DataFrame([row])


def aggregate_brier(
    cells: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average boosted Brier scores within, then equally across horizons."""
    parameter_columns = [
        "parameter_id",
        "max_leaf_nodes",
        "learning_rate",
        "max_iter",
        "l2_regularization",
        "min_samples_leaf",
    ]
    horizons, summary = aggregation.two_stage_horizon_means(
        cells,
        candidate_columns=parameter_columns,
        metric_columns=("adjusted_brier", "mixture_share_error"),
        count_column="adjusted_brier",
    )
    horizons = (
        horizons.rename(
            columns={
                "adjusted_brier": "mean_brier",
                "mixture_share_error": "mean_share_error",
            }
        )
        .sort_values(["parameter_id", "horizon"])
    )
    summary = (
        summary.rename(
            columns={
                "adjusted_brier": "selection_brier",
                "mixture_share_error": "selection_share_error",
            }
        )
        .sort_values("selection_brier")
        .reset_index(drop=True)
    )
    return horizons, summary


def translate_cluster_cells(cells: pd.DataFrame) -> pd.DataFrame:
    """Translate family-neutral shard cells to the legacy boosting table schema."""
    rows = []
    for row in cells.to_dict("records"):
        parameters = BoostingParameters(**json.loads(row["hyperparameters"]))
        rows.append(
            {
                "parameter_id": parameters.identifier,
                **asdict(parameters),
                "train_start": row["train_start"],
                "train_end": row["train_end"],
                "validation_year": row["target_year"],
                "horizon": row["horizon"],
                "n_validation": row["n_observations"],
                "actual_second_share": row["actual_second_share"],
                "mixture_share": row["mixture_share"],
                "mixture_share_error": (
                    row["mixture_share"] - row["actual_second_share"]
                ),
                "mean_adjusted_probability": row["mean_probability"],
                "adjusted_brier": row["brier_score"],
                "adjusted_log_loss": row["log_loss"],
                "adjusted_hard_share_050": row["hard_share_050"],
                "fit_seconds": np.nan,
                "prediction_seconds": np.nan,
                "n_iter_fitted": parameters.max_iter,
                "optimizer_converged": row["optimizer_converged"],
                "mixture_at_boundary": row["mixture_at_boundary"],
                "mixture_em_difference": row["mixture_em_difference"],
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["parameter_id", "train_start", "validation_year"])
        .reset_index(drop=True)
    )


def run_boosting_challenger(
    data_dir: str | Path = config.SELECTION_DATA_DIR,
    output_dir: str | Path = config.TABLE_DIR,
    fold_model_dir: str | Path = config.BOOSTING_FOLD_MODEL_DIR,
    final_model_file: str | Path = config.SELECTED_BOOSTING_MODEL_FILE,
    figure_dir: str | Path = config.FIGURE_DIR,
) -> dict[str, pd.DataFrame]:
    """Run the frozen staged screen, complete reverse comparison, and refit."""
    output_dir = Path(output_dir)
    fold_model_dir = Path(fold_model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_model_dir.mkdir(parents=True, exist_ok=True)
    data_by_year = model_selection.load_selection_years(data_dir=data_dir)
    folds = list(reversed(temporal_folds.reverse_folds()))
    cells_file = output_dir / "boosting_challenger_checkpoint_cells.csv"
    cells = checkpoints.read_csv(cells_file)
    if cells.empty:
        # Migrate runs made before the complete screen checkpoint and compact
        # final-cell deliverable were assigned distinct filenames.
        cells = checkpoints.read_csv(output_dir / "boosting_challenger_cells.csv")

    screen_fold = folds[0]
    screen_candidates = structure_grid()
    cells = evaluate_grid(
        data_by_year,
        [screen_fold],
        screen_candidates,
        cells,
        cells_file,
        fold_model_dir,
        stage="boosting_screen",
    )
    screen_ids = {candidate.identifier for candidate in screen_candidates}
    _, screen_summary = aggregate_brier(
        cells.loc[
            (cells["train_start"] == screen_fold.train_start)
            & cells["parameter_id"].isin(screen_ids)
        ]
    )
    survivors = [
        _parameters_from_row(row)
        for _, row in screen_summary.head(
            config.BOOSTING_SCREEN_SURVIVORS
        ).iterrows()
    ]
    cells = evaluate_grid(
        data_by_year,
        folds,
        survivors,
        cells,
        cells_file,
        fold_model_dir,
        stage="boosting_survivors",
    )
    survivor_ids = {candidate.identifier for candidate in survivors}
    survivor_cells = cells.loc[cells["parameter_id"].isin(survivor_ids)]
    _, survivor_summary = aggregate_brier(survivor_cells)
    best_structure = _parameters_from_row(survivor_summary.iloc[0])

    refinements = refinement_grid(best_structure)
    cells = evaluate_grid(
        data_by_year,
        folds,
        refinements,
        cells,
        cells_file,
        fold_model_dir,
        stage="boosting_refinement",
    )
    eligible = [*survivors, *refinements]
    eligible_ids = {candidate.identifier for candidate in eligible}
    final_cells = cells.loc[cells["parameter_id"].isin(eligible_ids)].copy()
    horizons, summary = aggregate_brier(final_cells)
    winner = _parameters_from_row(summary.iloc[0])
    comparison = compare_with_logistic(
        final_cells.loc[final_cells["parameter_id"] == winner.identifier],
        output_dir / "mixture_reverse_cell_shares.csv",
    )
    decision = pd.DataFrame(
        [
            {
                **summary.iloc[0].to_dict(),
                "screen_train_start": screen_fold.train_start,
                "screen_survivors": config.BOOSTING_SCREEN_SURVIVORS,
                "selection_metric": "mixture_adjusted_brier_equal_horizon_weight",
                "model_file": str(final_model_file),
            }
        ]
    )
    final_training = pd.concat(
        [data_by_year[year] for year in config.TRAIN_YEARS], ignore_index=True
    )
    final_model_file = Path(final_model_file)
    if final_model_file.exists():
        cached_final = load_boosting_model(final_model_file)
    else:
        cached_final = None
    if (
        cached_final is not None
        and cached_final.parameters == winner
        and cached_final.train_years == tuple(config.TRAIN_YEARS)
    ):
        final_model = cached_final
        final_fit = {
            "fit_seconds": np.nan,
            "n_iter_fitted": final_model.classifier.n_iter_,
        }
    else:
        final_model, final_fit = fit_boosting_ratio_model(final_training, winner)
        save_boosting_model(final_model, final_model_file)
    decision["final_fit_seconds"] = final_fit["fit_seconds"]
    decision["final_n_iter_fitted"] = final_fit["n_iter_fitted"]

    outputs = {
        "screen_summary": screen_summary,
        "cells": final_cells,
        "horizons": horizons,
        "summary": summary,
        "comparison": comparison,
        "decision": decision,
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / f"boosting_challenger_{name}.csv", index=False)
    outputs.update(
        run_boosting_diagnostics(
            data_by_year,
            winner,
            final_model,
            fold_model_dir,
            output_dir,
            figure_dir,
        )
    )
    return outputs


def run_boosting_diagnostics(
    data_by_year: dict[int, pd.DataFrame],
    parameters: BoostingParameters,
    final_model: BoostingDensityRatioModel,
    fold_model_dir: str | Path,
    output_dir: str | Path,
    figure_dir: str | Path,
    n_bins: int = calibration.DEFAULT_N_BINS,
) -> dict[str, pd.DataFrame]:
    """Run reverse and forward calibration checks using persisted fits."""
    output_dir = Path(output_dir)
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    reverse_metrics_file = output_dir / "boosting_calibration_reverse_metrics.csv"
    reverse_bins_file = output_dir / "boosting_calibration_reverse_bins.csv"
    reverse_metrics = checkpoints.read_csv(reverse_metrics_file)
    reverse_bins = checkpoints.read_csv(reverse_bins_file)
    for fold in reversed(temporal_folds.reverse_folds()):
        model = load_boosting_model(
            boosting_model_path(fold.train_years, parameters, fold_model_dir)
        )
        for validation_year in fold.validation_years:
            reverse_metrics, reverse_bins = _diagnose_cell(
                model,
                data_by_year[validation_year],
                fold,
                "reverse",
                reverse_metrics,
                reverse_bins,
                reverse_metrics_file,
                reverse_bins_file,
                n_bins,
            )
    reverse_horizons, reverse_summary = calibration.aggregate_reverse_metrics(
        reverse_metrics
    )
    reverse_horizon_bins = calibration.aggregate_reliability_bins(
        reverse_bins, ["horizon"]
    )

    forward_metrics_file = output_dir / "boosting_calibration_forward_metrics.csv"
    forward_bins_file = output_dir / "boosting_calibration_forward_bins.csv"
    forward_metrics = checkpoints.read_csv(forward_metrics_file)
    forward_bins = checkpoints.read_csv(forward_bins_file)
    forward_fold = temporal_folds.forward_fold(
        config.TRAIN_YEARS, config.VALIDATE_YEARS
    )
    for validation_year in forward_fold.validation_years:
        forward_metrics, forward_bins = _diagnose_cell(
            final_model,
            data_by_year[validation_year],
            forward_fold,
            "forward_robustness",
            forward_metrics,
            forward_bins,
            forward_metrics_file,
            forward_bins_file,
            n_bins,
        )
    forward_summary = calibration.aggregate_forward_metrics(forward_metrics)
    outputs = {
        "boosting_reverse_metrics": reverse_metrics,
        "boosting_reverse_bins": reverse_bins,
        "boosting_reverse_horizons": reverse_horizons,
        "boosting_reverse_summary": reverse_summary,
        "boosting_reverse_horizon_bins": reverse_horizon_bins,
        "boosting_forward_metrics": forward_metrics,
        "boosting_forward_bins": forward_bins,
        "boosting_forward_summary": forward_summary,
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    calibration.render_reliability_panels(
        reverse_horizon_bins,
        panel="horizon",
        output_file=figure_dir / "boosting_calibration_reverse_horizons.pdf",
    )
    calibration.render_reliability_panels(
        forward_bins,
        panel="validation_year",
        output_file=figure_dir / "boosting_calibration_forward_years.pdf",
    )
    return outputs


def _diagnose_cell(
    model: BoostingDensityRatioModel,
    target: pd.DataFrame,
    fold: temporal_folds.TemporalFold,
    design: str,
    metrics: pd.DataFrame,
    bins: pd.DataFrame,
    metrics_file: Path,
    bins_file: Path,
    n_bins: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation_year = int(target["year"].iloc[0])
    if _diagnostic_cell_present(metrics, fold.train_start, validation_year) and (
        _diagnostic_cell_present(bins, fold.train_start, validation_year)
    ):
        return metrics, bins
    evaluated = evaluation.evaluate_target(
        model,
        target,
        fold,
        label_var=config.LABEL_VAR,
        second_lien_class=config.SECOND_LIEN_CLASS,
    )
    estimate = evaluated.mixture_estimate
    probability = evaluated.probability
    y_second = target[config.LABEL_VAR].to_numpy() == config.SECOND_LIEN_CLASS
    horizon = fold.horizon_for(validation_year)
    metadata = {
        "evaluation_design": design,
        "estimator": "hist_gradient_boosting_mixture",
        "parameter_id": model.parameters.identifier,
        "train_start": fold.train_start,
        "train_end": fold.train_end,
        "validation_year": validation_year,
        "horizon": horizon,
    }
    diagnostic = diagnostics.evaluate_cell(
        y_second,
        probability,
        metadata=metadata,
        n_bins=n_bins,
        metrics=evaluated.metrics,
        additional_metrics={
            "mixture_share": evaluated.result.mixture_share,
            "share_optimizer_converged": estimate.optimizer_converged,
            "share_at_boundary": estimate.at_boundary,
        },
    )
    metric_row = diagnostic.metrics
    bin_rows = diagnostic.bins
    key_columns = ("train_start", "validation_year")
    metrics = checkpoints.replace_rows(
        metrics, metric_row, metrics_file, key_columns=key_columns
    )
    bins = checkpoints.replace_rows(
        bins, bin_rows, bins_file, key_columns=key_columns
    )
    return metrics, bins


def evaluate_grid(
    data_by_year: dict[int, pd.DataFrame],
    folds: Iterable[temporal_folds.TemporalFold],
    candidates: Iterable[BoostingParameters],
    cells: pd.DataFrame,
    checkpoint_file: str | Path,
    model_dir: str | Path,
    stage: str = "boosting_grid",
) -> pd.DataFrame:
    """Evaluate boosting candidates through the shared runner and shards."""
    from .density_ratio.families.gradient_boosting import GradientBoostingFamily

    checkpoint_file = Path(checkpoint_file)
    model_dir = Path(model_dir)
    candidates = list(candidates)
    configurations = tuple(
        ModelConfiguration.from_mapping(
            "hist_gradient_boosting",
            "primitive_continuous_and_native_categories",
            asdict(candidate),
            random_seed=config.BOOSTING_RANDOM_STATE,
        )
        for candidate in candidates
    )
    aggregated = run_grid(
        data_by_year,
        list(folds),
        {"primitive_continuous_and_native_categories": configurations},
        GradientBoostingFamily(model_dir),
        stage=stage,
        artifact_root=model_dir,
        output_root=model_dir / "runner",
    )
    new_cells = translate_cluster_cells(aggregated.cells)
    expected_ids = {candidate.identifier for candidate in candidates}
    if set(new_cells["parameter_id"]) != expected_ids:
        raise ValueError("Shared runner returned an unexpected boosting configuration")
    keys = ["parameter_id", "train_start", "validation_year"]
    cells = (
        pd.concat([cells, new_cells], ignore_index=True)
        .drop_duplicates(keys, keep="last")
        .sort_values(keys)
        .reset_index(drop=True)
    )
    checkpoints.write_csv(cells, checkpoint_file)
    return cells


def compare_with_logistic(
    boosting_cells: pd.DataFrame, logistic_cells_file: str | Path
) -> pd.DataFrame:
    """Join the selected boosted cells to frozen adjusted-logistic results."""
    logistic = pd.read_csv(logistic_cells_file)
    return evaluation.merge_cell_metrics(
        boosting_cells,
        primary_metric="adjusted_brier",
        primary_output="boosting_brier",
        comparisons={
            "logistic_brier": (logistic, "adjusted_brier_known_source_prior")
        },
        difference_columns={
            "logistic_brier": "boosting_minus_logistic_brier"
        },
    )


def _parameters_from_row(row: pd.Series) -> BoostingParameters:
    return BoostingParameters(
        max_leaf_nodes=int(row["max_leaf_nodes"]),
        learning_rate=float(row["learning_rate"]),
        max_iter=int(row["max_iter"]),
        l2_regularization=float(row["l2_regularization"]),
        min_samples_leaf=int(row["min_samples_leaf"]),
    )


def _diagnostic_cell_present(
    frame: pd.DataFrame, train_start: int, validation_year: int
) -> bool:
    return checkpoints.rows_present(
        frame,
        {"train_start": train_start, "validation_year": validation_year},
    )
