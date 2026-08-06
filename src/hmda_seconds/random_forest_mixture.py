"""Fixed Random Forest challenger with target-year mixture adjustment."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import calibration, config, model_selection
from .density_ratio import checkpoints, diagnostics, evaluation
from .density_ratio import folds as temporal_folds
from .density_ratio.families.random_forest import (
    RandomForestDensityRatioModel,
    fit_forest_ratio_model,
    forest_features,
    forest_model_path,
    load_forest_model,
    save_forest_model,
)
from .density_ratio.pipeline import run_grid
from .density_ratio.protocols import ModelConfiguration

__all__ = [
    "RandomForestDensityRatioModel",
    "fit_forest_ratio_model",
    "forest_features",
    "forest_model_path",
    "load_forest_model",
    "save_forest_model",
]

ESTIMATOR = "random_forest_mixture"


def run_random_forest_mixture(
    data_dir: str | Path = config.SELECTION_DATA_DIR,
    output_dir: str | Path = config.TABLE_DIR,
    figure_dir: str | Path = config.FIGURE_DIR,
    fold_model_dir: str | Path = config.RF_MIXTURE_FOLD_MODEL_DIR,
    final_model_file: str | Path = config.RF_MIXTURE_MODEL_FILE,
    n_bins: int = calibration.DEFAULT_N_BINS,
) -> dict[str, pd.DataFrame]:
    """Run fixed reverse and forward RF-mixture comparisons resumably."""
    output_dir = Path(output_dir)
    figure_dir = Path(figure_dir)
    fold_model_dir = Path(fold_model_dir)
    final_model_file = Path(final_model_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    fold_model_dir.mkdir(parents=True, exist_ok=True)
    data_by_year = model_selection.load_selection_years(data_dir=data_dir)

    reverse_metrics_file = output_dir / "rf_mixture_reverse_metrics.csv"
    reverse_bins_file = output_dir / "rf_mixture_reverse_bins.csv"
    reverse_folds = list(reversed(temporal_folds.reverse_folds()))
    reverse_metrics = _reverse_metrics_from_runner(
        data_by_year, reverse_folds, fold_model_dir
    )
    checkpoints.write_csv(reverse_metrics, reverse_metrics_file)
    reverse_bins = checkpoints.read_csv(reverse_bins_file)
    for fold in reverse_folds:
        path = forest_model_path(fold.train_years, fold_model_dir)
        model = load_forest_model(path)
        for validation_year in fold.validation_years:
            reverse_metrics, reverse_bins = _evaluate_cell(
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

    final_years = tuple(config.TRAIN_YEARS)
    if final_model_file.exists():
        final_model = load_forest_model(final_model_file)
    else:
        final_training = pd.concat(
            [data_by_year[year] for year in final_years], ignore_index=True
        )
        final_model, _ = fit_forest_ratio_model(final_training)
        save_forest_model(final_model, final_model_file)
    if final_model.train_years != final_years:
        raise RuntimeError("Final RF mixture model has unexpected training years")

    forward_metrics_file = output_dir / "rf_mixture_forward_metrics.csv"
    forward_bins_file = output_dir / "rf_mixture_forward_bins.csv"
    forward_metrics = checkpoints.read_csv(forward_metrics_file)
    forward_bins = checkpoints.read_csv(forward_bins_file)
    forward_fold = temporal_folds.forward_fold(
        final_years, config.VALIDATE_YEARS
    )
    for validation_year in forward_fold.validation_years:
        forward_metrics, forward_bins = _evaluate_cell(
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
    comparison = estimator_comparison(
        reverse_metrics,
        output_dir / "mixture_calibration_reverse_metrics.csv",
        output_dir / "boosting_reverse_metrics.csv",
    )
    outputs = {
        "reverse_metrics": reverse_metrics,
        "reverse_bins": reverse_bins,
        "reverse_horizons": reverse_horizons,
        "reverse_summary": reverse_summary,
        "reverse_horizon_bins": reverse_horizon_bins,
        "forward_metrics": forward_metrics,
        "forward_bins": forward_bins,
        "forward_summary": forward_summary,
        "comparison": comparison,
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / f"rf_mixture_{name}.csv", index=False)
    calibration.render_reliability_panels(
        reverse_horizon_bins,
        panel="horizon",
        output_file=figure_dir / "rf_mixture_reverse_horizons.pdf",
    )
    calibration.render_reliability_panels(
        forward_bins,
        panel="validation_year",
        output_file=figure_dir / "rf_mixture_forward_years.pdf",
    )
    return outputs


def _reverse_metrics_from_runner(
    data_by_year: dict[int, pd.DataFrame],
    folds: list[temporal_folds.TemporalFold],
    model_dir: Path,
) -> pd.DataFrame:
    """Translate shared RF shard cells into the established metric schema."""
    from .density_ratio.families.random_forest import RandomForestFamily

    configuration = ModelConfiguration.from_mapping(
        "random_forest",
        "raw_continuous_and_full_one_hot_categories",
        {
            "max_depth": config.RF_KWARGS["max_depth"],
            "n_estimators": config.RF_KWARGS["n_estimators"],
        },
        random_seed=config.RF_KWARGS["random_state"],
    )
    aggregated = run_grid(
        data_by_year,
        folds,
        {configuration.specification: (configuration,)},
        RandomForestFamily(model_dir),
        stage="random_forest_reverse",
        artifact_root=model_dir,
        output_root=model_dir / "runner",
    )
    rows = []
    for row in aggregated.cells.to_dict("records"):
        rows.append(
            {
                "evaluation_design": "reverse",
                "estimator": ESTIMATOR,
                "n_estimators": config.RF_KWARGS["n_estimators"],
                "max_depth": config.RF_KWARGS["max_depth"],
                "train_start": row["train_start"],
                "train_end": row["train_end"],
                "validation_year": row["target_year"],
                "horizon": row["horizon"],
                "n": row["n_observations"],
                "brier_score": row["brier_score"],
                "log_loss": row["log_loss"],
                "observed_second_share": row["actual_second_share"],
                "mean_predicted_second_share": row["mean_probability"],
                "calibration_mean_error": row["calibration_mean_error"],
                "calibration_intercept": row["calibration_intercept"],
                "calibration_slope": row["calibration_slope"],
                "mixture_share": row["mixture_share"],
                "adjusted_hard_share_050": row["hard_share_050"],
                "share_optimizer_converged": row["optimizer_converged"],
                "share_at_boundary": row["mixture_at_boundary"],
                "mixture_em_difference": row["mixture_em_difference"],
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["train_start", "validation_year"]
    ).reset_index(drop=True)


def estimator_comparison(
    forest: pd.DataFrame,
    logistic_file: str | Path,
    boosting_file: str | Path,
) -> pd.DataFrame:
    """Join all three mixture estimators' reverse-cell Brier scores."""
    return evaluation.merge_cell_metrics(
        forest,
        primary_metric="brier_score",
        primary_output="forest_brier",
        comparisons={
            "logistic_brier": (pd.read_csv(logistic_file), "brier_score"),
            "boosting_brier": (pd.read_csv(boosting_file), "brier_score"),
        },
        difference_columns={
            "logistic_brier": "forest_minus_logistic_brier",
            "boosting_brier": "forest_minus_boosting_brier",
        },
    )


def _evaluate_cell(
    model: RandomForestDensityRatioModel,
    target: pd.DataFrame,
    fold: temporal_folds.TemporalFold,
    design: str,
    metrics: pd.DataFrame,
    bins: pd.DataFrame,
    metrics_file: Path,
    bins_file: Path,
    n_bins: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    year = int(target["year"].iloc[0])
    if _cell_present(metrics, fold.train_start, year) and _cell_present(
        bins, fold.train_start, year
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
    horizon = fold.horizon_for(year)
    metadata = {
        "evaluation_design": design,
        "estimator": ESTIMATOR,
        "n_estimators": config.RF_KWARGS["n_estimators"],
        "max_depth": config.RF_KWARGS["max_depth"],
        "train_start": fold.train_start,
        "train_end": fold.train_end,
        "validation_year": year,
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
            "adjusted_hard_share_050": evaluated.result.hard_share_050,
            "share_optimizer_converged": estimate.optimizer_converged,
            "share_at_boundary": estimate.at_boundary,
            "mixture_em_difference": estimate.share - estimate.em_share,
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


def _cell_present(
    frame: pd.DataFrame, train_start: int, validation_year: int
) -> bool:
    return checkpoints.rows_present(
        frame,
        {"train_start": train_start, "validation_year": validation_year},
    )
