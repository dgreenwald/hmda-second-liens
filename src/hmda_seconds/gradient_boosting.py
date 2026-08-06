"""Step 9 histogram-gradient-boosting density-ratio challenger."""

from __future__ import annotations

import pickle
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier

from . import calibration, config, mixture, model_selection

BOOSTING_FEATURES = [*config.CONTINUOUS_VARS, *config.CATEGORY_VARS]
CATEGORICAL_MASK = np.array([False, False, True, True])
PROBABILITY_EPSILON = 1e-12


@dataclass(frozen=True, order=True)
class BoostingParameters:
    """A reproducible, compact histogram-boosting specification."""

    max_leaf_nodes: int
    learning_rate: float
    max_iter: int = config.BOOSTING_BASE_MAX_ITER
    l2_regularization: float = config.BOOSTING_BASE_L2
    min_samples_leaf: int = config.BOOSTING_MIN_SAMPLES_LEAF

    @property
    def identifier(self) -> str:
        """Return a stable label suitable for tables and artifact names."""
        rate = _number_label(self.learning_rate)
        l2 = _number_label(self.l2_regularization)
        return (
            f"leaves_{self.max_leaf_nodes}__lr_{rate}__iter_{self.max_iter}"
            f"__l2_{l2}__minleaf_{self.min_samples_leaf}"
        )


@dataclass
class BoostingDensityRatioModel:
    """Persisted equal-source-prior boosting density-ratio model."""

    classifier: HistGradientBoostingClassifier
    parameters: BoostingParameters
    train_years: tuple[int, ...]
    feature_names: tuple[str, ...] = tuple(BOOSTING_FEATURES)

    def log_ratio(self, frame: pd.DataFrame) -> np.ndarray:
        """Return clipped balanced-prior log odds as the log density ratio."""
        features = boosting_features(frame)
        second_column = list(self.classifier.classes_).index(
            config.SECOND_LIEN_CLASS
        )
        probability = self.classifier.predict_proba(features)[:, second_column]
        probability = np.clip(
            probability, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON
        )
        return logit(probability)


def boosting_features(frame: pd.DataFrame) -> np.ndarray:
    """Construct the four primitive boosting features without engineering."""
    missing = set(BOOSTING_FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing boosting features: {sorted(missing)}")
    features = frame[BOOSTING_FEATURES].to_numpy(dtype=float)
    if not np.isfinite(features).all():
        raise ValueError("Boosting features contain non-finite values")
    for index, variable in enumerate(config.CATEGORY_VARS, start=2):
        unknown = set(np.unique(features[:, index])) - set(
            config.CATEGORY_LEVELS[variable]
        )
        if unknown:
            raise ValueError(f"Unknown {variable} levels: {sorted(unknown)}")
    return features


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


def fit_boosting_ratio_model(
    training: pd.DataFrame,
    parameters: BoostingParameters,
) -> tuple[BoostingDensityRatioModel, dict]:
    """Fit an equal-source-prior histogram-boosting classifier."""
    features = boosting_features(training)
    labels = training[config.LABEL_VAR].to_numpy()
    is_second = labels == config.SECOND_LIEN_CLASS
    weights = mixture.equal_source_prior_weights(training, is_second)
    classifier = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=parameters.learning_rate,
        max_iter=parameters.max_iter,
        max_leaf_nodes=parameters.max_leaf_nodes,
        min_samples_leaf=parameters.min_samples_leaf,
        l2_regularization=parameters.l2_regularization,
        categorical_features=CATEGORICAL_MASK,
        early_stopping=False,
        random_state=config.BOOSTING_RANDOM_STATE,
    )
    start = time.perf_counter()
    classifier.fit(features, labels, sample_weight=weights)
    fit_seconds = time.perf_counter() - start
    fitted = BoostingDensityRatioModel(
        classifier=classifier,
        parameters=parameters,
        train_years=tuple(sorted(pd.unique(training["year"]))),
    )
    return fitted, {
        "fit_seconds": fit_seconds,
        "n_iter_fitted": int(classifier.n_iter_),
    }


def evaluate_target_year(
    fitted: BoostingDensityRatioModel,
    target: pd.DataFrame,
    fold: model_selection.ReverseFold,
    fit_diagnostics: dict,
) -> pd.DataFrame:
    """Estimate the target mixture share and score adjusted probabilities."""
    start = time.perf_counter()
    log_ratio = fitted.log_ratio(target)
    prediction_seconds = time.perf_counter() - start
    estimate = mixture.estimate_mixture_share(log_ratio)
    probability = mixture.adjusted_probability(log_ratio, estimate.share)
    actual_second = (
        target[config.LABEL_VAR].to_numpy() == config.SECOND_LIEN_CLASS
    )
    actual_share = float(actual_second.mean())
    row = {
        "parameter_id": fitted.parameters.identifier,
        **asdict(fitted.parameters),
        "train_start": fold.train_start,
        "train_end": fold.train_end,
        "validation_year": int(target["year"].iloc[0]),
        "horizon": fold.train_start - int(target["year"].iloc[0]),
        "n_validation": len(target),
        "actual_second_share": actual_share,
        "mixture_share": estimate.share,
        "mixture_share_error": estimate.share - actual_share,
        "mean_adjusted_probability": float(probability.mean()),
        "adjusted_brier": float(np.mean((probability - actual_second) ** 2)),
        "adjusted_log_loss": _binary_log_loss(actual_second, probability),
        "adjusted_hard_share_050": float((probability >= 0.5).mean()),
        "fit_seconds": fit_diagnostics["fit_seconds"],
        "prediction_seconds": prediction_seconds,
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
    horizons = (
        cells.groupby([*parameter_columns, "horizon"], as_index=False)
        .agg(
            mean_brier=("adjusted_brier", "mean"),
            mean_share_error=("mixture_share_error", "mean"),
            n_cells=("adjusted_brier", "size"),
        )
        .sort_values(["parameter_id", "horizon"])
    )
    summary = (
        horizons.groupby(parameter_columns, as_index=False)
        .agg(
            selection_brier=("mean_brier", "mean"),
            selection_share_error=("mean_share_error", "mean"),
            n_horizons=("horizon", "nunique"),
            n_cells=("n_cells", "sum"),
        )
        .sort_values("selection_brier")
        .reset_index(drop=True)
    )
    return horizons, summary


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
    folds = list(reversed(model_selection.reverse_folds()))
    cells_file = output_dir / "boosting_challenger_checkpoint_cells.csv"
    cells = _read_csv_if_exists(cells_file)
    if cells.empty:
        # Migrate runs made before the complete screen checkpoint and compact
        # final-cell deliverable were assigned distinct filenames.
        cells = _read_csv_if_exists(output_dir / "boosting_challenger_cells.csv")

    screen_fold = folds[0]
    screen_candidates = structure_grid()
    cells = evaluate_grid(
        data_by_year,
        [screen_fold],
        screen_candidates,
        cells,
        cells_file,
        fold_model_dir,
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
        data_by_year, folds, survivors, cells, cells_file, fold_model_dir
    )
    survivor_ids = {candidate.identifier for candidate in survivors}
    survivor_cells = cells.loc[cells["parameter_id"].isin(survivor_ids)]
    _, survivor_summary = aggregate_brier(survivor_cells)
    best_structure = _parameters_from_row(survivor_summary.iloc[0])

    refinements = refinement_grid(best_structure)
    cells = evaluate_grid(
        data_by_year, folds, refinements, cells, cells_file, fold_model_dir
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
    reverse_metrics = _read_csv_if_exists(reverse_metrics_file)
    reverse_bins = _read_csv_if_exists(reverse_bins_file)
    for fold in reversed(model_selection.reverse_folds()):
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
    forward_metrics = _read_csv_if_exists(forward_metrics_file)
    forward_bins = _read_csv_if_exists(forward_bins_file)
    forward_fold = model_selection.ReverseFold(
        tuple(config.TRAIN_YEARS), tuple(config.VALIDATE_YEARS)
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
    forward_summary = pd.DataFrame(
        [
            {
                **{
                    column: forward_metrics[column].mean()
                    for column in calibration.METRIC_COLUMNS
                },
                "n_cells": len(forward_metrics),
                "weighting": "equal_across_validation_years",
            }
        ]
    )
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
    fold: model_selection.ReverseFold,
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
    log_ratio = model.log_ratio(target)
    estimate = mixture.estimate_mixture_share(log_ratio)
    probability = mixture.adjusted_probability(log_ratio, estimate.share)
    y_second = target[config.LABEL_VAR].to_numpy() == config.SECOND_LIEN_CLASS
    horizon = (
        fold.train_start - validation_year
        if design == "reverse"
        else validation_year - fold.train_end
    )
    metadata = {
        "evaluation_design": design,
        "estimator": "hist_gradient_boosting_mixture",
        "parameter_id": model.parameters.identifier,
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
                "mixture_share": estimate.share,
                "share_optimizer_converged": estimate.optimizer_converged,
                "share_at_boundary": estimate.at_boundary,
            }
        ]
    )
    bin_rows = calibration.reliability_bins(
        y_second, probability, n_bins
    ).assign(**metadata)
    metrics = _replace_diagnostic_cell(metrics, metric_row, metrics_file)
    bins = _replace_diagnostic_cell(bins, bin_rows, bins_file)
    return metrics, bins


def evaluate_grid(
    data_by_year: dict[int, pd.DataFrame],
    folds: Iterable[model_selection.ReverseFold],
    candidates: Iterable[BoostingParameters],
    cells: pd.DataFrame,
    checkpoint_file: str | Path,
    model_dir: str | Path,
) -> pd.DataFrame:
    """Evaluate missing fold/candidate cells and checkpoint each target year."""
    checkpoint_file = Path(checkpoint_file)
    for fold in folds:
        training = None
        for candidate in candidates:
            missing_years = [
                year
                for year in fold.validation_years
                if not _cell_present(
                    cells, candidate.identifier, fold.train_start, year
                )
            ]
            if not missing_years:
                continue
            model_file = boosting_model_path(
                fold.train_years, candidate, model_dir
            )
            if model_file.exists():
                fitted = load_boosting_model(model_file)
                fit_diagnostics = {
                    "fit_seconds": np.nan,
                    "n_iter_fitted": fitted.classifier.n_iter_,
                }
            else:
                if training is None:
                    training = pd.concat(
                        [data_by_year[year] for year in fold.train_years],
                        ignore_index=True,
                    )
                fitted, fit_diagnostics = fit_boosting_ratio_model(
                    training, candidate
                )
                save_boosting_model(fitted, model_file)
            for validation_year in missing_years:
                row = evaluate_target_year(
                    fitted,
                    data_by_year[validation_year],
                    fold,
                    fit_diagnostics,
                )
                cells = _upsert_cell(cells, row, checkpoint_file)
    return cells


def compare_with_logistic(
    boosting_cells: pd.DataFrame, logistic_cells_file: str | Path
) -> pd.DataFrame:
    """Join the selected boosted cells to frozen adjusted-logistic results."""
    logistic = pd.read_csv(logistic_cells_file)
    logistic = logistic[
        [
            "train_start",
            "validation_year",
            "horizon",
            "adjusted_brier_known_source_prior",
        ]
    ].rename(
        columns={"adjusted_brier_known_source_prior": "logistic_brier"}
    )
    comparison = boosting_cells[
        ["train_start", "validation_year", "horizon", "adjusted_brier"]
    ].rename(columns={"adjusted_brier": "boosting_brier"})
    comparison = comparison.merge(
        logistic,
        on=["train_start", "validation_year", "horizon"],
        validate="one_to_one",
    )
    comparison["boosting_minus_logistic_brier"] = (
        comparison["boosting_brier"] - comparison["logistic_brier"]
    )
    return comparison


def save_boosting_model(
    model: BoostingDensityRatioModel, model_file: str | Path
) -> None:
    """Persist a complete boosted density-ratio fit."""
    model_file = Path(model_file)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    with model_file.open("wb") as file:
        pickle.dump(model, file)


def load_boosting_model(model_file: str | Path) -> BoostingDensityRatioModel:
    """Load a trusted local boosted density-ratio fit."""
    with Path(model_file).open("rb") as file:
        model = pickle.load(file)
    if not isinstance(model, BoostingDensityRatioModel):
        raise TypeError("Saved object is not a BoostingDensityRatioModel")
    return model


def boosting_model_path(
    train_years: Iterable[int],
    parameters: BoostingParameters,
    model_dir: str | Path = config.BOOSTING_FOLD_MODEL_DIR,
) -> Path:
    """Return the deterministic path for one fold/candidate artifact."""
    years = tuple(train_years)
    if not years:
        raise ValueError("train_years cannot be empty")
    return Path(model_dir) / (
        f"{parameters.identifier}__train_{min(years)}_{max(years)}.pkl"
    )


def _parameters_from_row(row: pd.Series) -> BoostingParameters:
    return BoostingParameters(
        max_leaf_nodes=int(row["max_leaf_nodes"]),
        learning_rate=float(row["learning_rate"]),
        max_iter=int(row["max_iter"]),
        l2_regularization=float(row["l2_regularization"]),
        min_samples_leaf=int(row["min_samples_leaf"]),
    )


def _binary_log_loss(y: np.ndarray, probability: np.ndarray) -> float:
    probability = np.clip(
        probability, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON
    )
    return float(-np.mean(y * np.log(probability) + (~y) * np.log1p(-probability)))


def _number_label(value: float) -> str:
    return format(value, ".12g").replace(".", "p")


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _cell_present(
    cells: pd.DataFrame,
    parameter_id: str,
    train_start: int,
    validation_year: int,
) -> bool:
    if cells.empty or "adjusted_brier" not in cells:
        return False
    matching = (
        (cells["parameter_id"] == parameter_id)
        & (cells["train_start"] == train_start)
        & (cells["validation_year"] == validation_year)
    )
    return bool((matching & cells["adjusted_brier"].notna()).any())


def _diagnostic_cell_present(
    frame: pd.DataFrame, train_start: int, validation_year: int
) -> bool:
    return not frame.empty and bool(
        (
            (frame["train_start"] == train_start)
            & (frame["validation_year"] == validation_year)
        ).any()
    )


def _replace_diagnostic_cell(
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


def _upsert_cell(
    existing: pd.DataFrame, new: pd.DataFrame, path: Path
) -> pd.DataFrame:
    row = new.iloc[0]
    if not existing.empty:
        keep = ~(
            (existing["parameter_id"] == row["parameter_id"])
            & (existing["train_start"] == row["train_start"])
            & (existing["validation_year"] == row["validation_year"])
        )
        existing = existing.loc[keep]
    combined = pd.concat([existing, new], ignore_index=True)
    combined.to_csv(path, index=False)
    return combined
