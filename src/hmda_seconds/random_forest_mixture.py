"""Fixed Random Forest challenger with target-year mixture adjustment."""

from __future__ import annotations

import pickle
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import parallel_backend
from scipy.special import logit
from sklearn.ensemble import RandomForestClassifier

from . import calibration, config, mixture, model_selection
from .density_ratio import adapters, evaluation
from .density_ratio import folds as temporal_folds

PROBABILITY_EPSILON = 1e-12
ESTIMATOR = "random_forest_mixture"


@dataclass
class RandomForestDensityRatioModel:
    """Persisted equal-source-prior forest and its fixed feature schema."""

    classifier: RandomForestClassifier
    train_years: tuple[int, ...]
    feature_names: tuple[str, ...]

    def log_ratio(self, frame: pd.DataFrame) -> np.ndarray:
        """Return clipped balanced-prior forest log odds."""
        features, names = forest_features(frame)
        if names != self.feature_names:
            raise RuntimeError("Random Forest feature columns changed")
        second_column = list(self.classifier.classes_).index(
            config.SECOND_LIEN_CLASS
        )
        with parallel_backend("threading"):
            probability = self.classifier.predict_proba(features)[
                :, second_column
            ]
        probability = np.clip(
            probability, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON
        )
        return logit(probability)


def forest_features(frame: pd.DataFrame) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build raw continuous and full one-hot categorical RF features."""
    required = [*config.CONTINUOUS_VARS, *config.CATEGORY_VARS]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing Random Forest features: {sorted(missing)}")
    blocks = []
    names = []
    for variable in config.CONTINUOUS_VARS:
        values = frame[variable].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"{variable} contains non-finite values")
        blocks.append(values[:, None])
        names.append(variable)
    for variable in config.CATEGORY_VARS:
        values = frame[variable].to_numpy()
        levels = config.CATEGORY_LEVELS[variable]
        unknown = set(pd.unique(values)) - set(levels)
        if unknown:
            raise ValueError(f"Unknown {variable} levels: {sorted(unknown)}")
        blocks.append((values[:, None] == np.asarray(levels)).astype(np.float32))
        names.extend(f"{variable}_{level}" for level in levels)
    return np.column_stack(blocks), tuple(names)


def fit_forest_ratio_model(
    training: pd.DataFrame,
) -> tuple[RandomForestDensityRatioModel, dict]:
    """Fit the fixed RF with equal class-prior mass in every source year."""
    features, names = forest_features(training)
    labels = training[config.LABEL_VAR].to_numpy()
    is_second = labels == config.SECOND_LIEN_CLASS
    weights = mixture.equal_source_prior_weights(training, is_second)
    classifier = RandomForestClassifier(**config.RF_KWARGS)
    start = time.perf_counter()
    # Force thread-based tree parallelism. Process-based joblib can memmap the
    # multi-million-row weight vector and produced a corrupted apparent shape
    # after several consecutive fold fits in the real-data run.
    with parallel_backend("threading"):
        classifier.fit(features, labels, sample_weight=weights)
    fit_seconds = time.perf_counter() - start
    model = RandomForestDensityRatioModel(
        classifier=classifier,
        train_years=tuple(sorted(pd.unique(training["year"]))),
        feature_names=names,
    )
    return model, {"fit_seconds": fit_seconds}


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
    reverse_metrics = _read(reverse_metrics_file)
    reverse_bins = _read(reverse_bins_file)
    for fold in reversed(temporal_folds.reverse_folds()):
        path = forest_model_path(fold.train_years, fold_model_dir)
        if path.exists():
            model = load_forest_model(path)
        else:
            training = pd.concat(
                [data_by_year[year] for year in fold.train_years],
                ignore_index=True,
            )
            model, _ = fit_forest_ratio_model(training)
            save_forest_model(model, path)
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
    forward_metrics = _read(forward_metrics_file)
    forward_bins = _read(forward_bins_file)
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
    forward_summary = _simple_summary(forward_metrics)
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


def estimator_comparison(
    forest: pd.DataFrame,
    logistic_file: str | Path,
    boosting_file: str | Path,
) -> pd.DataFrame:
    """Join all three mixture estimators' reverse-cell Brier scores."""
    keys = ["train_start", "validation_year", "horizon"]
    result = forest[[*keys, "brier_score"]].rename(
        columns={"brier_score": "forest_brier"}
    )
    logistic = pd.read_csv(logistic_file)[[*keys, "brier_score"]].rename(
        columns={"brier_score": "logistic_brier"}
    )
    boosting = pd.read_csv(boosting_file)[[*keys, "brier_score"]].rename(
        columns={"brier_score": "boosting_brier"}
    )
    result = result.merge(logistic, on=keys, validate="one_to_one")
    result = result.merge(boosting, on=keys, validate="one_to_one")
    result["forest_minus_logistic_brier"] = (
        result["forest_brier"] - result["logistic_brier"]
    )
    result["forest_minus_boosting_brier"] = (
        result["forest_brier"] - result["boosting_brier"]
    )
    return result


def save_forest_model(
    model: RandomForestDensityRatioModel, model_file: str | Path
) -> None:
    """Persist the complete forest density-ratio model."""
    model_file = Path(model_file)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    with model_file.open("wb") as file:
        pickle.dump(model, file)


def load_forest_model(model_file: str | Path) -> RandomForestDensityRatioModel:
    """Load a trusted local forest density-ratio model."""
    with Path(model_file).open("rb") as file:
        model = pickle.load(file)
    if not isinstance(model, RandomForestDensityRatioModel):
        raise TypeError("Saved object is not a RandomForestDensityRatioModel")
    return model


def forest_model_path(
    train_years: Iterable[int],
    model_dir: str | Path = config.RF_MIXTURE_FOLD_MODEL_DIR,
) -> Path:
    """Return the deterministic artifact path for one reverse source window."""
    years = tuple(train_years)
    if not years:
        raise ValueError("train_years cannot be empty")
    return Path(model_dir) / (
        f"rf_50_depth_10__train_{min(years)}_{max(years)}.pkl"
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
        adapters.adapt_random_forest_model(model),
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
    metric_row = pd.DataFrame(
        [
            {
                **metadata,
                **evaluation.probability_metrics(y_second, probability),
                "mixture_share": evaluated.result.mixture_share,
                "adjusted_hard_share_050": evaluated.result.hard_share_050,
                "share_optimizer_converged": estimate.optimizer_converged,
                "share_at_boundary": estimate.at_boundary,
                "mixture_em_difference": estimate.share - estimate.em_share,
            }
        ]
    )
    bin_rows = calibration.reliability_bins(
        y_second, probability, n_bins
    ).assign(**metadata)
    metrics = _replace_cell(metrics, metric_row, metrics_file)
    bins = _replace_cell(bins, bin_rows, bins_file)
    return metrics, bins


def _simple_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                **{
                    column: metrics[column].mean()
                    for column in calibration.METRIC_COLUMNS
                },
                "n_cells": len(metrics),
                "weighting": "equal_across_validation_years",
            }
        ]
    )


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _cell_present(
    frame: pd.DataFrame, train_start: int, validation_year: int
) -> bool:
    return not frame.empty and bool(
        (
            (frame["train_start"] == train_start)
            & (frame["validation_year"] == validation_year)
        ).any()
    )


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
