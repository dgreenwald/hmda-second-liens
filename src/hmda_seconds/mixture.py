"""Density-ratio estimation of target-year second-lien count shares."""

from __future__ import annotations

import copy
import pickle
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

from . import config, model_selection
from .density_ratio import folds as temporal_folds
from .logistic_features import FeatureSpecification, LogisticFeatureTransformer

YEAR_EFFECT_SCALE = 100.0
SHARE_BOUND = 1e-9
SHARE_TOLERANCE = 1e-8


@dataclass
class RatioVariant:
    """A normalized feature-density ratio for one source-model variant."""

    name: str
    feature_coefficients: np.ndarray
    log_ratio_offset: float
    mean_ratio_first: float
    mean_inverse_ratio_second: float

    def log_ratio(self, features: np.ndarray) -> np.ndarray:
        return features @ self.feature_coefficients + self.log_ratio_offset


@dataclass
class DensityRatioModels:
    """Fold-fitted transformer, raw classifier, and ratio variants."""

    transformer: LogisticFeatureTransformer
    raw_classifier: LogisticRegression
    pooled: RatioVariant
    year_fixed_effect: RatioVariant
    known_source_prior: RatioVariant
    source_year_diagnostics: pd.DataFrame
    fit_diagnostics: pd.DataFrame
    specification: FeatureSpecification
    regularization_c: float

    def features(self, frame: pd.DataFrame) -> np.ndarray:
        return self.transformer.transform(frame)

    def raw_probability(self, features: np.ndarray) -> np.ndarray:
        column = list(self.raw_classifier.classes_).index(
            config.SECOND_LIEN_CLASS
        )
        return self.raw_classifier.predict_proba(features)[:, column]


@dataclass
class KnownSourcePriorModel:
    """Frozen feature transform and equal-prior density-ratio classifier."""

    transformer: LogisticFeatureTransformer
    ratio: RatioVariant
    fit_diagnostics: dict
    specification: FeatureSpecification
    regularization_c: float
    train_years: tuple[int, ...]

    def log_ratio(self, frame: pd.DataFrame) -> np.ndarray:
        return self.ratio.log_ratio(self.transformer.transform(frame))


@dataclass(frozen=True)
class MixtureShareEstimate:
    """Bounded likelihood and EM estimates for one target sample."""

    share: float
    mean_log_likelihood: float
    optimizer_converged: bool
    optimizer_iterations: int
    at_boundary: bool
    em_share: float
    em_converged: bool
    em_iterations: int


def fit_density_ratio_models(
    training: pd.DataFrame,
    specification: FeatureSpecification,
    regularization_c: float,
    year_effect_scale: float = YEAR_EFFECT_SCALE,
) -> DensityRatioModels:
    """Fit pooled and source-year-intercept density-ratio models."""
    years = tuple(sorted(pd.unique(training["year"])))
    if len(years) < 2:
        raise ValueError("At least two source years are required for year effects")
    if year_effect_scale <= 1:
        raise ValueError("year_effect_scale must exceed one")

    transformer = LogisticFeatureTransformer(specification)
    features = transformer.fit_transform(training)
    labels = training[config.LABEL_VAR].to_numpy()
    y_second = labels == config.SECOND_LIEN_CLASS

    pooled_models, pooled_fit = model_selection.fit_regularization_path(
        features, labels, [regularization_c]
    )
    pooled_classifier = pooled_models[regularization_c]
    pooled_coefficients = pooled_classifier.coef_[0].copy()
    pooled_variant = _normalized_ratio_variant(
        "pooled", features, y_second, pooled_coefficients
    )

    year_matrix = _year_indicator_matrix(
        training["year"].to_numpy(), years, year_effect_scale
    )
    fixed_effect_features = np.column_stack([features, year_matrix])
    fixed_models, fixed_fit = model_selection.fit_regularization_path(
        fixed_effect_features, labels, [regularization_c]
    )
    fixed_classifier = fixed_models[regularization_c]
    n_features = features.shape[1]
    fixed_coefficients = fixed_classifier.coef_[0, :n_features].copy()
    fixed_variant = _normalized_ratio_variant(
        "year_fixed_effect", features, y_second, fixed_coefficients
    )

    # Reweight each source year to a 50/50 class prior. Bayes' rule then makes
    # the fitted log odds an estimate of log(f_1 / f_0), including its
    # intercept, rather than a posterior tied to the observed source prior.
    prior_weights = equal_source_prior_weights(training, y_second)
    prior_models, prior_fit = model_selection.fit_regularization_path(
        features, labels, [regularization_c], sample_weight=prior_weights
    )
    prior_classifier = prior_models[regularization_c]
    prior_variant = classifier_ratio_variant(
        "known_source_prior", features, y_second, prior_classifier
    )

    effective_year_coefficients = (
        fixed_classifier.coef_[0, n_features:] * year_effect_scale
    )
    source_year_diagnostics = _source_year_diagnostics(
        training,
        years,
        float(fixed_classifier.intercept_[0]),
        effective_year_coefficients,
        fixed_variant.log_ratio_offset,
    )
    fit_diagnostics = pd.DataFrame(
        [
            {
                "variant": "pooled",
                **pooled_fit[regularization_c],
                "mean_ratio_first": pooled_variant.mean_ratio_first,
                "mean_inverse_ratio_second": (
                    pooled_variant.mean_inverse_ratio_second
                ),
                "log_ratio_offset": pooled_variant.log_ratio_offset,
            },
            {
                "variant": "year_fixed_effect",
                **fixed_fit[regularization_c],
                "mean_ratio_first": fixed_variant.mean_ratio_first,
                "mean_inverse_ratio_second": (
                    fixed_variant.mean_inverse_ratio_second
                ),
                "log_ratio_offset": fixed_variant.log_ratio_offset,
                "year_effect_scale": year_effect_scale,
            },
            {
                "variant": "known_source_prior",
                **prior_fit[regularization_c],
                "mean_ratio_first": prior_variant.mean_ratio_first,
                "mean_inverse_ratio_second": (
                    prior_variant.mean_inverse_ratio_second
                ),
                "log_ratio_offset": prior_variant.log_ratio_offset,
            },
        ]
    )
    return DensityRatioModels(
        transformer=transformer,
        raw_classifier=copy.deepcopy(pooled_classifier),
        pooled=pooled_variant,
        year_fixed_effect=fixed_variant,
        known_source_prior=prior_variant,
        source_year_diagnostics=source_year_diagnostics,
        fit_diagnostics=fit_diagnostics,
        specification=specification,
        regularization_c=regularization_c,
    )


def save_density_ratio_models(
    model: DensityRatioModels, model_file: str | Path
) -> None:
    """Persist the complete pooled/fixed-effect/known-prior fold fit."""
    model_file = Path(model_file)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    with model_file.open("wb") as file:
        pickle.dump(model, file)


def load_density_ratio_models(model_file: str | Path) -> DensityRatioModels:
    """Load a trusted local complete ratio-model bundle."""
    with Path(model_file).open("rb") as file:
        model = pickle.load(file)
    if not isinstance(model, DensityRatioModels):
        raise TypeError("Saved object is not a DensityRatioModels bundle")
    return model


def density_ratio_models_path(
    train_years: Iterable[int],
    specification: FeatureSpecification,
    regularization_c: float,
    model_dir: str | Path = config.MIXTURE_FOLD_MODEL_DIR,
) -> Path:
    """Return the deterministic path for a complete mixture fold fit."""
    years = tuple(train_years)
    if not years:
        raise ValueError("train_years cannot be empty")
    c_label = format(regularization_c, ".12g").replace(".", "p")
    return Path(model_dir) / (
        f"all_ratio_variants__{specification.name}__c_{c_label}"
        f"__train_{min(years)}_{max(years)}.pkl"
    )


def fit_known_source_prior_model(
    training: pd.DataFrame,
    specification: FeatureSpecification,
    regularization_c: float,
    model_file: str | Path | None = None,
) -> KnownSourcePriorModel:
    """Fit only the frozen equal-source-prior ratio used after selection."""
    transformer = LogisticFeatureTransformer(specification)
    features = transformer.fit_transform(training)
    labels = training[config.LABEL_VAR].to_numpy()
    y_second = labels == config.SECOND_LIEN_CLASS
    prior_weights = equal_source_prior_weights(training, y_second)
    models, diagnostics = model_selection.fit_regularization_path(
        features, labels, [regularization_c], sample_weight=prior_weights
    )
    classifier = models[regularization_c]
    fitted = KnownSourcePriorModel(
        transformer=transformer,
        ratio=classifier_ratio_variant(
            "known_source_prior", features, y_second, classifier
        ),
        fit_diagnostics=diagnostics[regularization_c],
        specification=specification,
        regularization_c=regularization_c,
        train_years=tuple(sorted(pd.unique(training["year"]))),
    )
    if model_file is not None:
        save_known_source_prior_model(fitted, model_file)
    return fitted


def save_known_source_prior_model(
    model: KnownSourcePriorModel, model_file: str | Path
) -> None:
    """Persist every parameter needed to reproduce a fold's log ratios."""
    model_file = Path(model_file)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    with model_file.open("wb") as file:
        pickle.dump(model, file)


def load_known_source_prior_model(
    model_file: str | Path,
) -> KnownSourcePriorModel:
    """Load a trusted local known-prior fold model."""
    with Path(model_file).open("rb") as file:
        model = pickle.load(file)
    if not isinstance(model, KnownSourcePriorModel):
        raise TypeError("Saved object is not a KnownSourcePriorModel")
    return model


def known_source_prior_model_path(
    train_years: Iterable[int],
    specification: FeatureSpecification,
    regularization_c: float,
    model_dir: str | Path = config.MIXTURE_FOLD_MODEL_DIR,
) -> Path:
    """Return the deterministic artifact path for a frozen fold model."""
    years = tuple(train_years)
    if not years:
        raise ValueError("train_years cannot be empty")
    c_label = format(regularization_c, ".12g").replace(".", "p")
    return Path(model_dir) / (
        f"known_source_prior__{specification.name}__c_{c_label}"
        f"__train_{min(years)}_{max(years)}.pkl"
    )


def estimate_mixture_share(
    log_ratio: np.ndarray,
    tolerance: float = SHARE_TOLERANCE,
    max_em_iterations: int = 100,
) -> MixtureShareEstimate:
    """Estimate one target second-lien share by likelihood and EM."""
    log_ratio = _finite_vector(log_ratio, "log_ratio")

    def negative_mean_likelihood(share: float) -> float:
        return -mean_mixture_log_likelihood(log_ratio, share)

    result = minimize_scalar(
        negative_mean_likelihood,
        bounds=(SHARE_BOUND, 1.0 - SHARE_BOUND),
        method="bounded",
        options={"xatol": tolerance, "maxiter": 500},
    )
    share = float(result.x)
    at_boundary = bool(
        share <= 10 * SHARE_BOUND or share >= 1.0 - 10 * SHARE_BOUND
    )
    em_share, em_converged, em_iterations = estimate_mixture_share_em(
        log_ratio,
        # EM is a numerical consistency check, not the primary estimator.
        # Starting it at the bounded optimum avoids arbitrarily slow fixed-point
        # convergence when the likelihood optimum is close to a boundary.
        initial_share=share,
        tolerance=tolerance,
        max_iterations=max_em_iterations,
    )
    return MixtureShareEstimate(
        share=share,
        mean_log_likelihood=-float(result.fun),
        optimizer_converged=bool(result.success),
        optimizer_iterations=int(result.nfev),
        at_boundary=at_boundary,
        em_share=em_share,
        em_converged=em_converged,
        em_iterations=em_iterations,
    )


def estimate_mixture_share_em(
    log_ratio: np.ndarray,
    initial_share: float = 0.5,
    tolerance: float = SHARE_TOLERANCE,
    max_iterations: int = 10_000,
) -> tuple[float, bool, int]:
    """Estimate the target share by posterior-responsibility iteration."""
    log_ratio = _finite_vector(log_ratio, "log_ratio")
    if not 0 < initial_share < 1:
        raise ValueError("initial_share must lie strictly between zero and one")
    share = float(initial_share)
    for iteration in range(1, max_iterations + 1):
        updated = float(adjusted_probability(log_ratio, share).mean())
        if abs(updated - share) < tolerance:
            return updated, True, iteration
        share = float(np.clip(updated, SHARE_BOUND, 1.0 - SHARE_BOUND))
    return share, False, max_iterations


def adjusted_probability(log_ratio: np.ndarray, share: float) -> np.ndarray:
    """Return target posterior probabilities for a proposed target share."""
    if not 0 < share < 1:
        raise ValueError("share must lie strictly between zero and one")
    return expit(_finite_vector(log_ratio, "log_ratio") + logit(share))


def mean_mixture_log_likelihood(log_ratio: np.ndarray, share: float) -> float:
    """Mean target log likelihood after dropping the first-lien density."""
    if not 0 < share < 1:
        raise ValueError("share must lie strictly between zero and one")
    values = np.logaddexp(np.log1p(-share), np.log(share) + log_ratio)
    return float(values.mean())


def run_reverse_mixture_validation(
    data_dir: str | Path = config.SELECTION_DATA_DIR,
    output_dir: str | Path = config.TABLE_DIR,
    model_file: str | Path = config.SELECTED_LOGISTIC_MODEL_FILE,
    train_starts: Iterable[int] | None = None,
    fold_model_dir: str | Path = config.MIXTURE_FOLD_MODEL_DIR,
) -> dict[str, pd.DataFrame]:
    """Run resumable density-ratio share validation over all reverse cells."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = model_selection.load_selected_model(model_file)
    data_by_year = model_selection.load_selection_years(data_dir=data_dir)
    cells_file = output_dir / "mixture_reverse_cell_shares.csv"
    intercepts_file = output_dir / "mixture_source_year_intercepts.csv"
    ratio_file = output_dir / "mixture_ratio_fit_diagnostics.csv"
    cells = _read_csv_if_exists(cells_file)
    intercepts = _read_csv_if_exists(intercepts_file)
    ratio_diagnostics = _read_csv_if_exists(ratio_file)

    folds = list(reversed(temporal_folds.reverse_folds()))
    if train_starts is not None:
        requested = set(train_starts)
        available = {fold.train_start for fold in folds}
        unknown = requested - available
        if unknown:
            raise ValueError(f"Unknown reverse-fold starts: {sorted(unknown)}")
        folds = [fold for fold in folds if fold.train_start in requested]

    for fold in folds:
        missing_years = [
            year
            for year in fold.validation_years
            if not _cell_present(cells, fold.train_start, year)
        ]
        diagnostics_complete = _fold_present(intercepts, fold.train_start) and (
            _fold_present(ratio_diagnostics, fold.train_start)
        )
        if not missing_years and diagnostics_complete:
            continue
        fold_model_file = density_ratio_models_path(
            fold.train_years,
            selected.specification,
            selected.regularization_c,
            fold_model_dir,
        )
        if fold_model_file.exists():
            fitted = load_density_ratio_models(fold_model_file)
        else:
            training = pd.concat(
                [data_by_year[year] for year in fold.train_years],
                ignore_index=True,
            )
            fitted = fit_density_ratio_models(
                training, selected.specification, selected.regularization_c
            )
            save_density_ratio_models(fitted, fold_model_file)
        source_metadata = {
            "specification": selected.specification.name,
            "regularization_c": selected.regularization_c,
            "train_start": fold.train_start,
            "train_end": fold.train_end,
        }
        fold_intercepts = fitted.source_year_diagnostics.assign(**source_metadata)
        intercepts = _replace_fold_checkpoint(
            intercepts, fold_intercepts, intercepts_file, fold.train_start
        )
        fold_ratio = fitted.fit_diagnostics.assign(**source_metadata)
        ratio_diagnostics = _replace_fold_checkpoint(
            ratio_diagnostics, fold_ratio, ratio_file, fold.train_start
        )

        for validation_year in missing_years:
            row = _evaluate_target_year(
                fitted,
                data_by_year[validation_year],
                fold,
                validation_year,
            )
            cells = _upsert_cell_checkpoint(cells, row, cells_file)

    horizon_summary, overall_summary = aggregate_share_errors(cells)
    horizon_summary.to_csv(
        output_dir / "mixture_reverse_horizon_summary.csv", index=False
    )
    overall_summary.to_csv(
        output_dir / "mixture_reverse_estimator_summary.csv", index=False
    )
    return {
        "cells": cells,
        "horizons": horizon_summary,
        "summary": overall_summary,
        "source_intercepts": intercepts,
        "ratio_diagnostics": ratio_diagnostics,
    }


def aggregate_share_errors(
    cells: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize share estimators within and then equally across horizons."""
    estimate_columns = {
        "raw_mean_probability": "raw_mean_probability",
        "raw_hard_share": "raw_hard_share",
        "mixture_pooled": "mixture_share_pooled",
        "mixture_year_fixed_effect": "mixture_share_year_fixed_effect",
        "mixture_known_source_prior": "mixture_share_known_source_prior",
        "adjusted_hard_pooled": "adjusted_hard_share_pooled",
        "adjusted_hard_year_fixed_effect": (
            "adjusted_hard_share_year_fixed_effect"
        ),
        "adjusted_hard_known_source_prior": (
            "adjusted_hard_share_known_source_prior"
        ),
    }
    parts = []
    for estimator, column in estimate_columns.items():
        part = cells[["train_start", "validation_year", "horizon"]].copy()
        part["estimator"] = estimator
        part["signed_error"] = cells[column] - cells["actual_second_share"]
        part["absolute_error"] = part["signed_error"].abs()
        part["squared_error"] = part["signed_error"] ** 2
        parts.append(part)
    errors = pd.concat(parts, ignore_index=True)
    horizons = (
        errors.groupby(["estimator", "horizon"], as_index=False)
        .agg(
            mean_signed_error=("signed_error", "mean"),
            mean_absolute_error=("absolute_error", "mean"),
            mean_squared_error=("squared_error", "mean"),
            n_cells=("validation_year", "size"),
        )
        .sort_values(["estimator", "horizon"])
    )
    summary = (
        horizons.groupby("estimator", as_index=False)
        .agg(
            selection_signed_error=("mean_signed_error", "mean"),
            selection_absolute_error=("mean_absolute_error", "mean"),
            selection_squared_error=("mean_squared_error", "mean"),
            n_horizons=("horizon", "nunique"),
            n_cells=("n_cells", "sum"),
        )
        .sort_values("selection_absolute_error")
        .reset_index(drop=True)
    )
    return horizons, summary


def _evaluate_target_year(
    fitted: DensityRatioModels,
    target: pd.DataFrame,
    fold: temporal_folds.TemporalFold,
    validation_year: int,
) -> pd.DataFrame:
    # Local import avoids a module cycle: the shared evaluator retains these
    # public mixture-estimation primitives as backward-compatible dependencies.
    from .density_ratio import evaluation

    features = fitted.features(target)
    raw_probability = fitted.raw_probability(features)
    actual_second = (
        target[config.LABEL_VAR].to_numpy() == config.SECOND_LIEN_CLASS
    )
    actual_share = float(actual_second.mean())
    raw_metrics = evaluation.probability_metrics(actual_second, raw_probability)
    row = {
        "specification": fitted.specification.name,
        "regularization_c": fitted.regularization_c,
        "train_start": fold.train_start,
        "train_end": fold.train_end,
        "validation_year": validation_year,
        "horizon": fold.horizon_for(validation_year),
        "n_validation": len(target),
        "actual_second_share": actual_share,
        "raw_mean_probability": float(raw_probability.mean()),
        "raw_hard_share": float((raw_probability >= 0.5).mean()),
        "raw_brier": raw_metrics["brier_score"],
        "raw_log_loss": raw_metrics["log_loss"],
    }
    for variant in (
        fitted.pooled,
        fitted.year_fixed_effect,
        fitted.known_source_prior,
    ):
        log_ratio = variant.log_ratio(features)
        evaluated = evaluation.evaluate_log_ratio(
            log_ratio,
            actual_second,
            model_id=(
                f"logistic__{variant.name}__{fitted.specification.name}"
                f"__train_{fold.train_start}_{fold.train_end}"
            ),
            fold=fold,
            target_year=validation_year,
        )
        estimate = evaluated.mixture_estimate
        result = evaluated.result
        suffix = variant.name
        row.update(
            {
                f"mixture_share_{suffix}": estimate.share,
                f"mixture_signed_error_{suffix}": estimate.share - actual_share,
                f"mixture_absolute_error_{suffix}": abs(
                    estimate.share - actual_share
                ),
                f"adjusted_hard_share_{suffix}": result.hard_share_050,
                f"adjusted_brier_{suffix}": result.brier_score,
                f"adjusted_log_loss_{suffix}": result.log_loss,
                f"mean_log_likelihood_{suffix}": (
                    estimate.mean_log_likelihood
                ),
                f"optimizer_converged_{suffix}": estimate.optimizer_converged,
                f"optimizer_iterations_{suffix}": estimate.optimizer_iterations,
                f"at_boundary_{suffix}": estimate.at_boundary,
                f"em_share_{suffix}": estimate.em_share,
                f"em_converged_{suffix}": estimate.em_converged,
                f"em_iterations_{suffix}": estimate.em_iterations,
                f"optimizer_em_difference_{suffix}": (
                    estimate.share - estimate.em_share
                ),
            }
        )
    return pd.DataFrame([row])


def _normalized_ratio_variant(
    name: str,
    features: np.ndarray,
    y_second: np.ndarray,
    coefficients: np.ndarray,
) -> RatioVariant:
    score = features @ coefficients
    first_score = score[~y_second]
    second_score = score[y_second]
    offset = -_log_mean_exp(first_score)
    mean_ratio_first = float(np.exp(_log_mean_exp(first_score + offset)))
    mean_inverse_ratio_second = float(
        np.exp(_log_mean_exp(-second_score - offset))
    )
    return RatioVariant(
        name=name,
        feature_coefficients=coefficients,
        log_ratio_offset=offset,
        mean_ratio_first=mean_ratio_first,
        mean_inverse_ratio_second=mean_inverse_ratio_second,
    )


def classifier_ratio_variant(
    name: str,
    features: np.ndarray,
    y_second: np.ndarray,
    classifier: LogisticRegression,
) -> RatioVariant:
    """Use the full equal-prior classifier predictor as a density ratio."""
    coefficients = classifier.coef_[0].copy()
    offset = float(classifier.intercept_[0])
    score = features @ coefficients + offset
    return RatioVariant(
        name=name,
        feature_coefficients=coefficients,
        log_ratio_offset=offset,
        mean_ratio_first=float(np.exp(_log_mean_exp(score[~y_second]))),
        mean_inverse_ratio_second=float(
            np.exp(_log_mean_exp(-score[y_second]))
        ),
    )


def equal_source_prior_weights(
    training: pd.DataFrame, y_second: np.ndarray
) -> np.ndarray:
    """Give both lien classes half of each source year's total weight."""
    weights = np.empty(len(training), dtype=float)
    years = training["year"].to_numpy()
    for year in np.unique(years):
        in_year = years == year
        share = float(y_second[in_year].mean())
        if not 0 < share < 1:
            raise ValueError(f"Source year {year} must contain both lien classes")
        weights[in_year & y_second] = 0.5 / share
        weights[in_year & ~y_second] = 0.5 / (1.0 - share)
    return weights


def _source_year_diagnostics(
    training: pd.DataFrame,
    years: tuple[int, ...],
    reference_intercept: float,
    effective_year_coefficients: np.ndarray,
    ratio_offset: float,
) -> pd.DataFrame:
    rows = []
    for index, year in enumerate(years):
        sample = training.loc[training["year"] == year, config.LABEL_VAR]
        share = float((sample == config.SECOND_LIEN_CLASS).mean())
        year_effect = 0.0 if index == 0 else effective_year_coefficients[index - 1]
        fitted_intercept = reference_intercept + year_effect
        adjusted_intercept = fitted_intercept - logit(share)
        rows.append(
            {
                "source_year": year,
                "n_source": len(sample),
                "observed_second_share": share,
                "observed_logit_share": float(logit(share)),
                "fitted_year_intercept": fitted_intercept,
                "intercept_minus_logit_share": adjusted_intercept,
                "log_ratio_offset": ratio_offset,
                "normalization_gap": adjusted_intercept - ratio_offset,
            }
        )
    return pd.DataFrame(rows)


def _year_indicator_matrix(
    observed_years: np.ndarray,
    levels: tuple[int, ...],
    scale: float,
) -> np.ndarray:
    unknown = set(np.unique(observed_years)) - set(levels)
    if unknown:
        raise ValueError(f"Unknown source years: {sorted(unknown)}")
    return np.column_stack(
        [(observed_years == year).astype(float) * scale for year in levels[1:]]
    )


def _log_mean_exp(values: np.ndarray) -> float:
    values = _finite_vector(values, "values")
    maximum = float(values.max())
    return maximum + float(np.log(np.exp(values - maximum).mean()))


def _finite_vector(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _cell_present(
    cells: pd.DataFrame, train_start: int, validation_year: int
) -> bool:
    required = "mixture_share_known_source_prior"
    if cells.empty or required not in cells:
        return False
    matching = (
        (cells["train_start"] == train_start)
        & (cells["validation_year"] == validation_year)
    )
    return bool((matching & cells[required].notna()).any())


def _fold_present(frame: pd.DataFrame, train_start: int) -> bool:
    return not frame.empty and bool((frame["train_start"] == train_start).any())


def _replace_fold_checkpoint(
    existing: pd.DataFrame,
    new: pd.DataFrame,
    path: Path,
    train_start: int,
) -> pd.DataFrame:
    if not existing.empty:
        existing = existing.loc[existing["train_start"] != train_start]
    combined = pd.concat([existing, new], ignore_index=True)
    combined.to_csv(path, index=False)
    return combined


def _upsert_cell_checkpoint(
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
