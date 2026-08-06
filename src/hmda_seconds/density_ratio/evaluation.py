"""Shared mixture adjustment and target-cell evaluation."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import mixture
from .protocols import EvaluationResult, FittedDensityRatioModel, TemporalFold

PROBABILITY_FLOOR = 1e-12
CANONICAL_THRESHOLD = 0.5


@dataclass(frozen=True)
class EvaluatedTarget:
    """Aggregate result plus transient arrays needed by diagnostics."""

    result: EvaluationResult
    mixture_estimate: mixture.MixtureShareEstimate
    log_ratio: np.ndarray
    probability: np.ndarray
    metrics: dict[str, float | int]
    log_ratio_seconds: float
    evaluation_seconds: float


@dataclass(frozen=True)
class AdjustedProbabilities:
    """Transient mixture estimate and probabilities for an unlabeled sample."""

    mixture_estimate: mixture.MixtureShareEstimate
    probability: np.ndarray


def evaluate_target(
    fitted: FittedDensityRatioModel,
    target: pd.DataFrame,
    fold: TemporalFold,
    *,
    label_var: str,
    second_lien_class: int,
    threshold: float = CANONICAL_THRESHOLD,
) -> EvaluatedTarget:
    """Evaluate one fitted density-ratio model on one labeled target year."""
    if fitted.train_years != fold.train_years:
        raise ValueError("fitted model source years do not match the fold")
    target_year = _target_year(target, fold)
    started = time.perf_counter()
    log_ratio = fitted.log_ratio(target)
    log_ratio_seconds = time.perf_counter() - started
    return evaluate_log_ratio(
        log_ratio,
        target[label_var].to_numpy() == second_lien_class,
        model_id=fitted.model_id,
        fold=fold,
        target_year=target_year,
        threshold=threshold,
        log_ratio_seconds=log_ratio_seconds,
        started=started,
    )


def evaluate_log_ratio(
    log_ratio: np.ndarray,
    y_second: np.ndarray,
    *,
    model_id: str,
    fold: TemporalFold,
    target_year: int,
    threshold: float = CANONICAL_THRESHOLD,
    log_ratio_seconds: float = 0.0,
    started: float | None = None,
) -> EvaluatedTarget:
    """Apply the common mixture and metric logic to precomputed log ratios."""
    if target_year not in fold.target_years:
        raise ValueError(f"Year {target_year} is not a target in {fold.fold_id}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    log_ratio = _finite_vector(log_ratio, "log_ratio")
    y_second = np.asarray(y_second, dtype=bool)
    if y_second.ndim != 1 or len(y_second) != len(log_ratio):
        raise ValueError("labels and log ratios must be aligned one-dimensional arrays")
    if not len(y_second):
        raise ValueError("target arrays cannot be empty")
    if started is None:
        started = time.perf_counter()

    adjusted = adjust_log_ratio(log_ratio)
    estimate = adjusted.mixture_estimate
    probability = adjusted.probability
    metrics = evaluate_sample(y_second, probability)
    result = EvaluationResult(
        model_id=model_id,
        fold_id=fold.fold_id,
        target_year=target_year,
        horizon=fold.horizon_for(target_year),
        n_observations=len(y_second),
        actual_second_share=metrics["observed_second_share"],
        mixture_share=estimate.share,
        mean_probability=metrics["mean_predicted_second_share"],
        hard_share_050=float((probability >= threshold).mean()),
        brier_score=metrics["brier_score"],
        log_loss=metrics["log_loss"],
        calibration_mean_error=_finite_or_none(metrics["calibration_mean_error"]),
        calibration_intercept=_finite_or_none(metrics["calibration_intercept"]),
        calibration_slope=_finite_or_none(metrics["calibration_slope"]),
        optimizer_converged=estimate.optimizer_converged,
        mixture_at_boundary=estimate.at_boundary,
        mixture_em_difference=estimate.share - estimate.em_share,
    )
    return EvaluatedTarget(
        result=result,
        mixture_estimate=estimate,
        log_ratio=log_ratio,
        probability=probability,
        metrics=metrics,
        log_ratio_seconds=log_ratio_seconds,
        evaluation_seconds=time.perf_counter() - started,
    )


def adjust_log_ratio(log_ratio: np.ndarray) -> AdjustedProbabilities:
    """Estimate the mixture share and return adjusted probabilities."""
    log_ratio = _finite_vector(log_ratio, "log_ratio")
    estimate = mixture.estimate_mixture_share(log_ratio)
    probability = mixture.adjusted_probability(log_ratio, estimate.share)
    return AdjustedProbabilities(estimate, probability)


def evaluate_sample(
    y_second: np.ndarray, probability: np.ndarray
) -> dict[str, float | int]:
    """Compute the canonical metrics for one labeled probability sample."""
    y = np.asarray(y_second, dtype=bool)
    probability = np.asarray(probability, dtype=float)
    validate_probability_inputs(y, probability)
    clipped = np.clip(probability, PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR)
    intercept, slope = calibration_coefficients(y, probability)
    observed = float(y.mean())
    predicted = float(probability.mean())
    return {
        "n": len(y),
        "brier_score": float(np.mean((probability - y) ** 2)),
        "log_loss": float(
            -np.mean(y * np.log(clipped) + (~y) * np.log1p(-clipped))
        ),
        "observed_second_share": observed,
        "mean_predicted_second_share": predicted,
        "calibration_mean_error": predicted - observed,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


# Compatibility alias for callers using the former public name.
probability_metrics = evaluate_sample


def metric_record(
    y_second: np.ndarray,
    probability: np.ndarray,
    *,
    metadata: Mapping[str, object],
    additional: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Combine canonical sample metrics with cell metadata and extensions."""
    metrics = evaluate_sample(y_second, probability)
    return metric_record_from_metrics(metrics, metadata=metadata, additional=additional)


def metric_record_from_metrics(
    metrics: Mapping[str, float | int],
    *,
    metadata: Mapping[str, object],
    additional: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build a cell record from metrics already computed by this module."""
    additional = {} if additional is None else additional
    groups = (metadata, metrics, additional)
    keys = [key for group in groups for key in group]
    duplicates = {key for key in keys if keys.count(key) > 1}
    if duplicates:
        names = ", ".join(sorted(duplicates))
        raise ValueError(f"metric record fields overlap: {names}")
    return {**metadata, **metrics, **additional}


def validate_probability_inputs(
    y_second: np.ndarray, probability: np.ndarray
) -> None:
    """Validate aligned binary labels and finite unit-interval probabilities."""
    if y_second.ndim != 1 or probability.ndim != 1:
        raise ValueError("Labels and probabilities must be one-dimensional")
    if len(y_second) != len(probability) or not len(y_second):
        raise ValueError("Labels and probabilities must be nonempty and aligned")
    if not np.isfinite(probability).all():
        raise ValueError("Probabilities must be finite")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("Probabilities must lie in [0, 1]")
    if np.unique(y_second).size < 2:
        raise ValueError("Calibration sample must contain both outcome classes")


def calibration_coefficients(
    y_true_second: np.ndarray,
    probability: np.ndarray,
    tolerance: float = 1e-10,
    max_iter: int = 100,
) -> tuple[float, float]:
    """Fit ``y ~ intercept + slope * logit(p)`` by Newton iteration."""
    y = np.asarray(y_true_second, dtype=float)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-12, 1 - 1e-12)
    score = np.log(probability / (1.0 - probability))
    prevalence = np.clip(y.mean(), 1e-12, 1 - 1e-12)
    coefficients = np.array([np.log(prevalence / (1.0 - prevalence)), 0.0])

    def objective(candidate: np.ndarray) -> float:
        linear = candidate[0] + candidate[1] * score
        return float(np.sum(y * linear - np.logaddexp(0.0, linear)))

    for _ in range(max_iter):
        linear = coefficients[0] + coefficients[1] * score
        fitted = np.empty_like(linear)
        positive = linear >= 0
        fitted[positive] = 1.0 / (1.0 + np.exp(-linear[positive]))
        exp_linear = np.exp(linear[~positive])
        fitted[~positive] = exp_linear / (1.0 + exp_linear)
        weight = fitted * (1.0 - fitted)
        gradient = np.array([(y - fitted).sum(), np.dot(score, y - fitted)])
        information = np.array(
            [
                [weight.sum(), np.dot(weight, score)],
                [np.dot(weight, score), np.dot(weight, score * score)],
            ]
        )
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError:
            return np.nan, np.nan
        current_objective = objective(coefficients)
        step_scale = 1.0
        while step_scale >= 2.0**-20:
            candidate = coefficients + step_scale * step
            if objective(candidate) >= current_objective - 1e-8:
                break
            step_scale /= 2.0
        else:
            return np.nan, np.nan
        accepted_step = step_scale * step
        coefficients = candidate
        if np.max(np.abs(accepted_step)) < tolerance:
            break
    return float(coefficients[0]), float(coefficients[1])


def _target_year(target: pd.DataFrame, fold: TemporalFold) -> int:
    if "year" not in target:
        raise ValueError("target frame is missing year")
    years = pd.unique(target["year"])
    if len(years) != 1:
        raise ValueError("target frame must contain exactly one year")
    year = int(years[0])
    if year not in fold.target_years:
        raise ValueError(f"Year {year} is not a target in {fold.fold_id}")
    return year


def _finite_vector(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")
    return values


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None
