"""Small numerical primitives shared across density-ratio model families."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class ProbabilisticClassifier(Protocol):
    """Structural type for classifiers exposing class probabilities."""

    classes_: np.ndarray

    def predict_proba(self, features: object) -> np.ndarray: ...


def finite_vector(values: np.ndarray, name: str) -> np.ndarray:
    """Return a nonempty one-dimensional finite float array."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def log_mean_exp(values: np.ndarray) -> float:
    """Compute the log of the mean exponential without avoidable overflow."""
    array = finite_vector(values, "values")
    maximum = float(array.max())
    return maximum + float(np.log(np.exp(array - maximum).mean()))


def predict_class_probability(
    classifier: ProbabilisticClassifier,
    features: object,
    target_class: object,
) -> np.ndarray:
    """Return the probability column corresponding to ``target_class``."""
    classes = np.asarray(classifier.classes_)
    matches = np.flatnonzero(classes == target_class)
    if len(matches) != 1:
        raise ValueError(f"Classifier does not contain unique class {target_class!r}")
    probability = np.asarray(classifier.predict_proba(features), dtype=float)
    if probability.ndim != 2 or probability.shape[1] != len(classes):
        raise ValueError("predict_proba output does not align with classifier classes")
    return probability[:, int(matches[0])]
