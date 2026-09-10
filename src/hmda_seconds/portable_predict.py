"""Standalone frozen logistic prediction; requires only NumPy and Python 3.11+."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1
SPECIFICATION = "spline_lti__purchaser_type"
CONTINUOUS = ("log_lti", "log_county_value_to_loan")
CATEGORIES = {"purchaser_type": list(range(10)), "loan_type": [1, 2, 3, 4]}


def _array(value, name, shape=None):
    array = np.asarray(value, dtype=float)
    if not np.isfinite(array).all() or (shape is not None and array.shape != shape):
        raise ValueError(f"Invalid shape or non-finite values for {name}")
    return array


def _basis(values, knots):
    # Keep the same operations and ordering as LogisticFeatureTransformer.
    columns = [values]
    penultimate, last = knots[-2:]
    denominator = last - penultimate
    scale = (last - knots[0]) ** 2
    for knot in knots[:-2]:
        term = np.maximum(values - knot, 0.0) ** 3
        term -= ((last - knot) / denominator) * np.maximum(
            values - penultimate, 0.0
        ) ** 3
        term += ((penultimate - knot) / denominator) * np.maximum(
            values - last, 0.0
        ) ** 3
        columns.append(term / scale)
    return np.column_stack(columns)


def _feature_names():
    names = [f"log_lti_rcs_{i}" for i in range(1, 4)]
    names.append("log_county_value_to_loan")
    for variable, levels in CATEGORIES.items():
        names.extend(f"{variable}_{level}" for level in levels[1:])
    for variable in CONTINUOUS:
        names.extend(f"{variable}_x_purchaser_type_{i}" for i in range(1, 10))
    return names


class PortableLogisticModel:
    """One fixed feature function with saved, total intercepts by target year."""

    def __init__(self, payload):
        try:
            self._initialize(payload)
        except (KeyError, TypeError, OverflowError) as exc:
            raise ValueError(f"Malformed portable model: {exc}") from exc

    def _initialize(self, payload):
        if payload["format_version"] != FORMAT_VERSION:
            raise ValueError("Unsupported portable model format version")
        if payload["specification"] != SPECIFICATION:
            raise ValueError("Unsupported logistic specification")
        if payload["category_levels"] != CATEGORIES:
            raise ValueError("Unsupported category levels or reference categories")
        if payload["feature_names"] != _feature_names():
            raise ValueError("Incorrect feature ordering")
        self.coefficients = _array(
            payload["coefficients"], "coefficients", (len(_feature_names()),)
        )
        self.knots = _array(payload["knots"], "knots", (4,))
        if np.any(np.diff(self.knots) <= 0):
            raise ValueError("Spline knots must be strictly increasing")
        self.raw_location = _array(payload["raw_location"], "raw_location", (2,))
        self.raw_scale = _array(payload["raw_scale"], "raw_scale", (2,))
        self.basis_location = _array(payload["basis_location"], "basis_location", (3,))
        self.basis_scale = _array(payload["basis_scale"], "basis_scale", (3,))
        if np.any(self.raw_scale <= 0) or np.any(self.basis_scale <= 0):
            raise ValueError("Scales must be positive")
        offset = float(_array(payload["density_ratio_offset"], "offset", ()))
        self.intercepts = {}
        for year, row in payload["annual"].items():
            if str(int(year)) != year:
                raise ValueError("Annual keys must be integer years")
            share = float(_array(row["mixture_share"], "mixture_share", ()))
            intercept = float(_array(row["intercept"], "intercept", ()))
            if not 0 < share < 1:
                raise ValueError("Mixture shares must be strictly between zero and one")
            expected = offset + np.log(share) - np.log1p(-share)
            if not np.isclose(intercept, expected, rtol=0, atol=1e-12):
                raise ValueError("Annual intercept is inconsistent with mixture share")
            self.intercepts[int(year)] = intercept
        if not self.intercepts:
            raise ValueError("At least one annual intercept is required")

    def _features(self, inputs):
        values = {}
        size = None
        for name in (*CONTINUOUS, *CATEGORIES):
            try:
                array = _array(inputs[name], name)
            except KeyError as exc:
                raise ValueError(f"Missing input column: {name}") from exc
            if array.ndim != 1 or (size is not None and len(array) != size):
                raise ValueError("Input columns must be aligned one-dimensional arrays")
            size = len(array)
            values[name] = array
        standardized = [
            (values[name] - self.raw_location[i]) / self.raw_scale[i]
            for i, name in enumerate(CONTINUOUS)
        ]
        blocks = [
            (_basis(values["log_lti"], self.knots) - self.basis_location)
            / self.basis_scale,
            standardized[1][:, None],
        ]
        indicators = {}
        for name, levels in CATEGORIES.items():
            if not np.isin(values[name], levels).all():
                raise ValueError(f"Unknown {name} category")
            indicators[name] = np.column_stack(
                [values[name] == level for level in levels[1:]]
            ).astype(float)
            blocks.append(indicators[name])
        blocks.extend(x[:, None] * indicators["purchaser_type"] for x in standardized)
        features = np.column_stack(blocks)
        if not np.isfinite(features).all():
            raise ValueError("Input magnitude caused non-finite transformed features")
        return features

    def predict_proba_second_lien(self, inputs, *, year):
        """Predict with saved intercepts; year is a scalar or an aligned vector."""
        features = self._features(inputs)
        years = _array(year, "year")
        if years.ndim == 0:
            years = np.full(len(features), years.item())
            # Validate scalar years even for an empty batch.
            if float(year) not in self.intercepts:
                raise ValueError(f"No saved intercept for year {year}")
        if years.shape != (len(features),):
            raise ValueError(
                "year must be a scalar or an aligned one-dimensional array"
            )
        missing = set(years.tolist()) - self.intercepts.keys()
        if missing:
            raise ValueError(f"No saved intercept for years {sorted(missing)}")
        intercepts = np.array([self.intercepts[y] for y in years])
        score = features @ self.coefficients + intercepts
        if not np.isfinite(score).all():
            raise ValueError("Input magnitude caused non-finite scores")
        probability = np.empty_like(score)
        positive = score >= 0
        probability[positive] = 1 / (1 + np.exp(-score[positive]))
        exp_score = np.exp(score[~positive])
        probability[~positive] = exp_score / (1 + exp_score)
        return probability

    def predict(self, inputs, *, year):
        """Return 2 for probabilities >= 0.5, otherwise 1."""
        return np.where(self.predict_proba_second_lien(inputs, year=year) >= 0.5, 2, 1)


def load_model(path):
    """Load a portable JSON model without importing the training package."""
    return PortableLogisticModel(json.loads(Path(path).read_text()))
