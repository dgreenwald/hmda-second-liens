"""Fold-fitted feature construction for logistic model selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config

CONTINUOUS_FORMS = ("linear", "spline_lti", "spline_both")
CORE_INTERACTION_STRUCTURES = ("none", "loan_type", "purchaser_type", "both")
INTERACTION_STRUCTURES = (*CORE_INTERACTION_STRUCTURES, "purchaser_type_spline_lti")
GEOGRAPHY_OPTIONS = (None, "region", "state")
SPLINE_QUANTILES = (0.05, 0.35, 0.65, 0.95)

CENSUS_REGION_BY_STATE = {
    **dict.fromkeys((9, 23, 25, 33, 34, 36, 42, 44, 50), "northeast"),
    **dict.fromkeys((17, 18, 19, 20, 26, 27, 29, 31, 38, 39, 46, 55), "midwest"),
    **dict.fromkeys(
        (1, 5, 10, 11, 12, 13, 21, 22, 24, 28, 37, 40, 45, 47, 48, 51, 54),
        "south",
    ),
    **dict.fromkeys((2, 4, 6, 8, 15, 16, 30, 32, 35, 41, 49, 53, 56), "west"),
}
REGION_LEVELS = ("south", "northeast", "midwest", "west")
STATE_LEVELS = tuple(sorted(CENSUS_REGION_BY_STATE))


@dataclass(frozen=True)
class FeatureSpecification:
    """A predeclared continuous-form and indicator-interaction combination."""

    continuous_form: str
    interactions: str
    geography: str | None = None

    def __post_init__(self) -> None:
        if self.continuous_form not in CONTINUOUS_FORMS:
            raise ValueError(f"Unknown continuous form {self.continuous_form!r}")
        if self.interactions not in INTERACTION_STRUCTURES:
            raise ValueError(f"Unknown interaction structure {self.interactions!r}")
        if (
            self.interactions == "purchaser_type_spline_lti"
            and self.continuous_form == "linear"
        ):
            raise ValueError(
                "Purchaser LTI spline interactions require an LTI spline main effect"
            )
        if self.geography not in GEOGRAPHY_OPTIONS:
            raise ValueError(f"Unknown geography option {self.geography!r}")

    @property
    def name(self) -> str:
        parts = [self.continuous_form, self.interactions]
        if self.geography is not None:
            parts.append(self.geography)
        return "__".join(parts)


def core_specifications() -> list[FeatureSpecification]:
    """Return the frozen 3-by-4 non-geographic candidate grid."""
    return [
        FeatureSpecification(form, interactions)
        for form in CONTINUOUS_FORMS
        for interactions in CORE_INTERACTION_STRUCTURES
    ]


class LogisticFeatureTransformer:
    """Fit reference coding, scaling, knots, and feature names on one fold."""

    def __init__(
        self,
        specification: FeatureSpecification,
        spline_quantiles: tuple[float, ...] = SPLINE_QUANTILES,
    ) -> None:
        self.specification = specification
        self.spline_quantiles = spline_quantiles

    def fit(self, df: pd.DataFrame) -> LogisticFeatureTransformer:
        """Estimate all fold-specific transformations from training rows only."""
        self.raw_location_ = {}
        self.raw_scale_ = {}
        self.knots_ = {}
        self.basis_location_ = {}
        self.basis_scale_ = {}

        for variable in config.CONTINUOUS_VARS:
            values = _finite_values(df[variable], variable)
            self.raw_location_[variable] = float(values.mean())
            self.raw_scale_[variable] = _positive_scale(values.std(ddof=0))
            if self._uses_spline(variable):
                knots = np.quantile(values, self.spline_quantiles)
                if len(np.unique(knots)) != len(knots):
                    raise ValueError(f"Spline knots are not unique for {variable}")
                basis = restricted_cubic_basis(values, knots)
                self.knots_[variable] = knots
                self.basis_location_[variable] = basis.mean(axis=0)
                self.basis_scale_[variable] = np.array(
                    [_positive_scale(value) for value in basis.std(axis=0, ddof=0)]
                )

        # Construct once during fit to freeze names and validate category support.
        _, names = self._transform(df)
        self.feature_names_ = names
        self.n_features_in_ = len(names)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply the training-fold transformation to another frame."""
        if not hasattr(self, "feature_names_"):
            raise RuntimeError("Transformer must be fitted before transform")
        features, names = self._transform(df)
        if names != self.feature_names_:
            raise RuntimeError("Feature columns changed after fitting")
        return features

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        self.fit(df)
        return self.transform(df)

    def _transform(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        blocks = []
        names = []
        standardized_raw = {}
        continuous_bases = {}
        continuous_names = {}

        for variable in config.CONTINUOUS_VARS:
            values = _finite_values(df[variable], variable)
            standardized_raw[variable] = (
                values - self.raw_location_[variable]
            ) / self.raw_scale_[variable]
            if self._uses_spline(variable):
                basis = restricted_cubic_basis(values, self.knots_[variable])
                basis = (
                    basis - self.basis_location_[variable]
                ) / self.basis_scale_[variable]
                basis_names = [
                    f"{variable}_rcs_{index}"
                    for index in range(1, basis.shape[1] + 1)
                ]
                blocks.append(basis)
                names.extend(basis_names)
                continuous_bases[variable] = basis
                continuous_names[variable] = basis_names
            else:
                blocks.append(standardized_raw[variable][:, None])
                names.append(variable)
                continuous_bases[variable] = standardized_raw[variable][:, None]
                continuous_names[variable] = [variable]

        indicators = {}
        for variable in config.CATEGORY_VARS:
            matrix, indicator_names = _reference_indicators(
                df[variable], config.CATEGORY_LEVELS[variable], variable
            )
            indicators[variable] = (matrix, indicator_names)
            blocks.append(matrix)
            names.extend(indicator_names)

        for variable in self._interaction_variables():
            matrix, indicator_names = indicators[variable]
            for continuous in config.CONTINUOUS_VARS:
                if self._uses_spline_interaction(variable, continuous):
                    for basis, basis_name in zip(
                        continuous_bases[continuous].T,
                        continuous_names[continuous],
                        strict=True,
                    ):
                        blocks.append(basis[:, None] * matrix)
                        names.extend(
                            f"{basis_name}_x_{indicator}"
                            for indicator in indicator_names
                        )
                else:
                    blocks.append(standardized_raw[continuous][:, None] * matrix)
                    names.extend(
                        f"{continuous}_x_{indicator}"
                        for indicator in indicator_names
                    )

        geography, geography_names = self._geography_indicators(df)
        if geography is not None:
            blocks.append(geography)
            names.extend(geography_names)

        return np.column_stack(blocks), names

    def _uses_spline(self, variable: str) -> bool:
        if self.specification.continuous_form == "spline_both":
            return True
        return (
            self.specification.continuous_form == "spline_lti"
            and variable == "log_lti"
        )

    def _interaction_variables(self) -> tuple[str, ...]:
        if self.specification.interactions == "both":
            return "loan_type", "purchaser_type"
        if self.specification.interactions == "none":
            return ()
        if self.specification.interactions == "purchaser_type_spline_lti":
            return ("purchaser_type",)
        return (self.specification.interactions,)

    def _uses_spline_interaction(
        self, indicator_variable: str, continuous_variable: str
    ) -> bool:
        return (
            self.specification.interactions == "purchaser_type_spline_lti"
            and indicator_variable == "purchaser_type"
            and continuous_variable == "log_lti"
        )

    def _geography_indicators(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray | None, list[str]]:
        if self.specification.geography is None:
            return None, []
        if "state_code" not in df:
            raise ValueError("state_code is required for a geographic challenger")
        if self.specification.geography == "state":
            return _reference_indicators(df["state_code"], STATE_LEVELS, "state")

        region = pd.to_numeric(df["state_code"], errors="coerce").map(
            CENSUS_REGION_BY_STATE
        )
        if region.isna().any():
            raise ValueError("Census region is missing for at least one state_code")
        return _reference_indicators(region, REGION_LEVELS, "region")


def restricted_cubic_basis(values: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Return a natural/restricted cubic basis with linear tails."""
    values = np.asarray(values, dtype=float)
    knots = np.asarray(knots, dtype=float)
    if len(knots) < 3 or np.any(np.diff(knots) <= 0):
        raise ValueError("Restricted cubic knots must be strictly increasing")

    columns = [values]
    penultimate = knots[-2]
    last = knots[-1]
    denominator = last - penultimate
    scale = (last - knots[0]) ** 2
    for knot in knots[:-2]:
        term = np.maximum(values - knot, 0.0) ** 3
        term -= (
            (last - knot) / denominator
        ) * np.maximum(values - penultimate, 0.0) ** 3
        term += (
            (penultimate - knot) / denominator
        ) * np.maximum(values - last, 0.0) ** 3
        columns.append(term / scale)
    return np.column_stack(columns)


def _reference_indicators(
    values: pd.Series, levels, prefix: str
) -> tuple[np.ndarray, list[str]]:
    levels = tuple(levels)
    observed = set(pd.Series(values).dropna().unique())
    unknown = observed - set(levels)
    if unknown:
        raise ValueError(f"Unknown {prefix} levels: {sorted(unknown)}")
    matrix = np.column_stack(
        [np.asarray(values == level, dtype=float) for level in levels[1:]]
    )
    names = [f"{prefix}_{level}" for level in levels[1:]]
    return matrix, names


def _finite_values(values: pd.Series, variable: str) -> np.ndarray:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{variable} contains non-finite values")
    return array


def _positive_scale(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Cannot standardize a constant or non-finite feature")
    return float(value)
