"""Validation shared by the density-ratio model families."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from ... import config
from ..protocols import ModelConfiguration


def validate_request(
    training: pd.DataFrame,
    configurations: Sequence[ModelConfiguration],
    train_years: tuple[int, ...],
    family_name: str,
) -> None:
    """Reject ambiguous family requests before any estimator is fitted."""
    if not configurations:
        raise ValueError("configurations cannot be empty")
    if not train_years:
        raise ValueError("train_years cannot be empty")
    if tuple(sorted(set(train_years))) != train_years:
        raise ValueError("train_years must be unique and sorted")
    observed_years = tuple(sorted(int(year) for year in pd.unique(training["year"])))
    if observed_years != train_years:
        raise ValueError(
            f"train_years {train_years} do not match training data {observed_years}"
        )
    unknown_labels = set(pd.unique(training[config.LABEL_VAR])) - {
        config.FIRST_LIEN_CLASS,
        config.SECOND_LIEN_CLASS,
    }
    if unknown_labels:
        raise ValueError(f"Unknown training labels: {sorted(unknown_labels)}")
    for configuration in configurations:
        if configuration.family != family_name:
            raise ValueError(
                f"Expected family {family_name!r}, got {configuration.family!r}"
            )


def require_parameters(
    configuration: ModelConfiguration, expected: set[str]
) -> dict[str, object]:
    """Return parameters after enforcing the family's exact public schema."""
    parameters = configuration.parameter_dict()
    names = set(parameters)
    if names != expected:
        raise ValueError(
            f"Configuration {configuration.specification!r} requires parameters "
            f"{sorted(expected)}, got {sorted(names)}"
        )
    return parameters
