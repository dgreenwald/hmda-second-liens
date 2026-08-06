"""Compatibility adapters for fitted models at their existing pickle paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..gradient_boosting import BoostingDensityRatioModel
    from ..mixture import KnownSourcePriorModel
    from ..random_forest_mixture import RandomForestDensityRatioModel


class _ExistingLogRatioModel(Protocol):
    train_years: tuple[int, ...]

    def log_ratio(self, frame: pd.DataFrame) -> np.ndarray:
        ...


@dataclass(frozen=True)
class ExistingFittedModelAdapter:
    """Add protocol metadata while delegating all numerical behavior."""

    model_id: str
    fitted: _ExistingLogRatioModel

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must be a nonempty string")

    @property
    def train_years(self) -> tuple[int, ...]:
        """Return the source years stored by the existing fitted object."""
        return self.fitted.train_years

    def log_ratio(self, frame: pd.DataFrame) -> np.ndarray:
        """Delegate without changing the existing numerical calculation."""
        return self.fitted.log_ratio(frame)


def adapt_known_source_prior_model(
    fitted: KnownSourcePriorModel, model_id: str
) -> ExistingFittedModelAdapter:
    """Adapt an existing logistic density-ratio fit."""
    from ..mixture import KnownSourcePriorModel

    _require_type(fitted, KnownSourcePriorModel)
    return ExistingFittedModelAdapter(model_id, fitted)


def adapt_boosting_model(
    fitted: BoostingDensityRatioModel, model_id: str
) -> ExistingFittedModelAdapter:
    """Adapt an existing histogram-gradient-boosting fit."""
    from ..gradient_boosting import BoostingDensityRatioModel

    _require_type(fitted, BoostingDensityRatioModel)
    return ExistingFittedModelAdapter(model_id, fitted)


def adapt_random_forest_model(
    fitted: RandomForestDensityRatioModel, model_id: str
) -> ExistingFittedModelAdapter:
    """Adapt an existing Random Forest density-ratio fit."""
    from ..random_forest_mixture import RandomForestDensityRatioModel

    _require_type(fitted, RandomForestDensityRatioModel)
    return ExistingFittedModelAdapter(model_id, fitted)


def _require_type(fitted: object, expected: type) -> None:
    if not isinstance(fitted, expected):
        raise TypeError(
            f"Expected {expected.__name__}, got {type(fitted).__name__}"
        )
