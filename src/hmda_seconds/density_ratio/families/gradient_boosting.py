"""Histogram-gradient-boosting density-ratio family."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ... import gradient_boosting
from .. import adapters
from ..protocols import FittedDensityRatioModel, ModelConfiguration
from ._validation import require_parameters, validate_request

SPECIFICATION = "primitive_continuous_and_native_categories"
PARAMETERS = {
    "max_leaf_nodes",
    "learning_rate",
    "max_iter",
    "l2_regularization",
    "min_samples_leaf",
}


@dataclass(frozen=True)
class GradientBoostingFamily:
    """Fit and persist independent frozen boosting configurations."""

    artifact_dir: Path
    family_name: str = "hist_gradient_boosting"

    def __init__(self, artifact_dir: str | Path) -> None:
        object.__setattr__(self, "artifact_dir", Path(artifact_dir))
        object.__setattr__(self, "family_name", "hist_gradient_boosting")

    def fit_many(
        self,
        training: pd.DataFrame,
        configurations: Sequence[ModelConfiguration],
        *,
        train_years: tuple[int, ...],
    ) -> Mapping[str, FittedDensityRatioModel]:
        """Fit and save every boosting candidate."""
        validate_request(training, configurations, train_years, self.family_name)
        result: dict[str, FittedDensityRatioModel] = {}
        for configuration in configurations:
            if configuration.random_seed != gradient_boosting.config.BOOSTING_RANDOM_STATE:
                raise ValueError(
                    "Boosting random_seed does not match the frozen estimator seed"
                )
            if configuration.specification != SPECIFICATION:
                raise ValueError(
                    f"Unknown boosting specification {configuration.specification!r}"
                )
            values = require_parameters(configuration, PARAMETERS)
            parameters = gradient_boosting.BoostingParameters(
                max_leaf_nodes=int(values["max_leaf_nodes"]),
                learning_rate=float(values["learning_rate"]),
                max_iter=int(values["max_iter"]),
                l2_regularization=float(values["l2_regularization"]),
                min_samples_leaf=int(values["min_samples_leaf"]),
            )
            model, _ = gradient_boosting.fit_boosting_ratio_model(
                training, parameters
            )
            path = gradient_boosting.boosting_model_path(
                train_years, parameters, self.artifact_dir
            )
            gradient_boosting.save_boosting_model(model, path)
            adapted = adapters.adapt_boosting_model(model)
            if adapted.model_id in result:
                raise ValueError(f"Duplicate model_id: {adapted.model_id}")
            result[adapted.model_id] = adapted
        return result
