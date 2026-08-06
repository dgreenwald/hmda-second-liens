"""Fixed Random Forest density-ratio family."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ... import config, random_forest_mixture
from .. import adapters
from ..protocols import FittedDensityRatioModel, ModelConfiguration
from ._validation import require_parameters, validate_request

SPECIFICATION = "raw_continuous_and_full_one_hot_categories"


@dataclass(frozen=True)
class RandomForestFamily:
    """Fit and persist the single frozen Random Forest robustness model."""

    artifact_dir: Path
    family_name: str = "random_forest"

    def __init__(self, artifact_dir: str | Path) -> None:
        object.__setattr__(self, "artifact_dir", Path(artifact_dir))
        object.__setattr__(self, "family_name", "random_forest")

    def fit_many(
        self,
        training: pd.DataFrame,
        configurations: Sequence[ModelConfiguration],
        *,
        train_years: tuple[int, ...],
    ) -> Mapping[str, FittedDensityRatioModel]:
        """Fit and save the fixed forest, rejecting attempts to retune it."""
        validate_request(training, configurations, train_years, self.family_name)
        if len(configurations) != 1:
            raise ValueError("Random Forest family accepts exactly one configuration")
        configuration = configurations[0]
        if configuration.random_seed != config.RF_KWARGS["random_state"]:
            raise ValueError(
                "Random Forest random_seed does not match the frozen estimator seed"
            )
        if configuration.specification != SPECIFICATION:
            raise ValueError(
                f"Unknown Random Forest specification {configuration.specification!r}"
            )
        values = require_parameters(configuration, {"max_depth", "n_estimators"})
        expected = {
            "max_depth": config.RF_KWARGS["max_depth"],
            "n_estimators": config.RF_KWARGS["n_estimators"],
        }
        if values != expected:
            raise ValueError(f"Random Forest configuration is frozen at {expected}")
        model, _ = random_forest_mixture.fit_forest_ratio_model(training)
        path = random_forest_mixture.forest_model_path(
            train_years, self.artifact_dir
        )
        random_forest_mixture.save_forest_model(model, path)
        adapted = adapters.adapt_random_forest_model(model)
        return {adapted.model_id: adapted}
