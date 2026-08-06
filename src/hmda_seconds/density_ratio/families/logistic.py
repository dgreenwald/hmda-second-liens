"""Known-source-prior ridge-logistic density-ratio family."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ... import mixture, mixture_logistic_selection
from ...logistic_features import FeatureSpecification
from .. import adapters, artifacts
from ..protocols import FittedDensityRatioModel, ModelConfiguration
from ._validation import require_parameters, validate_request


@dataclass(frozen=True)
class LogisticFamily:
    """Fit ridge paths while reusing one transformed matrix per specification."""

    artifact_dir: Path
    family_name: str = "logistic"

    def __init__(self, artifact_dir: str | Path) -> None:
        object.__setattr__(self, "artifact_dir", Path(artifact_dir))
        object.__setattr__(self, "family_name", "logistic")

    def fit_many(
        self,
        training: pd.DataFrame,
        configurations: Sequence[ModelConfiguration],
        *,
        train_years: tuple[int, ...],
    ) -> Mapping[str, FittedDensityRatioModel]:
        """Fit and save all requested ridge candidates."""
        validate_request(training, configurations, train_years, self.family_name)
        grouped: dict[FeatureSpecification, list[tuple[ModelConfiguration, float]]] = (
            defaultdict(list)
        )
        seen = set()
        for configuration in configurations:
            if configuration.random_seed is not None:
                raise ValueError("Logistic fitting does not use a random seed")
            parameters = require_parameters(configuration, {"C"})
            regularization_c = float(parameters["C"])
            if regularization_c <= 0:
                raise ValueError("C must be positive")
            specification = _parse_specification(configuration.specification)
            key = (specification, regularization_c)
            if key in seen:
                raise ValueError(f"Duplicate logistic configuration: {key}")
            seen.add(key)
            grouped[specification].append((configuration, regularization_c))

        result: dict[str, FittedDensityRatioModel] = {}
        for specification, entries in grouped.items():
            fitted = {}
            missing = []
            for _, regularization_c in entries:
                path = mixture.known_source_prior_model_path(
                    train_years, specification, regularization_c, self.artifact_dir
                )
                if path.exists():
                    model = mixture.load_known_source_prior_model(path)
                    # Upgrade metadata-free legacy artifacts at the compatibility boundary.
                    if artifacts.load_metadata(path) is None:
                        mixture.save_known_source_prior_model(model, path)
                    fitted[regularization_c] = model
                else:
                    missing.append(regularization_c)
            if missing:
                new_fits, _ = mixture_logistic_selection.fit_candidate_path(
                    training, specification, missing
                )
                fitted.update(new_fits)
            for _, regularization_c in entries:
                model = fitted[regularization_c]
                path = mixture.known_source_prior_model_path(
                    train_years, specification, regularization_c, self.artifact_dir
                )
                if not path.exists() or artifacts.load_metadata(path) is None:
                    mixture.save_known_source_prior_model(model, path)
                adapted = adapters.adapt_known_source_prior_model(model)
                if adapted.model_id in result:
                    raise ValueError(f"Duplicate model_id: {adapted.model_id}")
                result[adapted.model_id] = adapted
        return result


def _parse_specification(name: str) -> FeatureSpecification:
    parts = name.split("__")
    if len(parts) not in (2, 3):
        raise ValueError(f"Invalid logistic specification {name!r}")
    geography = None if len(parts) == 2 else parts[2]
    specification = FeatureSpecification(parts[0], parts[1], geography)
    if specification.name != name:
        raise ValueError(f"Noncanonical logistic specification {name!r}")
    return specification
