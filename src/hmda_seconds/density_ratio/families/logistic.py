"""Known-source-prior ridge-logistic density-ratio family."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ... import config, model_selection
from ...logistic_features import FeatureSpecification, LogisticFeatureTransformer
from .. import artifacts, numerical
from ..protocols import FittedDensityRatioModel, ModelConfiguration
from ..weighting import equal_source_prior_weights
from ._validation import require_parameters, validate_request


@dataclass
class RatioVariant:
    """A normalized linear feature-density ratio."""

    name: str
    feature_coefficients: np.ndarray
    log_ratio_offset: float
    mean_ratio_first: float
    mean_inverse_ratio_second: float

    def log_ratio(self, features: np.ndarray) -> np.ndarray:
        return features @ self.feature_coefficients + self.log_ratio_offset


@dataclass
class KnownSourcePriorModel:
    """Frozen transform and equal-source-prior logistic density ratio."""

    transformer: LogisticFeatureTransformer
    ratio: RatioVariant
    fit_diagnostics: dict
    specification: FeatureSpecification
    regularization_c: float
    train_years: tuple[int, ...]
    n_training: int = 0
    n_first_lien: int = 0
    n_second_lien: int = 0

    @property
    def model_id(self) -> str:
        c_label = _number_label(self.regularization_c)
        return (
            f"logistic__{self.specification.name}__c_{c_label}"
            f"__train_{min(self.train_years)}_{max(self.train_years)}"
        )

    def log_ratio(self, frame: pd.DataFrame) -> np.ndarray:
        return self.ratio.log_ratio(self.transformer.transform(frame))


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
        mean_ratio_first=float(np.exp(numerical.log_mean_exp(score[~y_second]))),
        mean_inverse_ratio_second=float(
            np.exp(numerical.log_mean_exp(-score[y_second]))
        ),
    )


def fit_candidate_path(
    training: pd.DataFrame,
    specification: FeatureSpecification,
    c_values: Sequence[float],
) -> tuple[dict[float, KnownSourcePriorModel], dict[float, dict]]:
    """Fit and retain every equal-prior ridge value on one transformed fold."""
    transformer = LogisticFeatureTransformer(specification)
    features = transformer.fit_transform(training)
    labels = training[config.LABEL_VAR].to_numpy()
    y_second = labels == config.SECOND_LIEN_CLASS
    weights = equal_source_prior_weights(training, y_second)
    counts = artifacts.training_counts(
        training,
        label_var=config.LABEL_VAR,
        first_lien_class=config.FIRST_LIEN_CLASS,
        second_lien_class=config.SECOND_LIEN_CLASS,
    )
    classifiers, diagnostics = model_selection.fit_regularization_path(
        features, labels, c_values, sample_weight=weights
    )
    train_years = tuple(sorted(pd.unique(training["year"])))
    models = {
        regularization_c: KnownSourcePriorModel(
            transformer=transformer,
            ratio=classifier_ratio_variant(
                "known_source_prior", features, y_second, classifier
            ),
            fit_diagnostics=diagnostics[regularization_c],
            specification=specification,
            regularization_c=regularization_c,
            train_years=train_years,
            n_training=counts[0],
            n_first_lien=counts[1],
            n_second_lien=counts[2],
        )
        for regularization_c, classifier in classifiers.items()
    }
    return models, diagnostics


def fit_known_source_prior_model(
    training: pd.DataFrame,
    specification: FeatureSpecification,
    regularization_c: float,
    model_file: str | Path | None = None,
) -> KnownSourcePriorModel:
    """Fit one equal-source-prior logistic density-ratio model."""
    models, _ = fit_candidate_path(training, specification, [regularization_c])
    fitted = models[regularization_c]
    if model_file is not None:
        save_known_source_prior_model(fitted, model_file)
    return fitted


def save_known_source_prior_model(
    model: KnownSourcePriorModel, model_file: str | Path
) -> None:
    model_file = Path(model_file)
    artifacts.save_fitted_model(
        model,
        model_file,
        model_id=model.model_id,
        configuration=ModelConfiguration.from_mapping(
            "logistic", model.specification.name, {"C": model.regularization_c}
        ),
        train_years=model.train_years,
        counts=(
            getattr(model, "n_training", 0),
            getattr(model, "n_first_lien", 0),
            getattr(model, "n_second_lien", 0),
        ),
        feature_names=tuple(model.transformer.feature_names_),
        weighting="equal_class_mass_within_source_year",
        source_prior="one_half",
    )


def load_known_source_prior_model(model_file: str | Path) -> KnownSourcePriorModel:
    model, metadata = artifacts.load_pickle_artifact(model_file, KnownSourcePriorModel)
    artifacts.validate_metadata_identity(
        metadata,
        model_id=model.model_id,
        train_years=model.train_years,
    )
    return model


def known_source_prior_model_path(
    train_years: Sequence[int],
    specification: FeatureSpecification,
    regularization_c: float,
    model_dir: str | Path = config.MIXTURE_FOLD_MODEL_DIR,
) -> Path:
    years = tuple(train_years)
    if not years:
        raise ValueError("train_years cannot be empty")
    c_label = _number_label(regularization_c)
    return Path(model_dir) / (
        f"known_source_prior__{specification.name}__c_{c_label}"
        f"__train_{min(years)}_{max(years)}.pkl"
    )


def _number_label(value: float) -> str:
    return format(value, ".12g").replace(".", "p")


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
                path = known_source_prior_model_path(
                    train_years, specification, regularization_c, self.artifact_dir
                )
                if path.exists():
                    model = load_known_source_prior_model(path)
                    fitted[regularization_c] = model
                else:
                    missing.append(regularization_c)
            if missing:
                new_fits, _ = fit_candidate_path(training, specification, missing)
                fitted.update(new_fits)
            for _, regularization_c in entries:
                model = fitted[regularization_c]
                path = known_source_prior_model_path(
                    train_years, specification, regularization_c, self.artifact_dir
                )
                if not path.exists():
                    save_known_source_prior_model(model, path)
                if model.model_id in result:
                    raise ValueError(f"Duplicate model_id: {model.model_id}")
                result[model.model_id] = model
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
