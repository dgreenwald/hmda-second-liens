"""Histogram-gradient-boosting density-ratio family."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier

from ... import config
from .. import adapters, artifacts
from ..protocols import FittedDensityRatioModel, ModelConfiguration
from ..weighting import equal_source_prior_weights
from ._validation import require_parameters, validate_request

SPECIFICATION = "primitive_continuous_and_native_categories"
PARAMETERS = {
    "max_leaf_nodes",
    "learning_rate",
    "max_iter",
    "l2_regularization",
    "min_samples_leaf",
}
BOOSTING_FEATURES = [*config.CONTINUOUS_VARS, *config.CATEGORY_VARS]
CATEGORICAL_MASK = np.array([False, False, True, True])
PROBABILITY_EPSILON = 1e-12


@dataclass(frozen=True, order=True)
class BoostingParameters:
    """A reproducible, compact histogram-boosting specification."""

    max_leaf_nodes: int
    learning_rate: float
    max_iter: int = config.BOOSTING_BASE_MAX_ITER
    l2_regularization: float = config.BOOSTING_BASE_L2
    min_samples_leaf: int = config.BOOSTING_MIN_SAMPLES_LEAF

    @property
    def identifier(self) -> str:
        rate = _number_label(self.learning_rate)
        l2 = _number_label(self.l2_regularization)
        return (
            f"leaves_{self.max_leaf_nodes}__lr_{rate}__iter_{self.max_iter}"
            f"__l2_{l2}__minleaf_{self.min_samples_leaf}"
        )


@dataclass
class BoostingDensityRatioModel:
    """Persisted equal-source-prior boosting density-ratio model."""

    classifier: HistGradientBoostingClassifier
    parameters: BoostingParameters
    train_years: tuple[int, ...]
    feature_names: tuple[str, ...] = tuple(BOOSTING_FEATURES)
    n_training: int = 0
    n_first_lien: int = 0
    n_second_lien: int = 0

    def log_ratio(self, frame: pd.DataFrame) -> np.ndarray:
        features = boosting_features(frame)
        second_column = list(self.classifier.classes_).index(
            config.SECOND_LIEN_CLASS
        )
        probability = self.classifier.predict_proba(features)[:, second_column]
        probability = np.clip(
            probability, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON
        )
        return logit(probability)


def boosting_features(frame: pd.DataFrame) -> np.ndarray:
    """Construct the four primitive boosting features without engineering."""
    missing = set(BOOSTING_FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing boosting features: {sorted(missing)}")
    features = frame[BOOSTING_FEATURES].to_numpy(dtype=float)
    if not np.isfinite(features).all():
        raise ValueError("Boosting features contain non-finite values")
    for index, variable in enumerate(config.CATEGORY_VARS, start=2):
        unknown = set(np.unique(features[:, index])) - set(
            config.CATEGORY_LEVELS[variable]
        )
        if unknown:
            raise ValueError(f"Unknown {variable} levels: {sorted(unknown)}")
    return features


def fit_boosting_ratio_model(
    training: pd.DataFrame, parameters: BoostingParameters
) -> tuple[BoostingDensityRatioModel, dict]:
    """Fit an equal-source-prior histogram-boosting classifier."""
    features = boosting_features(training)
    labels = training[config.LABEL_VAR].to_numpy()
    is_second = labels == config.SECOND_LIEN_CLASS
    weights = equal_source_prior_weights(training, is_second)
    counts = artifacts.training_counts(
        training,
        label_var=config.LABEL_VAR,
        first_lien_class=config.FIRST_LIEN_CLASS,
        second_lien_class=config.SECOND_LIEN_CLASS,
    )
    classifier = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=parameters.learning_rate,
        max_iter=parameters.max_iter,
        max_leaf_nodes=parameters.max_leaf_nodes,
        min_samples_leaf=parameters.min_samples_leaf,
        l2_regularization=parameters.l2_regularization,
        categorical_features=CATEGORICAL_MASK,
        early_stopping=False,
        random_state=config.BOOSTING_RANDOM_STATE,
    )
    start = time.perf_counter()
    classifier.fit(features, labels, sample_weight=weights)
    fitted = BoostingDensityRatioModel(
        classifier=classifier,
        parameters=parameters,
        train_years=tuple(sorted(pd.unique(training["year"]))),
        n_training=counts[0],
        n_first_lien=counts[1],
        n_second_lien=counts[2],
    )
    return fitted, {
        "fit_seconds": time.perf_counter() - start,
        "n_iter_fitted": int(classifier.n_iter_),
    }


def save_boosting_model(
    model: BoostingDensityRatioModel, model_file: str | Path
) -> None:
    model_file = Path(model_file)
    adapted = adapters.adapt_boosting_model(model)
    artifacts.save_fitted_model(
        model,
        model_file,
        model_id=adapted.model_id,
        configuration=ModelConfiguration.from_mapping(
            "hist_gradient_boosting",
            SPECIFICATION,
            asdict(model.parameters),
            random_seed=config.BOOSTING_RANDOM_STATE,
        ),
        train_years=model.train_years,
        counts=(
            getattr(model, "n_training", 0),
            getattr(model, "n_first_lien", 0),
            getattr(model, "n_second_lien", 0),
        ),
        feature_names=model.feature_names,
        weighting="equal_class_mass_within_source_year",
        source_prior="one_half",
    )


def load_boosting_model(model_file: str | Path) -> BoostingDensityRatioModel:
    model, metadata = artifacts.load_pickle_artifact(
        model_file, BoostingDensityRatioModel
    )
    artifacts.validate_metadata_identity(
        metadata,
        model_id=adapters.adapt_boosting_model(model).model_id,
        train_years=model.train_years,
    )
    return model


def boosting_model_path(
    train_years: Sequence[int],
    parameters: BoostingParameters,
    model_dir: str | Path = config.BOOSTING_FOLD_MODEL_DIR,
) -> Path:
    years = tuple(train_years)
    if not years:
        raise ValueError("train_years cannot be empty")
    return Path(model_dir) / (
        f"{parameters.identifier}__train_{min(years)}_{max(years)}.pkl"
    )


def _number_label(value: float) -> str:
    return format(value, ".12g").replace(".", "p")


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
            if configuration.random_seed != config.BOOSTING_RANDOM_STATE:
                raise ValueError(
                    "Boosting random_seed does not match the frozen estimator seed"
                )
            if configuration.specification != SPECIFICATION:
                raise ValueError(
                    f"Unknown boosting specification {configuration.specification!r}"
                )
            values = require_parameters(configuration, PARAMETERS)
            parameters = BoostingParameters(
                max_leaf_nodes=int(values["max_leaf_nodes"]),
                learning_rate=float(values["learning_rate"]),
                max_iter=int(values["max_iter"]),
                l2_regularization=float(values["l2_regularization"]),
                min_samples_leaf=int(values["min_samples_leaf"]),
            )
            path = boosting_model_path(
                train_years, parameters, self.artifact_dir
            )
            if path.exists():
                model = load_boosting_model(path)
            else:
                model, _ = fit_boosting_ratio_model(training, parameters)
            if not path.exists() or artifacts.load_metadata(path) is None:
                save_boosting_model(model, path)
            adapted = adapters.adapt_boosting_model(model)
            if adapted.model_id in result:
                raise ValueError(f"Duplicate model_id: {adapted.model_id}")
            result[adapted.model_id] = adapted
        return result
