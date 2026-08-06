"""Fixed Random Forest density-ratio family."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import parallel_backend
from scipy.special import logit
from sklearn.ensemble import RandomForestClassifier

from ... import config
from .. import adapters, artifacts
from ..protocols import FittedDensityRatioModel, ModelConfiguration
from ..weighting import equal_source_prior_weights
from ._validation import require_parameters, validate_request

SPECIFICATION = "raw_continuous_and_full_one_hot_categories"
PROBABILITY_EPSILON = 1e-12


@dataclass
class RandomForestDensityRatioModel:
    """Persisted equal-source-prior forest and its fixed feature schema."""

    classifier: RandomForestClassifier
    train_years: tuple[int, ...]
    feature_names: tuple[str, ...]
    n_training: int = 0
    n_first_lien: int = 0
    n_second_lien: int = 0

    def log_ratio(self, frame: pd.DataFrame) -> np.ndarray:
        features, names = forest_features(frame)
        if names != self.feature_names:
            raise RuntimeError("Random Forest feature columns changed")
        second_column = list(self.classifier.classes_).index(
            config.SECOND_LIEN_CLASS
        )
        with parallel_backend("threading"):
            probability = self.classifier.predict_proba(features)[:, second_column]
        probability = np.clip(
            probability, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON
        )
        return logit(probability)


def forest_features(frame: pd.DataFrame) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build raw continuous and full one-hot categorical RF features."""
    required = [*config.CONTINUOUS_VARS, *config.CATEGORY_VARS]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing Random Forest features: {sorted(missing)}")
    blocks = []
    names = []
    for variable in config.CONTINUOUS_VARS:
        values = frame[variable].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"{variable} contains non-finite values")
        blocks.append(values[:, None])
        names.append(variable)
    for variable in config.CATEGORY_VARS:
        values = frame[variable].to_numpy()
        levels = config.CATEGORY_LEVELS[variable]
        unknown = set(pd.unique(values)) - set(levels)
        if unknown:
            raise ValueError(f"Unknown {variable} levels: {sorted(unknown)}")
        blocks.append((values[:, None] == np.asarray(levels)).astype(np.float32))
        names.extend(f"{variable}_{level}" for level in levels)
    return np.column_stack(blocks), tuple(names)


def fit_forest_ratio_model(
    training: pd.DataFrame,
) -> tuple[RandomForestDensityRatioModel, dict]:
    """Fit the fixed RF with equal class-prior mass in every source year."""
    features, names = forest_features(training)
    labels = training[config.LABEL_VAR].to_numpy()
    is_second = labels == config.SECOND_LIEN_CLASS
    weights = equal_source_prior_weights(training, is_second)
    counts = artifacts.training_counts(
        training,
        label_var=config.LABEL_VAR,
        first_lien_class=config.FIRST_LIEN_CLASS,
        second_lien_class=config.SECOND_LIEN_CLASS,
    )
    classifier = RandomForestClassifier(**config.RF_KWARGS)
    start = time.perf_counter()
    with parallel_backend("threading"):
        classifier.fit(features, labels, sample_weight=weights)
    model = RandomForestDensityRatioModel(
        classifier=classifier,
        train_years=tuple(sorted(pd.unique(training["year"]))),
        feature_names=names,
        n_training=counts[0],
        n_first_lien=counts[1],
        n_second_lien=counts[2],
    )
    return model, {"fit_seconds": time.perf_counter() - start}


def save_forest_model(
    model: RandomForestDensityRatioModel, model_file: str | Path
) -> None:
    model_file = Path(model_file)
    adapted = adapters.adapt_random_forest_model(model)
    artifacts.save_fitted_model(
        model,
        model_file,
        model_id=adapted.model_id,
        configuration=ModelConfiguration.from_mapping(
            "random_forest",
            SPECIFICATION,
            {
                "max_depth": config.RF_KWARGS["max_depth"],
                "n_estimators": config.RF_KWARGS["n_estimators"],
            },
            random_seed=config.RF_KWARGS["random_state"],
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


def load_forest_model(model_file: str | Path) -> RandomForestDensityRatioModel:
    model, metadata = artifacts.load_pickle_artifact(
        model_file, RandomForestDensityRatioModel
    )
    artifacts.validate_metadata_identity(
        metadata,
        model_id=adapters.adapt_random_forest_model(model).model_id,
        train_years=model.train_years,
    )
    return model


def forest_model_path(
    train_years: Sequence[int],
    model_dir: str | Path = config.RF_MIXTURE_FOLD_MODEL_DIR,
) -> Path:
    years = tuple(train_years)
    if not years:
        raise ValueError("train_years cannot be empty")
    return Path(model_dir) / (
        f"rf_50_depth_10__train_{min(years)}_{max(years)}.pkl"
    )


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
        path = forest_model_path(train_years, self.artifact_dir)
        if path.exists():
            model = load_forest_model(path)
        else:
            model, _ = fit_forest_ratio_model(training)
        if not path.exists() or artifacts.load_metadata(path) is None:
            save_forest_model(model, path)
        adapted = adapters.adapt_random_forest_model(model)
        return {adapted.model_id: adapted}
