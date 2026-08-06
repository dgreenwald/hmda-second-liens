"""Structural interfaces and immutable records for density-ratio models."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Literal, Protocol, TypeAlias, runtime_checkable

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
Direction: TypeAlias = Literal["reverse", "forward"]
ParameterValue: TypeAlias = str | int | float | bool | None


@runtime_checkable
class FittedDensityRatioModel(Protocol):
    """Minimal interface consumed by density-ratio evaluation."""

    @property
    def model_id(self) -> str:
        """Return the stable identity of this particular fitted model."""

    @property
    def train_years(self) -> tuple[int, ...]:
        """Return the labeled source years used for fitting."""

    def log_ratio(self, frame: pd.DataFrame) -> np.ndarray:
        """Return ``log(f_1(x) / f_0(x))`` in frame row order."""


@runtime_checkable
class DensityRatioFamily(Protocol):
    """Interface through which orchestration requests one family of fits."""

    @property
    def family_name(self) -> str:
        """Return the stable model-family name."""

    def fit_many(
        self,
        training: pd.DataFrame,
        configurations: Sequence[ModelConfiguration],
        *,
        train_years: tuple[int, ...],
    ) -> Mapping[str, FittedDensityRatioModel]:
        """Fit configurations, reusing family-specific work where possible."""


@dataclass(frozen=True)
class ModelConfiguration:
    """Serializable identity of one estimator configuration."""

    family: str
    specification: str
    hyperparameters: tuple[tuple[str, ParameterValue], ...] = ()
    random_seed: int | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_name(self.family, "family")
        _require_name(self.specification, "specification")
        _require_schema(self.schema_version)
        keys = [key for key, _ in self.hyperparameters]
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise ValueError("hyperparameters must have unique, sorted names")
        for key, value in self.hyperparameters:
            _require_name(key, "hyperparameter name")
            _require_parameter_value(value)

    @classmethod
    def from_mapping(
        cls,
        family: str,
        specification: str,
        hyperparameters: Mapping[str, ParameterValue],
        *,
        random_seed: int | None = None,
    ) -> ModelConfiguration:
        """Construct a configuration with deterministic parameter ordering."""
        return cls(
            family=family,
            specification=specification,
            hyperparameters=tuple(sorted(hyperparameters.items())),
            random_seed=random_seed,
        )

    def parameter_dict(self) -> dict[str, ParameterValue]:
        """Return a mutable copy of the estimator hyperparameters."""
        return dict(self.hyperparameters)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "family": self.family,
            "specification": self.specification,
            "hyperparameters": self.parameter_dict(),
            "random_seed": self.random_seed,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True)
class TemporalFold:
    """A model-independent temporal source/target fold."""

    fold_id: str
    train_years: tuple[int, ...]
    target_years: tuple[int, ...]
    direction: Direction
    horizons: tuple[int, ...]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_name(self.fold_id, "fold_id")
        _require_years(self.train_years, "train_years")
        _require_years(self.target_years, "target_years")
        _require_schema(self.schema_version)
        if self.direction not in ("reverse", "forward"):
            raise ValueError("direction must be 'reverse' or 'forward'")
        if set(self.train_years) & set(self.target_years):
            raise ValueError("training and target years must not overlap")
        if len(self.horizons) != len(self.target_years):
            raise ValueError("horizons must align one-to-one with target_years")
        if any(horizon <= 0 for horizon in self.horizons):
            raise ValueError("horizons must be positive")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        values = asdict(self)
        values["train_years"] = list(self.train_years)
        values["target_years"] = list(self.target_years)
        values["horizons"] = list(self.horizons)
        return values


@dataclass(frozen=True)
class ModelArtifactMetadata:
    """Metadata required to interpret and reproduce one saved fit."""

    model_id: str
    configuration: ModelConfiguration
    train_years: tuple[int, ...]
    n_training: int
    n_first_lien: int
    n_second_lien: int
    feature_names: tuple[str, ...]
    weighting: str
    source_prior: str
    artifact_path: str
    software_versions: tuple[tuple[str, str], ...] = ()
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_name(self.model_id, "model_id")
        _require_years(self.train_years, "train_years")
        _require_schema(self.schema_version)
        if min(self.n_training, self.n_first_lien, self.n_second_lien) < 0:
            raise ValueError("training counts must be nonnegative")
        if self.n_first_lien + self.n_second_lien != self.n_training:
            raise ValueError("lien-class counts must sum to n_training")
        if not self.feature_names or len(set(self.feature_names)) != len(
            self.feature_names
        ):
            raise ValueError("feature_names must be nonempty and unique")
        _require_name(self.weighting, "weighting")
        _require_name(self.source_prior, "source_prior")
        _require_name(self.artifact_path, "artifact_path")
        version_names = [name for name, _ in self.software_versions]
        if version_names != sorted(version_names) or len(version_names) != len(
            set(version_names)
        ):
            raise ValueError("software_versions must have unique, sorted names")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            **asdict(self),
            "configuration": self.configuration.to_dict(),
            "train_years": list(self.train_years),
            "feature_names": list(self.feature_names),
            "software_versions": dict(self.software_versions),
        }


@dataclass(frozen=True)
class EvaluationResult:
    """Common aggregate output for one fitted-model/target-year cell."""

    model_id: str
    fold_id: str
    target_year: int
    horizon: int
    n_observations: int
    actual_second_share: float
    mixture_share: float
    mean_probability: float
    hard_share_050: float
    brier_score: float
    log_loss: float
    optimizer_converged: bool
    mixture_at_boundary: bool
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_name(self.model_id, "model_id")
        _require_name(self.fold_id, "fold_id")
        _require_schema(self.schema_version)
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.n_observations <= 0:
            raise ValueError("n_observations must be positive")
        for name in (
            "actual_second_share",
            "mixture_share",
            "mean_probability",
            "hard_share_050",
            "brier_score",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and lie in [0, 1]")
        if not math.isfinite(self.log_loss) or self.log_loss < 0:
            raise ValueError("log_loss must be finite and nonnegative")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return asdict(self)


@dataclass(frozen=True)
class JobSpecification:
    """Serializable unit of local or distributed model work."""

    stage: str
    family: str
    specification: str
    train_years: tuple[int, ...]
    configurations: tuple[ModelConfiguration, ...]
    input_paths: tuple[tuple[str, str], ...]
    output_root: str
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_name(self.stage, "stage")
        _require_name(self.family, "family")
        _require_name(self.specification, "specification")
        _require_years(self.train_years, "train_years")
        _require_name(self.output_root, "output_root")
        _require_schema(self.schema_version)
        if not self.configurations:
            raise ValueError("configurations cannot be empty")
        if any(
            candidate.family != self.family
            or candidate.specification != self.specification
            for candidate in self.configurations
        ):
            raise ValueError("job configurations must match family/specification")
        path_names = [name for name, _ in self.input_paths]
        if path_names != sorted(path_names) or len(path_names) != len(
            set(path_names)
        ):
            raise ValueError("input_paths must have unique, sorted names")
        for name, path in self.input_paths:
            _require_name(name, "input path name")
            _require_name(path, "input path")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "stage": self.stage,
            "family": self.family,
            "specification": self.specification,
            "train_years": list(self.train_years),
            "configurations": [item.to_dict() for item in self.configurations],
            "input_paths": dict(self.input_paths),
            "output_root": self.output_root,
            "schema_version": self.schema_version,
        }


def _require_name(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")


def _require_years(years: tuple[int, ...], field: str) -> None:
    if not years:
        raise ValueError(f"{field} cannot be empty")
    if tuple(sorted(set(years))) != years:
        raise ValueError(f"{field} must be unique and increasing")


def _require_schema(schema_version: int) -> None:
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema version {schema_version}")


def _require_parameter_value(value: ParameterValue) -> None:
    if not isinstance(value, (str, int, float, bool, type(None))):
        raise TypeError("hyperparameter values must be JSON scalar values")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("floating-point hyperparameters must be finite")
