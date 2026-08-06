"""Immutable job-result shards and deterministic result aggregation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .protocols import (
    SCHEMA_VERSION,
    EvaluationResult,
    JobSpecification,
    ModelConfiguration,
    TemporalFold,
)


@dataclass(frozen=True)
class PlannedJob:
    """A job paired with its authoritative model-independent fold."""

    job: JobSpecification
    fold: TemporalFold

    def __post_init__(self) -> None:
        if self.job.train_years != self.fold.train_years:
            raise ValueError("job training years do not match its fold")


@dataclass(frozen=True)
class ShardModel:
    """Connect one fold-specific fit to its cross-fold configuration."""

    model_id: str
    configuration: ModelConfiguration
    artifact_path: str

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must be nonempty")
        if not self.artifact_path.strip():
            raise ValueError("artifact_path must be nonempty")

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "configuration": self.configuration.to_dict(),
            "artifact_path": self.artifact_path,
        }


@dataclass(frozen=True)
class ResultShard:
    """Complete, immutable aggregate output of exactly one planned job."""

    job: JobSpecification
    fold: TemporalFold
    models: tuple[ShardModel, ...]
    results: tuple[EvaluationResult, ...]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported shard schema {self.schema_version}")
        if self.job.train_years != self.fold.train_years:
            raise ValueError("shard job training years do not match its fold")
        model_ids = tuple(model.model_id for model in self.models)
        if not model_ids or len(set(model_ids)) != len(model_ids):
            raise ValueError("shard model IDs must be nonempty and unique")
        configurations = tuple(model.configuration for model in self.models)
        if len(set(configurations)) != len(configurations):
            raise ValueError("shard configurations must be unique")
        if set(configurations) != set(self.job.configurations):
            raise ValueError("shard models do not match planned configurations")
        result_keys = [
            (result.model_id, result.target_year) for result in self.results
        ]
        expected = {
            (model_id, year)
            for model_id in model_ids
            for year in self.fold.target_years
        }
        if set(result_keys) != expected or len(result_keys) != len(expected):
            raise ValueError("shard results are missing, duplicate, or unexpected")
        for result in self.results:
            if result.fold_id != self.fold.fold_id:
                raise ValueError("result fold ID does not match shard fold")
            if result.horizon != self.fold.horizon_for(result.target_year):
                raise ValueError("result horizon does not match shard fold")

    @property
    def logical_key(self) -> tuple[str, str, str, tuple[int, ...]]:
        """Return the job identity for duplicate/conflict detection."""
        return (
            self.job.stage,
            self.job.family,
            self.job.specification,
            self.job.train_years,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "job": self.job.to_dict(),
            "fold": self.fold.to_dict(),
            "models": [model.to_dict() for model in self.models],
            "results": [result.to_dict() for result in self.results],
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> ResultShard:
        """Restore a shard and enforce all completeness invariants."""
        job_values = _mapping(values, "job")
        fold_values = _mapping(values, "fold")
        model_values = _list(values, "models")
        result_values = _list(values, "results")
        fold = TemporalFold(
            fold_id=str(fold_values["fold_id"]),
            train_years=tuple(int(year) for year in fold_values["train_years"]),
            target_years=tuple(int(year) for year in fold_values["target_years"]),
            direction=str(fold_values["direction"]),
            horizons=tuple(int(value) for value in fold_values["horizons"]),
            schema_version=int(fold_values.get("schema_version", SCHEMA_VERSION)),
        )
        return cls(
            job=JobSpecification.from_dict(job_values),
            fold=fold,
            models=tuple(
                ShardModel(
                    model_id=str(item["model_id"]),
                    configuration=ModelConfiguration.from_dict(
                        _mapping(item, "configuration")
                    ),
                    artifact_path=str(item["artifact_path"]),
                )
                for item in model_values
            ),
            results=tuple(EvaluationResult(**item) for item in result_values),
            schema_version=int(values.get("schema_version", SCHEMA_VERSION)),
        )


@dataclass(frozen=True)
class AggregatedResults:
    """Deterministically ordered cell, horizon, and candidate tables."""

    cells: pd.DataFrame
    horizons: pd.DataFrame
    summary: pd.DataFrame


def shard_path(job: JobSpecification, shard_dir: str | Path | None = None) -> Path:
    """Return a deterministic path for one logical job and candidate set."""
    root = Path(job.output_root) / "shards" if shard_dir is None else Path(shard_dir)
    digest = hashlib.sha256(_canonical(job.to_dict())).hexdigest()[:12]
    label = "__".join(
        (
            _safe(job.stage),
            _safe(job.family),
            _safe(job.specification),
            f"train_{job.train_years[0]}_{job.train_years[-1]}",
            digest,
        )
    )
    return root / f"{label}.json"


def write_shard(shard: ResultShard, path: str | Path) -> ResultShard:
    """Publish a complete shard atomically, never replacing different content."""
    path = Path(path)
    values = _canonical(shard.to_dict()) + b"\n"
    if path.exists():
        existing = read_shard(path)
        if _canonical(existing.to_dict()) == _canonical(shard.to_dict()):
            return existing
        raise FileExistsError(f"Conflicting shard already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(values)
            file.flush()
            os.fsync(file.fileno())
        # Never overwrite a shard published by a competing worker.
        try:
            os.link(temporary, path)
        except FileExistsError:
            existing = read_shard(path)
            if _canonical(existing.to_dict()) != _canonical(shard.to_dict()):
                raise FileExistsError(f"Conflicting shard already exists: {path}")
            return existing
        return shard
    finally:
        temporary.unlink(missing_ok=True)


def read_shard(path: str | Path) -> ResultShard:
    """Read and fully validate one shard."""
    values = json.loads(Path(path).read_text())
    if not isinstance(values, dict):
        raise TypeError("shard must contain a JSON object")
    return ResultShard.from_dict(values)


def aggregate_shards(
    planned: Iterable[PlannedJob], shard_paths: Iterable[str | Path]
) -> AggregatedResults:
    """Validate the planned matrix and apply equal-horizon aggregation."""
    plans = list(planned)
    stages = {plan.job.stage for plan in plans}
    if len(stages) > 1:
        raise ValueError("cannot aggregate jobs from different stages")
    expected = {_job_key(plan.job): plan for plan in plans}
    if len(expected) != len(plans):
        raise ValueError("planned jobs contain duplicate logical keys")
    observed: dict[tuple[str, str, str, tuple[int, ...]], ResultShard] = {}
    for path in shard_paths:
        shard = read_shard(path)
        key = shard.logical_key
        if key in observed:
            raise ValueError(f"duplicate shard for logical job {key}")
        if key not in expected:
            raise ValueError(f"unexpected shard for logical job {key}")
        plan = expected[key]
        if shard.job != plan.job or shard.fold != plan.fold:
            raise ValueError(f"shard is incompatible with planned job {key}")
        observed[key] = shard
    missing = sorted(set(expected) - set(observed))
    if missing:
        raise ValueError(f"missing shards for planned jobs: {missing}")

    rows = []
    for key in sorted(observed):
        shard = observed[key]
        configuration_by_model = {
            model.model_id: model.configuration for model in shard.models
        }
        for result in shard.results:
            configuration = configuration_by_model[result.model_id]
            rows.append(
                {
                    **asdict(result),
                    "family": configuration.family,
                    "specification": configuration.specification,
                    "hyperparameters": json.dumps(
                        configuration.parameter_dict(), sort_keys=True
                    ),
                    "random_seed": configuration.random_seed,
                    "configuration_id": configuration_id(configuration),
                }
            )
    cells = pd.DataFrame(rows).sort_values(
        ["configuration_id", "horizon", "target_year", "fold_id"]
    ).reset_index(drop=True)
    group = [
        "configuration_id",
        "family",
        "specification",
        "hyperparameters",
        "random_seed",
    ]
    horizons = (
        cells.groupby([*group, "horizon"], dropna=False, as_index=False)
        .agg(
            mean_brier=("brier_score", "mean"),
            mean_log_loss=("log_loss", "mean"),
            mean_mixture_share=("mixture_share", "mean"),
            n_cells=("brier_score", "size"),
        )
        .sort_values(["configuration_id", "horizon"])
        .reset_index(drop=True)
    )
    summary = (
        horizons.groupby(group, dropna=False, as_index=False)
        .agg(
            selection_brier=("mean_brier", "mean"),
            mean_log_loss=("mean_log_loss", "mean"),
            n_horizons=("horizon", "size"),
            n_cells=("n_cells", "sum"),
        )
        .sort_values(["selection_brier", "configuration_id"])
        .reset_index(drop=True)
    )
    return AggregatedResults(cells, horizons, summary)


def configuration_id(configuration: ModelConfiguration) -> str:
    """Return a stable cross-fold candidate identifier."""
    digest = hashlib.sha256(_canonical(configuration.to_dict())).hexdigest()[:12]
    return f"{configuration.family}__{configuration.specification}__{digest}"


def _job_key(job: JobSpecification) -> tuple[str, str, str, tuple[int, ...]]:
    return (job.stage, job.family, job.specification, job.train_years)


def _canonical(values: object) -> bytes:
    return json.dumps(
        values, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _safe(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def _mapping(values: Mapping[str, object], name: str) -> Mapping[str, object]:
    result = values.get(name)
    if not isinstance(result, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return result


def _list(values: Mapping[str, object], name: str) -> list[Mapping[str, object]]:
    result = values.get(name)
    if not isinstance(result, list) or not all(isinstance(item, Mapping) for item in result):
        raise TypeError(f"{name} must be a list of mappings")
    return result
