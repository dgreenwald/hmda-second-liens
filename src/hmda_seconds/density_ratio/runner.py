"""Family-neutral orchestration for one fold or a sequential local job set."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from .. import config
from . import artifacts, evaluation
from .protocols import (
    DensityRatioFamily,
    FittedDensityRatioModel,
    ModelConfiguration,
)
from .shards import (
    PlannedJob,
    ResultShard,
    ShardModel,
    read_shard,
    shard_path,
    write_shard,
)


def run_job(
    planned: PlannedJob,
    data_by_year: Mapping[int, pd.DataFrame],
    family: DensityRatioFamily,
    *,
    artifact_root: str | Path,
    output_path: str | Path | None = None,
) -> ResultShard:
    """Fit, persist, evaluate, and atomically publish one complete shard."""
    job = planned.job
    fold = planned.fold
    path = shard_path(job) if output_path is None else Path(output_path)
    if path.exists():
        existing = read_shard(path)
        if existing.job != job or existing.fold != fold:
            raise FileExistsError(f"Existing shard does not match planned job: {path}")
        return existing
    if family.family_name != job.family:
        raise ValueError(
            f"Family object {family.family_name!r} does not match job {job.family!r}"
        )
    required_years = (*fold.train_years, *fold.target_years)
    missing_years = sorted(set(required_years) - set(data_by_year))
    if missing_years:
        raise ValueError(f"Missing input years: {missing_years}")
    training = pd.concat(
        [data_by_year[year] for year in fold.train_years], ignore_index=True
    )
    fitted = family.fit_many(
        training, job.configurations, train_years=fold.train_years
    )
    model_records = _artifact_records(
        fitted, job.configurations, Path(artifact_root)
    )
    results = []
    for model_id in sorted(fitted):
        model = fitted[model_id]
        for target_year in fold.target_years:
            target = data_by_year[target_year]
            evaluated = evaluation.evaluate_target(
                model,
                target,
                fold,
                label_var=config.LABEL_VAR,
                second_lien_class=config.SECOND_LIEN_CLASS,
            )
            results.append(evaluated.result)
    shard = ResultShard(
        job=job,
        fold=fold,
        models=tuple(sorted(model_records, key=lambda item: item.model_id)),
        results=tuple(
            sorted(results, key=lambda item: (item.model_id, item.target_year))
        ),
    )
    return write_shard(shard, path)


def run_local(
    planned_jobs: Sequence[PlannedJob],
    data_by_year: Mapping[int, pd.DataFrame],
    families: Mapping[str, DensityRatioFamily],
    *,
    artifact_roots: Mapping[str, str | Path],
) -> list[Path]:
    """Run planned jobs sequentially through the same single-job worker."""
    paths = []
    for planned in planned_jobs:
        family_name = planned.job.family
        if family_name not in families or family_name not in artifact_roots:
            raise ValueError(f"No local family/artifact root for {family_name!r}")
        path = shard_path(planned.job)
        run_job(
            planned,
            data_by_year,
            families[family_name],
            artifact_root=artifact_roots[family_name],
            output_path=path,
        )
        paths.append(path)
    return paths


def _artifact_records(
    fitted: Mapping[str, FittedDensityRatioModel],
    configurations: tuple[ModelConfiguration, ...],
    artifact_root: Path,
) -> list[ShardModel]:
    expected_ids = set(fitted)
    found = {}
    suffix = artifacts.METADATA_SUFFIX
    for sidecar in artifact_root.rglob(f"*{suffix}"):
        payload = sidecar.with_name(sidecar.name[: -len(suffix)])
        metadata = artifacts.validate_existing_artifact(
            payload, allow_legacy=False
        )
        if metadata is None or metadata.model_id not in expected_ids:
            continue
        if metadata.model_id in found:
            raise ValueError(f"Multiple artifacts found for {metadata.model_id}")
        found[metadata.model_id] = ShardModel(
            model_id=metadata.model_id,
            configuration=metadata.configuration,
            artifact_path=str(payload),
        )
    missing = sorted(expected_ids - set(found))
    if missing:
        raise ValueError(f"Missing persisted artifacts for fitted models: {missing}")
    if {item.configuration for item in found.values()} != set(configurations):
        raise ValueError("Persisted artifact configurations do not match the job")
    return list(found.values())
