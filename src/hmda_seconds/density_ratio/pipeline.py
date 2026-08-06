"""Shared local grid orchestration used by compatibility pipeline commands."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from .protocols import (
    DensityRatioFamily,
    JobSpecification,
    ModelConfiguration,
    TemporalFold,
)
from .runner import run_local
from .shards import AggregatedResults, PlannedJob, aggregate_shards


def run_grid(
    data_by_year: Mapping[int, pd.DataFrame],
    folds: Sequence[TemporalFold],
    configurations: Mapping[str, Sequence[ModelConfiguration]],
    family: DensityRatioFamily,
    *,
    stage: str,
    artifact_root: str | Path,
    output_root: str | Path,
) -> AggregatedResults:
    """Run a family grid through immutable shards and aggregate it deterministically."""
    folds = [
        TemporalFold(
            fold_id=fold.fold_id,
            train_years=fold.train_years,
            target_years=fold.target_years,
            direction=fold.direction,
            horizons=fold.horizons,
            schema_version=fold.schema_version,
        )
        for fold in folds
    ]
    plans = [
        PlannedJob(
            JobSpecification(
                stage=stage,
                family=family.family_name,
                specification=specification,
                train_years=fold.train_years,
                configurations=tuple(candidates),
                input_paths=(),
                output_root=str(output_root),
            ),
            fold,
        )
        for fold in folds
        for specification, candidates in configurations.items()
    ]
    paths = run_local(
        plans,
        data_by_year,
        {family.family_name: family},
        artifact_roots={family.family_name: artifact_root},
    )
    return aggregate_shards(plans, paths)
