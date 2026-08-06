"""Cluster manifest, family construction, and Slurm-array generation helpers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .. import model_selection
from .families import GradientBoostingFamily, LogisticFamily, RandomForestFamily
from .folds import reverse_folds
from .protocols import JobSpecification, ModelConfiguration
from .runner import run_job
from .shards import (
    PlannedJob,
    aggregate_shards,
    read_manifest,
    shard_path,
    write_manifest,
)

COARSE_C_VALUES = (1e-4, 1e-2, 1.0, 100.0)
PILOT_SPECIFICATIONS = (
    "linear__none",
    "spline_lti__purchaser_type_spline_lti",
)
FIRST_ORDER_STAGE = "mixture_logistic_first_order"
INCUMBENT_SPECIFICATION = "spline_lti__purchaser_type"
INCUMBENT_C_VALUES = (0.01, 0.1, 1.0)
FIRST_ORDER_FEATURE_SPECIFICATIONS = (
    "linear__purchaser_type",
    "spline_both__purchaser_type",
    "spline_lti__none",
    "spline_lti__loan_type",
    "spline_lti__both",
    "spline_lti__purchaser_type_spline_lti",
)


def reverse_fold_for_start(train_start: int):
    """Return the authoritative reverse fold identified by its first source year."""
    matches = [fold for fold in reverse_folds() if fold.train_start == train_start]
    if len(matches) != 1:
        raise ValueError(f"No unique reverse fold starts in {train_start}")
    return matches[0]


def make_job(
    *,
    stage: str,
    family: str,
    specification: str,
    train_start: int,
    configurations: tuple[ModelConfiguration, ...],
    data_dir: str | Path,
    output_root: str | Path,
) -> PlannedJob:
    """Build one cluster-neutral planned reverse-validation job."""
    fold = reverse_fold_for_start(train_start)
    job = JobSpecification(
        stage=stage,
        family=family,
        specification=specification,
        train_years=fold.train_years,
        configurations=configurations,
        input_paths=(("selection_data_dir", str(data_dir)),),
        output_root=str(output_root),
    )
    return PlannedJob(job, fold)


def pilot_jobs(
    *, data_dir: str | Path, output_root: str | Path, train_start: int = 2013
) -> list[PlannedJob]:
    """Return the simple and spline-heavy resource-pilot jobs without submitting."""
    return [
        make_job(
            stage="pilot",
            family="logistic",
            specification=specification,
            train_start=train_start,
            configurations=tuple(
                ModelConfiguration.from_mapping(
                    "logistic", specification, {"C": regularization_c}
                )
                for regularization_c in COARSE_C_VALUES
            ),
            data_dir=data_dir,
            output_root=output_root,
        )
        for specification in PILOT_SPECIFICATIONS
    ]


def first_order_logistic_jobs(
    *, data_dir: str | Path, output_root: str | Path
) -> list[PlannedJob]:
    """Return the frozen one-coordinate neighborhood over all reverse folds."""
    specifications = (INCUMBENT_SPECIFICATION, *FIRST_ORDER_FEATURE_SPECIFICATIONS)
    jobs = []
    for specification in specifications:
        regularization_values = (
            INCUMBENT_C_VALUES
            if specification == INCUMBENT_SPECIFICATION
            else (0.1,)
        )
        configurations = tuple(
            ModelConfiguration.from_mapping(
                "logistic", specification, {"C": regularization_c}
            )
            for regularization_c in regularization_values
        )
        for fold in reverse_folds():
            jobs.append(
                make_job(
                    stage=FIRST_ORDER_STAGE,
                    family="logistic",
                    specification=specification,
                    train_start=fold.train_start,
                    configurations=configurations,
                    data_dir=data_dir,
                    output_root=output_root,
                )
            )
    return jobs


def family_for(job: JobSpecification, artifact_root: str | Path):
    """Construct the requested Step 6 family with a common artifact root."""
    families = {
        "logistic": LogisticFamily,
        "hist_gradient_boosting": GradientBoostingFamily,
        "random_forest": RandomForestFamily,
    }
    try:
        family_type = families[job.family]
    except KeyError as error:
        raise ValueError(f"Unknown density-ratio family {job.family!r}") from error
    return family_type(artifact_root)


def expand_job_paths(planned: PlannedJob) -> PlannedJob:
    """Expand cluster environment variables in manifest-owned paths."""
    job = planned.job
    expanded = JobSpecification(
        stage=job.stage,
        family=job.family,
        specification=job.specification,
        train_years=job.train_years,
        configurations=job.configurations,
        input_paths=tuple(
            (name, os.path.expandvars(os.path.expanduser(path)))
            for name, path in job.input_paths
        ),
        output_root=os.path.expandvars(os.path.expanduser(job.output_root)),
        schema_version=job.schema_version,
    )
    return PlannedJob(expanded, planned.fold)


def configurations_from_json(
    family: str, specification: str, values: list[str]
) -> tuple[ModelConfiguration, ...]:
    """Parse repeated JSON hyperparameter objects from the worker CLI."""
    if not values:
        raise ValueError("at least one --configuration is required")
    configurations = []
    for value in values:
        parameters = json.loads(value)
        if not isinstance(parameters, dict):
            raise TypeError("each configuration must be a JSON object")
        random_seed = parameters.pop("random_seed", None)
        configurations.append(
            ModelConfiguration.from_mapping(
                family,
                specification,
                parameters,
                random_seed=random_seed,
            )
        )
    return tuple(configurations)


def write_slurm_array(
    planned: list[PlannedJob],
    *,
    destination: str | Path,
    repo_dir: str,
    activate: str,
    time_limit: str = "8:00:00",
    memory: str = "32G",
    job_name: str = "hmda-density-ratio-pilot",
    max_concurrent: int | None = None,
) -> tuple[Path, Path]:
    """Write a manifest and one Slurm array script; never submit either."""
    if not planned:
        raise ValueError("planned jobs cannot be empty")
    if max_concurrent is not None and max_concurrent <= 0:
        raise ValueError("max_concurrent must be positive")
    destination = Path(destination)
    if destination.is_absolute():
        raise ValueError("destination must be relative to the repository root")
    destination.mkdir(parents=True, exist_ok=True)
    manifest = write_manifest(planned, destination / "density_ratio_jobs.json")
    script = destination / "density_ratio_jobs.slurm"
    manifest_on_cluster = f"{repo_dir}/{manifest.as_posix()}"
    log_dir = f"{repo_dir}/{destination.as_posix()}"
    array = f"0-{len(planned) - 1}"
    if max_concurrent is not None:
        array += f"%{max_concurrent}"
    contents = f"""#!/bin/bash
#SBATCH --time={time_limit}
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/%x_%A_%a.out
#SBATCH --error={log_dir}/%x_%A_%a.err
#SBATCH --mem={memory}
#SBATCH --array={array}

set -euo pipefail
source {_expandable_quote(activate)}
cd {_expandable_quote(repo_dir)}
/usr/bin/time -v python scripts/run_density_ratio_job.py \\
    --manifest {_expandable_quote(manifest_on_cluster)} \\
    --job-index "${{SLURM_ARRAY_TASK_ID}}"
"""
    script.write_text(contents)
    return manifest, script


def execute_planned_job(planned: PlannedJob):
    """Load only required years and execute one planned worker job."""
    inputs = dict(planned.job.input_paths)
    data_dir = Path(inputs["selection_data_dir"])
    years = (*planned.fold.train_years, *planned.fold.target_years)
    data_by_year = model_selection.load_selection_years(data_dir, years)
    artifact_root = (
        Path(planned.job.output_root)
        / "models"
        / planned.job.family
        / planned.job.specification
    )
    family = family_for(planned.job, artifact_root)
    return run_job(
        planned,
        data_by_year,
        family,
        artifact_root=artifact_root,
    )


def aggregate_manifest(manifest: str | Path, output_dir: str | Path) -> list[Path]:
    """Validate a complete manifest and atomically write its three tables."""
    planned = [expand_job_paths(item) for item in read_manifest(manifest)]
    aggregated = aggregate_shards(
        planned, [shard_path(item.job) for item in planned]
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destinations = []
    for name in ("cells", "horizons", "summary"):
        destination = output_dir / f"density_ratio_{name}.csv"
        _atomic_csv(getattr(aggregated, name), destination)
        destinations.append(destination)
    return destinations


def _expandable_quote(value: str) -> str:
    """Quote a trusted generated path while retaining shell variable expansion."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`")
    return f'"{escaped}"'


def _atomic_csv(frame, destination: Path) -> None:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
