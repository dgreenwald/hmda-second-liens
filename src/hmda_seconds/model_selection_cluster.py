"""Immutable cluster orchestration for raw-logistic model selection."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from . import config, model_selection
from .density_ratio import artifacts
from .density_ratio.folds import reverse_folds
from .logistic_features import FeatureSpecification, core_specifications

SCHEMA_VERSION = 1
COARSE_STAGE = "coarse"
REFINEMENT_STAGE = "refinement"
STAGES = (COARSE_STAGE, REFINEMENT_STAGE)


@dataclass(frozen=True)
class RawLogisticJob:
    """One independently executable specification/fold ridge path."""

    stage: str
    specification: str
    train_start: int
    c_values: tuple[float, ...]
    data_dir: str
    output_root: str
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(f"Unknown raw-logistic stage {self.stage!r}")
        if self.specification not in {item.name for item in core_specifications()}:
            raise ValueError(f"Unknown core specification {self.specification!r}")
        fold_for_start(self.train_start)
        if not self.c_values or any(value <= 0 for value in self.c_values):
            raise ValueError("c_values must contain positive values")
        if tuple(sorted(set(self.c_values))) != self.c_values:
            raise ValueError("c_values must be unique and sorted")
        if not self.data_dir or not self.output_root:
            raise ValueError("data_dir and output_root must be nonempty")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema version {self.schema_version}")

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["c_values"] = list(self.c_values)
        return values

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> RawLogisticJob:
        return cls(
            stage=str(values["stage"]),
            specification=str(values["specification"]),
            train_start=int(values["train_start"]),
            c_values=tuple(float(value) for value in values["c_values"]),
            data_dir=str(values["data_dir"]),
            output_root=str(values["output_root"]),
            schema_version=int(values.get("schema_version", SCHEMA_VERSION)),
        )


def fold_for_start(train_start: int):
    """Return the authoritative reverse fold identified by its first year."""
    matches = [fold for fold in reverse_folds() if fold.train_start == train_start]
    if len(matches) != 1:
        raise ValueError(f"No unique reverse fold starts in {train_start}")
    return matches[0]


def coarse_jobs(*, data_dir: str | Path, output_root: str | Path) -> list[RawLogisticJob]:
    """Return the frozen 12-specification, nine-fold coarse search."""
    return _jobs(
        COARSE_STAGE,
        {
            specification.name: tuple(float(value) for value in config.LOGISTIC_SELECTION_COARSE_C)
            for specification in core_specifications()
        },
        data_dir,
        output_root,
    )


def refinement_jobs(
    coarse_summary: pd.DataFrame,
    *,
    data_dir: str | Path,
    output_root: str | Path,
) -> list[RawLogisticJob]:
    """Build the frozen adjacent-decade search from a complete coarse summary."""
    _validate_coarse_summary(coarse_summary)
    specifications = core_specifications()
    values = model_selection.refinement_grid(coarse_summary, specifications)
    return _jobs(
        REFINEMENT_STAGE,
        {item.name: tuple(float(value) for value in values[item]) for item in specifications},
        data_dir,
        output_root,
    )


def _jobs(stage, values_by_specification, data_dir, output_root):
    return [
        RawLogisticJob(
            stage=stage,
            specification=specification.name,
            train_start=fold.train_start,
            c_values=values_by_specification[specification.name],
            data_dir=str(data_dir),
            output_root=str(output_root),
        )
        for specification in core_specifications()
        for fold in reverse_folds()
    ]


def write_manifest(jobs: list[RawLogisticJob], path: str | Path) -> Path:
    """Atomically write an ordered array manifest."""
    if not jobs:
        raise ValueError("manifest cannot be empty")
    path = Path(path)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "jobs": [job.to_dict() for job in jobs],
    }
    _atomic_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
    )
    return path


def read_manifest(path: str | Path) -> list[RawLogisticJob]:
    """Read and validate a raw-logistic array manifest."""
    values = json.loads(Path(path).read_text())
    if not isinstance(values, dict) or values.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported raw-logistic manifest")
    entries = values.get("jobs")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Raw-logistic manifest must contain jobs")
    jobs = [RawLogisticJob.from_dict(entry) for entry in entries]
    keys = [(job.stage, job.specification, job.train_start) for job in jobs]
    if len(keys) != len(set(keys)):
        raise ValueError("Raw-logistic manifest contains duplicate jobs")
    if len({job.stage for job in jobs}) != 1:
        raise ValueError("Raw-logistic manifest mixes stages")
    expected_keys = {
        (specification.name, fold.train_start)
        for specification in core_specifications()
        for fold in reverse_folds()
    }
    if {(job.specification, job.train_start) for job in jobs} != expected_keys:
        raise ValueError("Raw-logistic manifest does not contain the complete 108-job grid")
    expected_path_length = 4 if jobs[0].stage == COARSE_STAGE else 2
    if any(len(job.c_values) != expected_path_length for job in jobs):
        raise ValueError("Raw-logistic manifest has an invalid ridge-path length")
    return jobs


def expand_job_paths(job: RawLogisticJob) -> RawLogisticJob:
    """Expand environment variables only on the execution host."""
    return RawLogisticJob(
        stage=job.stage,
        specification=job.specification,
        train_start=job.train_start,
        c_values=job.c_values,
        data_dir=os.path.expandvars(os.path.expanduser(job.data_dir)),
        output_root=os.path.expandvars(os.path.expanduser(job.output_root)),
    )


def specification_from_name(name: str) -> FeatureSpecification:
    """Restore one frozen core feature specification by name."""
    matches = [item for item in core_specifications() if item.name == name]
    if len(matches) != 1:
        raise ValueError(f"Unknown core specification {name!r}")
    return matches[0]


def shard_path(job: RawLogisticJob) -> Path:
    """Return the content-addressed result path for one planned job."""
    digest = hashlib.sha256(
        json.dumps(job.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    return Path(job.output_root) / "shards" / (
        f"raw_logistic__{job.stage}__{job.specification}"
        f"__train_{job.train_start}_{job.train_start + 3}__{digest}.json"
    )


def execute_job(job: RawLogisticJob) -> Path:
    """Fit, persist, evaluate, and atomically publish one planned job."""
    job = expand_job_paths(job)
    path = shard_path(job)
    if path.exists():
        shard = read_shard(path)
        _validate_shard(job, shard, validate_artifacts=True)
        return path

    fold = fold_for_start(job.train_start)
    years = (*fold.train_years, *fold.validation_years)
    data = model_selection.load_selection_years(job.data_dir, years)
    specification = specification_from_name(job.specification)
    model_dir = Path(job.output_root) / "models"
    cells = model_selection.evaluate_candidate_grid(
        data,
        {specification: job.c_values},
        folds=[fold],
        model_dir=model_dir,
        n_jobs=1,
    )
    artifact_paths = [
        str(model_selection.selected_model_path(fold.train_years, specification, value, model_dir))
        for value in job.c_values
    ]
    shard = {
        "schema_version": SCHEMA_VERSION,
        "job": job.to_dict(),
        "artifact_paths": artifact_paths,
        "cells": cells.to_dict(orient="records"),
    }
    _validate_shard(job, shard, validate_artifacts=True)
    _atomic_bytes(
        path,
        (json.dumps(shard, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
        replace=False,
    )
    return path


def read_shard(path: str | Path) -> dict[str, object]:
    values = json.loads(Path(path).read_text())
    if not isinstance(values, dict):
        raise TypeError("Raw-logistic shard must be a JSON object")
    return values


def aggregate_manifest(
    manifest: str | Path,
    output_dir: str | Path,
    *,
    coarse_cells: str | Path | None = None,
    model_output: str | Path = config.SELECTED_LOGISTIC_MODEL_FILE,
) -> list[Path]:
    """Validate all shards and write stage or combined selection tables."""
    jobs = [expand_job_paths(job) for job in read_manifest(manifest)]
    parts = []
    for job in jobs:
        path = shard_path(job)
        if not path.exists():
            raise ValueError(f"Missing shard for {job.specification}, train {job.train_start}")
        shard = read_shard(path)
        _validate_shard(job, shard, validate_artifacts=True)
        parts.append(pd.DataFrame(shard["cells"]))
    stage_cells = pd.concat(parts, ignore_index=True)
    stage_cells = _sort_cells(stage_cells)
    _validate_aggregated_cells(stage_cells, jobs)
    horizons, summary = model_selection.aggregate_brier_cells(stage_cells)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage = jobs[0].stage
    prefix = output_dir / f"logistic_selection_core_{stage}"
    destinations = _write_tables(prefix, stage_cells, horizons, summary)

    if stage == REFINEMENT_STAGE:
        if coarse_cells is None:
            raise ValueError("Refinement aggregation requires --coarse-cells")
        coarse = pd.read_csv(coarse_cells)
        expected_coarse_jobs = coarse_jobs(
            data_dir=jobs[0].data_dir,
            output_root=jobs[0].output_root,
        )
        _validate_aggregated_cells(coarse, expected_coarse_jobs)
        combined = _sort_cells(pd.concat([coarse, stage_cells], ignore_index=True))
        key = [
            "specification",
            "regularization_c",
            "train_start",
            "validation_year",
        ]
        if combined.duplicated(key).any():
            raise ValueError("Coarse and refinement cells overlap")
        combined_horizons, combined_summary = model_selection.aggregate_brier_cells(combined)
        destinations.extend(
            _write_tables(
                output_dir / "logistic_selection_core",
                combined,
                combined_horizons,
                combined_summary,
            )
        )
        winner = combined_summary.iloc[0]
        decision = pd.DataFrame(
            [{
                **winner.to_dict(),
                "model_file": str(model_output),
                "training_years": "2004-2007",
                "selection_metric": "raw_brier_equal_horizon_weight",
            }]
        )
        decision_path = output_dir / "logistic_selection_decision.csv"
        _atomic_csv(decision, decision_path)
        destinations.append(decision_path)
    return destinations


def finalize_selection(
    decision_file: str | Path,
    *,
    data_dir: str | Path = config.SELECTION_DATA_DIR,
    model_output: str | Path = config.SELECTED_LOGISTIC_MODEL_FILE,
    overwrite: bool = False,
) -> Path:
    """Refit the declared core winner on the frozen 2004--2007 sample."""
    model_output = Path(model_output)
    if model_output.exists() and not overwrite:
        raise FileExistsError(
            f"Selected model already exists: {model_output}; "
            "pass --overwrite to replace it"
        )
    decision = pd.read_csv(decision_file)
    if len(decision) != 1:
        raise ValueError("Selection decision must contain exactly one row")
    row = decision.iloc[0]
    specification = FeatureSpecification(
        str(row["continuous_form"]), str(row["interactions"])
    )
    data = model_selection.load_selection_years(data_dir, config.TRAIN_YEARS)
    training = pd.concat(
        [data[year] for year in config.TRAIN_YEARS], ignore_index=True
    )
    selected = model_selection.fit_selected_model(
        training, specification, float(row["regularization_c"])
    )
    model_selection.save_selected_model(selected, model_output)
    return model_output


def write_slurm_array(
    jobs: list[RawLogisticJob],
    *,
    destination: str | Path,
    repo_dir: str | Path,
    activate: str | None = None,
    account: str = "torch_pr_609_general",
    time_limit: str = "8:00:00",
    memory: str = "32G",
    max_concurrent: int | None = None,
) -> tuple[Path, Path]:
    """Write a manifest and Slurm array script without submitting it."""
    if not jobs:
        raise ValueError("jobs cannot be empty")
    if max_concurrent is not None and max_concurrent <= 0:
        raise ValueError("max_concurrent must be positive")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest = write_manifest(jobs, destination / "logistic_selection_jobs.json")
    array = f"0-{len(jobs) - 1}"
    if max_concurrent is not None:
        array += f"%{max_concurrent}"
    activation = f"source {_expandable_quote(activate)}\n" if activate else ""
    repo_dir = Path(repo_dir).resolve()
    script = destination / "logistic_selection_jobs.slurm"
    contents = f"""#!/bin/bash
#SBATCH --time={time_limit}
#SBATCH --job-name=hmda-logistic-{jobs[0].stage}
#SBATCH --account={account}
#SBATCH --output={destination.as_posix()}/%x_%A_%a.out
#SBATCH --error={destination.as_posix()}/%x_%A_%a.err
#SBATCH --mem={memory}
#SBATCH --array={array}

set -euo pipefail
{activation}cd {_expandable_quote(str(repo_dir))}
python scripts/run_logistic_selection_job.py \\
    --manifest {_expandable_quote(str(manifest))} \\
    --job-index "${{SLURM_ARRAY_TASK_ID}}"
"""
    script.write_text(contents)
    return manifest, script


def write_finalize_slurm(
    *,
    destination: str | Path,
    repo_dir: str | Path,
    decision_file: str | Path = config.TABLE_DIR / "logistic_selection_decision.csv",
    data_dir: str | Path = config.SELECTION_DATA_DIR,
    model_output: str | Path = config.SELECTED_LOGISTIC_MODEL_FILE,
    activate: str | None = None,
    account: str = "torch_pr_609_general",
    time_limit: str = "8:00:00",
    memory: str = "32G",
) -> Path:
    """Write the single-job Slurm script for the selected-model refit."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    repo_dir = Path(repo_dir).resolve()
    decision_file = Path(decision_file).resolve()
    data_dir = Path(data_dir).resolve()
    model_output = Path(model_output).resolve()
    activation = f"source {_expandable_quote(activate)}\n" if activate else ""
    script = destination / "finalize_logistic_selection.slurm"
    script.write_text(
        f"""#!/bin/bash
#SBATCH --time={time_limit}
#SBATCH --job-name=hmda-logistic-final
#SBATCH --account={account}
#SBATCH --output={destination.as_posix()}/%x_%j.out
#SBATCH --error={destination.as_posix()}/%x_%j.err
#SBATCH --mem={memory}

set -euo pipefail
{activation}cd {_expandable_quote(str(repo_dir))}
python scripts/finalize_logistic_selection.py \\
    --decision {_expandable_quote(str(decision_file))} \\
    --data-dir {_expandable_quote(str(data_dir))} \\
    --model-output {_expandable_quote(str(model_output))}
"""
    )
    return script


def _validate_coarse_summary(summary: pd.DataFrame) -> None:
    required = {
        "specification", "regularization_c", "n_horizons", "n_cells", "selection_brier"
    }
    missing = required - set(summary)
    if missing:
        raise ValueError(f"Coarse summary missing columns: {sorted(missing)}")
    expected_specs = {item.name for item in core_specifications()}
    if set(summary["specification"]) != expected_specs or len(summary) != 48:
        raise ValueError("Coarse summary must contain all 12 specifications and four C values")
    expected_c = {float(value) for value in config.LOGISTIC_SELECTION_COARSE_C}
    for specification, rows in summary.groupby("specification"):
        if set(rows["regularization_c"].astype(float)) != expected_c:
            raise ValueError(f"Incomplete coarse C grid for {specification}")
    if not (summary["n_horizons"] == 9).all() or not (summary["n_cells"] == 45).all():
        raise ValueError("Coarse summary does not contain the complete reverse design")


def _validate_aggregated_cells(
    cells: pd.DataFrame, jobs: list[RawLogisticJob]
) -> None:
    required = {
        "specification",
        "regularization_c",
        "train_start",
        "validation_year",
    }
    missing = required - set(cells)
    if missing:
        raise ValueError(f"Selection cells missing columns: {sorted(missing)}")
    expected = {
        (job.specification, float(value), job.train_start, validation_year)
        for job in jobs
        for value in job.c_values
        for validation_year in fold_for_start(job.train_start).validation_years
    }
    observed = list(
        zip(
            cells["specification"].astype(str),
            cells["regularization_c"].astype(float),
            cells["train_start"].astype(int),
            cells["validation_year"].astype(int),
        )
    )
    if set(observed) != expected or len(observed) != len(expected):
        raise ValueError("Selection cells are missing, duplicate, or unexpected")


def _validate_shard(job: RawLogisticJob, shard: dict, *, validate_artifacts: bool) -> None:
    if shard.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported raw-logistic shard schema")
    restored = RawLogisticJob.from_dict(shard.get("job", {}))
    if restored != job:
        raise ValueError("Raw-logistic shard does not match its planned job")
    fold = fold_for_start(job.train_start)
    cells = pd.DataFrame(shard.get("cells", []))
    expected = {
        (float(value), int(year))
        for value in job.c_values
        for year in fold.validation_years
    }
    if cells.empty or not {"regularization_c", "validation_year"} <= set(cells):
        raise ValueError("Raw-logistic shard has no valid cells")
    observed = list(
        zip(
            cells["regularization_c"].astype(float),
            cells["validation_year"].astype(int),
        )
    )
    if set(observed) != expected or len(observed) != len(expected):
        raise ValueError("Raw-logistic shard cells are missing, duplicate, or unexpected")
    if set(cells["specification"]) != {job.specification} or set(cells["train_start"]) != {job.train_start}:
        raise ValueError("Raw-logistic shard cell identity is inconsistent")
    paths = [Path(value) for value in shard.get("artifact_paths", [])]
    specification = specification_from_name(job.specification)
    expected_paths = {
        model_selection.selected_model_path(
            fold.train_years,
            specification,
            value,
            Path(job.output_root) / "models",
        )
        for value in job.c_values
    }
    if set(paths) != expected_paths or len(paths) != len(expected_paths):
        raise ValueError("Raw-logistic shard artifact list is incomplete")
    if validate_artifacts:
        for path in paths:
            metadata = artifacts.validate_existing_artifact(
                path, allow_legacy=False
            )
            if (
                metadata is None
                or metadata.configuration.family != "raw_logistic"
                or metadata.configuration.specification != job.specification
                or metadata.train_years != fold.train_years
                or float(metadata.configuration.parameter_dict()["C"])
                not in job.c_values
            ):
                raise ValueError(f"Artifact metadata does not match job: {path}")


def _sort_cells(cells: pd.DataFrame) -> pd.DataFrame:
    return cells.sort_values(
        ["specification", "regularization_c", "horizon", "validation_year", "train_start"]
    ).reset_index(drop=True)


def _write_tables(prefix, cells, horizons, summary):
    destinations = []
    for suffix, frame in (("cells", cells), ("horizons", horizons), ("summary", summary)):
        path = Path(f"{prefix}_{suffix}.csv")
        _atomic_csv(frame, path)
        destinations.append(path)
    return destinations


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    _atomic_bytes(path, frame.to_csv(index=False).encode())


def _atomic_bytes(path: Path, values: bytes, *, replace: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(values)
            file.flush()
            os.fsync(file.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise FileExistsError(f"Conflicting shard already exists: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _expandable_quote(value: str) -> str:
    if any(character in value for character in ('"', "`", "\\", "\n", "\r")):
        raise ValueError("cluster paths cannot contain quotes, backticks, or newlines")
    return f'"{value}"'
