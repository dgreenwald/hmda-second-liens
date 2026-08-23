"""Prepare and submit the dependent model-family diagnostic workflow."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from py_tools import cluster as cluster_tools

from . import config, model_family_comparison
from .density_ratio.cluster import write_slurm_array


@dataclass(frozen=True)
class PreparedComparisonRun:
    """Paths needed to submit and audit one comparison workflow."""

    run_dir: Path
    manifest: Path
    array_script: Path
    aggregate_script: Path


def prepare_run(
    *,
    repository_root: str | Path,
    run_root: str | Path,
    run_id: str | None = None,
    data_dir: str | Path = config.SELECTION_DATA_DIR,
    output_root: str | Path = config.MODEL_FAMILY_COMPARISON_DIR,
    table_dir: str | Path = config.TABLE_DIR,
    figure_dir: str | Path = config.FIGURE_DIR,
    activate: str | None = config.SLURM_ACTIVATE,
    account: str | None = config.SLURM_ACCOUNT,
    fit_time: str = config.SLURM_TIME,
    fit_memory: str = config.SLURM_MEMORY,
    aggregate_time: str = "1:00:00",
    aggregate_memory: str = "16G",
    max_concurrent: int | None = config.SLURM_MAX_CONCURRENT,
    **artifact_files,
) -> PreparedComparisonRun:
    """Write a run-specific array and its dependent aggregation job."""
    repository_root = Path(repository_root).resolve()
    resolved_id = run_id or datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    if not resolved_id or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        for character in resolved_id
    ):
        raise ValueError("run_id contains unsafe characters")
    run_dir = Path(run_root).resolve() / resolved_id
    if run_dir.exists():
        raise FileExistsError(f"Cluster comparison run already exists: {run_dir}")
    jobs = model_family_comparison.comparison_jobs(
        data_dir=data_dir,
        output_root=output_root,
        **artifact_files,
    )
    manifest, array_script = write_slurm_array(
        jobs,
        destination=run_dir,
        repo_dir=str(repository_root),
        activate=activate,
        account=account,
        time_limit=fit_time,
        memory=fit_memory,
        job_name="hmda-model-family-diagnostics",
        max_concurrent=max_concurrent,
    )
    aggregate_script = run_dir / "aggregate_model_family_comparison.slurm"
    cluster_tools.write_slurm_script(
        cluster_tools.SlurmJob(
            name="hmda-model-family-aggregate",
            command=(
                "python",
                repository_root / "scripts" / "aggregate_model_family_comparison.py",
                "--manifest",
                manifest.resolve(),
                "--output-dir",
                Path(table_dir).resolve(),
                "--figure-dir",
                Path(figure_dir).resolve(),
            ),
            workdir=repository_root,
            log_dir=run_dir,
            resources=cluster_tools.SlurmResources(
                time=aggregate_time,
                memory=aggregate_memory,
                account=account,
            ),
            activate=activate,
        ),
        aggregate_script,
    )
    return PreparedComparisonRun(run_dir, manifest, array_script, aggregate_script)


def submit_run(prepared: PreparedComparisonRun) -> dict[str, dict[str, str | None]]:
    """Submit the fit array and aggregate only after every task succeeds."""
    array = cluster_tools.submit_slurm(prepared.array_script)
    payload = {"fit_array": {"job_id": array.job_id, "dependency": None}}
    submission_file = prepared.run_dir / "submission.json"
    _atomic_json(submission_file, payload)
    aggregate = _submit_afterok(prepared.aggregate_script, array.job_id)
    payload["aggregate"] = {
        "job_id": aggregate.job_id,
        "dependency": f"afterok:{array.job_id}",
    }
    _atomic_json(submission_file, payload)
    return payload


def _submit_afterok(script: Path, dependency: str):
    result = subprocess.run(
        [
            "sbatch",
            "--parsable",
            f"--dependency=afterok:{dependency}",
            "--kill-on-invalid-dep=yes",
            str(script.resolve()),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stdout = result.stdout.strip()
    job_id, separator, cluster_name = stdout.partition(";")
    if not job_id:
        raise RuntimeError(f"Invalid sbatch response: {stdout!r}")
    return cluster_tools.SlurmSubmission(
        job_id=job_id,
        cluster=cluster_name if separator else None,
        stdout=stdout,
    )


def _atomic_json(path: Path, payload: dict) -> None:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
