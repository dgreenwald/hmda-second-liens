"""Canonical Slurm workflow for Step 10 historical model-family application."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from py_tools import cluster as cluster_tools

from . import config, historical_model_family


@dataclass(frozen=True)
class PreparedHistoricalWorkflow:
    orchestration_dir: Path
    manifest: Path
    array_script: Path
    aggregate_script: Path


def prepare_workflow(
    *,
    repository_root: str | Path,
    orchestration_dir: str | Path = config.OUTPUT_DIR / "slurm" / "historical_model_family",
    comparison_manifest: str | Path = config.OUTPUT_DIR / "slurm" / "model_family_comparison" / "density_ratio_jobs.json",
    selection_data_dir: str | Path = config.SELECTION_DATA_DIR,
    hmda_data_dir: str | Path = config.HMDA_DATA_DIR,
    output_root: str | Path = config.OUTPUT_DIR / "historical_model_family",
    table_dir: str | Path = config.TABLE_DIR,
    figure_dir: str | Path = config.FIGURE_DIR,
    activate: str | None = config.SLURM_ACTIVATE,
    account: str | None = config.SLURM_ACCOUNT,
    time_limit: str = config.SLURM_TIME,
    memory: str = config.SLURM_MEMORY,
    aggregate_time: str = "1:00:00",
    aggregate_memory: str = "16G",
    max_concurrent: int | None = config.SLURM_MAX_CONCURRENT,
) -> PreparedHistoricalWorkflow:
    """Generate one canonical year array and dependent aggregation job."""
    repository_root = Path(repository_root).resolve()
    orchestration_dir = Path(orchestration_dir).resolve()
    orchestration_dir.mkdir(parents=True, exist_ok=True)
    manifest = historical_model_family.write_historical_manifest(
        orchestration_dir / "historical_model_family_jobs.json",
        comparison_manifest=comparison_manifest,
        selection_data_dir=selection_data_dir,
        hmda_data_dir=hmda_data_dir,
        output_root=output_root,
    )
    array_script = orchestration_dir / "historical_model_family_jobs.slurm"
    cluster_tools.write_slurm_script(
        cluster_tools.SlurmJob(
            name="hmda-historical-model-family",
            command=(
                "python",
                "scripts/run_historical_model_family_job.py",
                "--manifest",
                manifest.resolve(),
                "--job-index",
                cluster_tools.SLURM_ARRAY_TASK_ID,
            ),
            workdir=repository_root,
            log_dir=orchestration_dir,
            resources=cluster_tools.SlurmResources(
                time=time_limit, memory=memory, account=account
            ),
            activate=activate,
            array=cluster_tools.SlurmArray(
                len(config.APPLY_YEARS), max_concurrent
            ),
        ),
        array_script,
    )
    aggregate_script = orchestration_dir / "aggregate_historical_model_family.slurm"
    cluster_tools.write_slurm_script(
        cluster_tools.SlurmJob(
            name="hmda-historical-model-family-aggregate",
            command=(
                "python",
                repository_root / "scripts" / "aggregate_historical_model_family.py",
                "--manifest",
                manifest.resolve(),
                "--output-dir",
                Path(table_dir).resolve(),
                "--figure-dir",
                Path(figure_dir).resolve(),
            ),
            workdir=repository_root,
            log_dir=orchestration_dir,
            resources=cluster_tools.SlurmResources(
                time=aggregate_time, memory=aggregate_memory, account=account
            ),
            activate=activate,
        ),
        aggregate_script,
    )
    return PreparedHistoricalWorkflow(
        orchestration_dir, manifest, array_script, aggregate_script
    )


def submit_workflow(prepared: PreparedHistoricalWorkflow) -> dict:
    """Submit aggregation with an afterok dependency on the complete year array."""
    array = cluster_tools.submit_slurm(prepared.array_script)
    payload = {"year_array": {"job_id": array.job_id, "dependency": None}}
    receipt = prepared.orchestration_dir / "submission.json"
    _atomic_json(receipt, payload)
    aggregate = _submit_afterok(prepared.aggregate_script, array.job_id)
    payload["aggregate"] = {
        "job_id": aggregate.job_id,
        "dependency": f"afterok:{array.job_id}",
    }
    _atomic_json(receipt, payload)
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
