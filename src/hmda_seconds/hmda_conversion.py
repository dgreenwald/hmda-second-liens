"""Generate and execute source-specific HMDA parquet conversion jobs."""

from __future__ import annotations

import json
from pathlib import Path

from py_tools.datasets import hmda

SOURCE_YEARS = {
    "ffiec_three_year": tuple(hmda.THREE_YEAR_LAR_YEARS),
    "ffiec_snapshot": tuple(hmda.SNAPSHOT_LAR_YEARS),
    "cfpb": tuple(hmda.CFPB_LAR_YEARS),
    "nara": tuple(hmda.NATIONAL_ARCHIVES_LAR_YEARS),
}


def conversion_jobs(
    years: list[int] | tuple[int, ...] | None = None,
    sources: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, int | str]]:
    """Return one deterministic job per supported year/source pair."""
    selected_sources = tuple(sources) if sources else tuple(SOURCE_YEARS)
    unknown = sorted(set(selected_sources) - set(SOURCE_YEARS))
    if unknown:
        raise ValueError(f"Unknown HMDA source(s): {', '.join(unknown)}")

    selected_years = {int(year) for year in years} if years else None
    jobs = [
        {"year": year, "source": source}
        for source in selected_sources
        for year in sorted(SOURCE_YEARS[source])
        if selected_years is None or year in selected_years
    ]
    if selected_years is not None:
        matched_years = {int(job["year"]) for job in jobs}
        unsupported = sorted(selected_years - matched_years)
        if unsupported:
            values = ", ".join(str(year) for year in unsupported)
            raise ValueError(f"Requested year(s) unavailable from selected sources: {values}")
    if not jobs:
        raise ValueError("No HMDA conversion jobs selected")
    return jobs


def write_conversion_slurm(
    jobs: list[dict[str, int | str]],
    *,
    destination: str | Path,
    repo_dir: str,
    data_dir: str,
    activate: str,
    time_limit: str = "4:00:00",
    memory: str = "16G",
    max_concurrent: int | None = None,
    chunksize: int = 100_000,
    compression: str = "zstd",
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Write a conversion manifest and Slurm array script without submitting it."""
    if not jobs:
        raise ValueError("jobs cannot be empty")
    if max_concurrent is not None and max_concurrent <= 0:
        raise ValueError("max_concurrent must be positive")
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")

    destination = Path(destination)
    if destination.is_absolute():
        raise ValueError("destination must be relative to the repository root")
    destination.mkdir(parents=True, exist_ok=True)
    manifest = destination / "hmda_parquet_jobs.json"
    manifest.write_text(json.dumps(jobs, indent=2) + "\n")

    array = f"0-{len(jobs) - 1}"
    if max_concurrent is not None:
        array += f"%{max_concurrent}"
    cluster_manifest = f"{repo_dir}/{manifest.as_posix()}"
    cluster_logs = f"{repo_dir}/{destination.as_posix()}"
    overwrite_flag = " --overwrite" if overwrite else ""
    script = destination / "hmda_parquet_jobs.slurm"
    script.write_text(
        f"""#!/bin/bash
#SBATCH --time={time_limit}
#SBATCH --job-name=hmda-parquet
#SBATCH --output={cluster_logs}/%x_%A_%a.out
#SBATCH --error={cluster_logs}/%x_%A_%a.err
#SBATCH --mem={memory}
#SBATCH --array={array}

set -euo pipefail
source {_expandable_quote(activate)}
cd {_expandable_quote(repo_dir)}
/usr/bin/time -v python scripts/run_hmda_parquet_job.py \\
    --manifest {_expandable_quote(cluster_manifest)} \\
    --job-index "${{SLURM_ARRAY_TASK_ID}}" \\
    --data-dir {_expandable_quote(data_dir)} \\
    --chunksize {chunksize} \\
    --compression {compression}{overwrite_flag}
"""
    )
    return manifest, script


def run_conversion_job(
    manifest: str | Path,
    job_index: int,
    *,
    data_dir: str | Path,
    chunksize: int = 100_000,
    compression: str = "zstd",
    overwrite: bool = False,
) -> Path:
    """Convert the single year/source pair at ``job_index``."""
    jobs = json.loads(Path(manifest).read_text())
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("HMDA conversion manifest must be a nonempty list")
    if job_index < 0 or job_index >= len(jobs):
        raise IndexError(f"Job index {job_index} outside manifest range 0-{len(jobs) - 1}")
    job = jobs[job_index]
    year = int(job["year"])
    source = str(job["source"])
    outputs = hmda.convert_lar(
        year,
        source=source,
        data_dir=Path(data_dir).expanduser(),
        overwrite=overwrite,
        chunksize=chunksize,
        compression=compression,
    )
    return Path(outputs[0])


def _expandable_quote(value: str) -> str:
    """Double-quote a trusted cluster path while retaining variable expansion."""
    if any(character in value for character in ('"', "`", "\\", "\n", "\r")):
        raise ValueError("cluster paths cannot contain quotes, backticks, or newlines")
    return f'"{value}"'
