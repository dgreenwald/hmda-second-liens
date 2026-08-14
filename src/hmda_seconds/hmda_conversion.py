"""Generate and execute source-specific HMDA parquet conversion jobs."""

from __future__ import annotations

import json
from pathlib import Path

from py_tools import cluster
from py_tools.datasets import hmda

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

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
    requested_sources = tuple(sources) if sources else ("all",)
    selected_sources = (
        tuple(SOURCE_YEARS)
        if "all" in requested_sources
        else tuple(dict.fromkeys(requested_sources))
    )
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
            raise ValueError(
                f"Requested year(s) unavailable from selected sources: {values}"
            )
    if not jobs:
        raise ValueError("No HMDA conversion jobs selected")
    return jobs


def write_conversion_slurm(
    jobs: list[dict[str, int | str]],
    *,
    destination: str | Path,
    data_dir: str | Path,
    repo_dir: str | Path = REPOSITORY_ROOT,
    activate: str | None = None,
    account: str = "torch_pr_609_general",
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
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")

    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest = destination / "hmda_parquet_jobs.json"
    manifest.write_text(json.dumps(jobs, indent=2) + "\n")

    command = [
        "python",
        "scripts/run_hmda_parquet_job.py",
        "--manifest",
        manifest,
        "--job-index",
        cluster.SLURM_ARRAY_TASK_ID,
        "--data-dir",
        data_dir,
        "--chunksize",
        str(chunksize),
        "--compression",
        compression,
    ]
    if overwrite:
        command.append("--overwrite")
    script = destination / "hmda_parquet_jobs.slurm"
    cluster.write_slurm_script(
        cluster.SlurmJob(
            name="hmda-parquet",
            command=tuple(command),
            workdir=repo_dir,
            log_dir=destination,
            resources=cluster.SlurmResources(
                time=time_limit,
                memory=memory,
                account=account,
            ),
            activate=activate,
            array=cluster.SlurmArray(len(jobs), max_concurrent),
        ),
        script,
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
    job = _manifest_job(manifest, job_index)
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


def conversion_job_name(manifest: str | Path, job_index: int) -> str:
    """Return the concise Slurm name for one manifest entry."""
    job = _manifest_job(manifest, job_index)
    return f"hmda-{job['year']}-{job['source']}"


def _manifest_job(manifest: str | Path, job_index: int) -> dict:
    """Read and validate one conversion job from a manifest."""
    jobs = json.loads(Path(manifest).read_text())
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("HMDA conversion manifest must be a nonempty list")
    if job_index < 0 or job_index >= len(jobs):
        raise IndexError(
            f"Job index {job_index} outside manifest range 0-{len(jobs) - 1}"
        )
    job = jobs[job_index]
    if not isinstance(job, dict) or "year" not in job or "source" not in job:
        raise ValueError(f"Invalid HMDA conversion job at index {job_index}")
    return job
