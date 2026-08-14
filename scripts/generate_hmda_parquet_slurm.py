#!/usr/bin/env python3
"""Generate one Slurm parquet-conversion task per HMDA year and source."""

from __future__ import annotations

import argparse
from pathlib import Path

from py_tools.cluster import submit_slurm

from hmda_seconds import config
from hmda_seconds.hmda_conversion import (
    conversion_jobs,
    write_conversion_slurm,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, action="append", dest="years")
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        choices=("all", "ffiec_three_year", "ffiec_snapshot", "cfpb", "nara"),
        help="Repeat for multiple sources; 'all' expands to every available source.",
    )
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--data-dir", type=Path, default=config.HMDA_DATA_DIR)
    parser.add_argument(
        "--activate",
        default=config.SLURM_ACTIVATE,
        help="Optional virtual-environment activation script.",
    )
    parser.add_argument("--account", default=config.SLURM_ACCOUNT)
    parser.add_argument("--time", default=config.SLURM_TIME)
    parser.add_argument("--memory", default=config.SLURM_MEMORY)
    parser.add_argument(
        "--max-concurrent", type=int, default=config.SLURM_MAX_CONCURRENT
    )
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit the generated Slurm script with sbatch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = args.destination or (config.OUTPUT_DIR / "slurm" / "hmda")
    if not destination.is_absolute():
        destination = REPOSITORY_ROOT / destination
    jobs = conversion_jobs(args.years, args.sources)
    manifest, script = write_conversion_slurm(
        jobs,
        destination=destination,
        data_dir=args.data_dir,
        repo_dir=REPOSITORY_ROOT,
        activate=args.activate,
        account=args.account,
        time_limit=args.time,
        memory=args.memory,
        max_concurrent=args.max_concurrent,
        chunksize=args.chunksize,
        compression=args.compression,
        overwrite=args.overwrite,
    )
    print(f"Wrote {manifest} ({len(jobs)} jobs)")
    print(f"Wrote {script}")
    if args.submit:
        submission = submit_slurm(script)
        print(f"Submitted batch job {submission.job_id}")
    else:
        print("No jobs were submitted. Pass --submit to submit automatically.")


if __name__ == "__main__":
    main()
