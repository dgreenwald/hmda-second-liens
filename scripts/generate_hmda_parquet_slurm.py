#!/usr/bin/env python3
"""Generate one Slurm parquet-conversion task per HMDA year and source."""

from __future__ import annotations

import argparse
from pathlib import Path

from hmda_seconds.hmda_conversion import conversion_jobs, write_conversion_slurm

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, action="append", dest="years")
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        choices=("ffiec_three_year", "ffiec_snapshot", "cfpb", "nara"),
        help="Repeat to include multiple sources; defaults to all sources.",
    )
    parser.add_argument("--destination", type=Path, default=Path("output/slurm/hmda"))
    parser.add_argument("--data-dir")
    parser.add_argument(
        "--activate",
        help="Optional virtual-environment activation script.",
    )
    parser.add_argument("--account", default="torch_pr_609_general")
    parser.add_argument("--time", default="4:00:00")
    parser.add_argument("--memory", default="16G")
    parser.add_argument("--max-concurrent", type=int)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = args.destination
    if not destination.is_absolute():
        destination = REPOSITORY_ROOT / destination
    jobs = conversion_jobs(args.years, args.sources)
    manifest, script = write_conversion_slurm(
        jobs,
        destination=destination,
        data_dir=args.data_dir,
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
    print("No jobs were submitted.")


if __name__ == "__main__":
    main()
