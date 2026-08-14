#!/usr/bin/env python3
"""Generate the two-job density-ratio pilot manifest and Slurm array script."""

from __future__ import annotations

import argparse
from pathlib import Path

from hmda_seconds import config
from hmda_seconds.density_ratio.cluster import pilot_jobs, write_slurm_array

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("output/slurm"))
    parser.add_argument("--repo-dir", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--data-dir", type=Path, default=config.SELECTION_DATA_DIR)
    parser.add_argument(
        "--output-root", type=Path, default=config.OUTPUT_DIR / "density_ratio"
    )
    parser.add_argument("--activate", default=config.SLURM_ACTIVATE)
    parser.add_argument("--account", default=config.SLURM_ACCOUNT)
    parser.add_argument("--train-start", type=int, default=2013)
    parser.add_argument("--time", default=config.SLURM_TIME)
    parser.add_argument("--memory", default=config.SLURM_MEMORY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = pilot_jobs(
        data_dir=args.data_dir,
        output_root=args.output_root,
        train_start=args.train_start,
    )
    manifest, script = write_slurm_array(
        jobs,
        destination=args.destination,
        repo_dir=args.repo_dir,
        activate=args.activate,
        account=args.account,
        time_limit=args.time,
        memory=args.memory,
    )
    print(f"Wrote {manifest}")
    print(f"Wrote {script}")
    print("No jobs were submitted.")


if __name__ == "__main__":
    main()
