#!/usr/bin/env python3
"""Generate the frozen first-order logistic manifest and Slurm array script."""

from __future__ import annotations

import argparse
from pathlib import Path

from hmda_seconds import config
from hmda_seconds.density_ratio.cluster import (
    first_order_logistic_jobs,
    write_slurm_array,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination", type=Path, default=Path("output/slurm/first_order")
    )
    parser.add_argument("--repo-dir", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--data-dir", type=Path, default=config.SELECTION_DATA_DIR)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=config.OUTPUT_DIR / "density_ratio",
    )
    parser.add_argument("--activate", default=config.SLURM_ACTIVATE)
    parser.add_argument("--account", default=config.SLURM_ACCOUNT)
    parser.add_argument("--time", default=config.SLURM_TIME)
    parser.add_argument("--memory", default=config.SLURM_MEMORY)
    parser.add_argument("--max-concurrent", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = first_order_logistic_jobs(
        data_dir=args.data_dir,
        output_root=args.output_root,
    )
    manifest, script = write_slurm_array(
        jobs,
        destination=args.destination,
        repo_dir=args.repo_dir,
        activate=args.activate,
        account=args.account,
        time_limit=args.time,
        memory=args.memory,
        job_name="hmda-logistic-first-order",
        max_concurrent=args.max_concurrent,
    )
    print(f"Wrote {manifest}")
    print(f"Wrote {script}")
    print(f"Planned {len(jobs)} jobs; no jobs were submitted.")


if __name__ == "__main__":
    main()
