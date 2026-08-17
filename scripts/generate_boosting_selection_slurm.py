#!/usr/bin/env python3
"""Generate one stage of the unrestricted boosting selection Slurm array."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hmda_seconds import config
from hmda_seconds.boosting_selection_cluster import (
    REFINEMENT_STAGE,
    SCREEN_STAGE,
    STAGES,
    SURVIVOR_STAGE,
    refinement_jobs,
    screen_jobs,
    survivor_jobs,
)
from hmda_seconds.density_ratio.cluster import write_slurm_array

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--prior-summary", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--repo-dir", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--data-dir", type=Path, default=config.SELECTION_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=config.BOOSTING_CLUSTER_DIR)
    parser.add_argument("--activate", default=config.SLURM_ACTIVATE)
    parser.add_argument("--account", default=config.SLURM_ACCOUNT)
    parser.add_argument("--time", default=config.SLURM_TIME)
    parser.add_argument("--memory", default=config.SLURM_MEMORY)
    parser.add_argument(
        "--max-concurrent", type=int, default=config.SLURM_MAX_CONCURRENT
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage_label = {
        SCREEN_STAGE: "screen",
        SURVIVOR_STAGE: "survivors",
        REFINEMENT_STAGE: "refinement",
    }[args.stage]
    destination = args.destination or (
        config.OUTPUT_DIR / "slurm" / "boosting" / stage_label
    )
    if args.stage == SCREEN_STAGE:
        if args.prior_summary is not None:
            raise ValueError("--prior-summary is not valid for the screen")
        jobs = screen_jobs(data_dir=args.data_dir, output_root=args.output_root)
    else:
        if args.prior_summary is None:
            raise ValueError("--prior-summary is required after the screen")
        summary = pd.read_csv(args.prior_summary)
        builder = survivor_jobs if args.stage == SURVIVOR_STAGE else refinement_jobs
        jobs = builder(summary, data_dir=args.data_dir, output_root=args.output_root)
    manifest, script = write_slurm_array(
        jobs,
        destination=destination,
        repo_dir=args.repo_dir,
        activate=args.activate,
        account=args.account,
        time_limit=args.time,
        memory=args.memory,
        job_name=f"hmda-boost-{stage_label}",
        max_concurrent=args.max_concurrent,
    )
    print(f"Wrote {manifest} ({len(jobs)} jobs)")
    print(f"Wrote {script}")
    print("No jobs were submitted. Review the script, then run sbatch on it.")


if __name__ == "__main__":
    main()
