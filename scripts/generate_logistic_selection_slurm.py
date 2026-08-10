#!/usr/bin/env python3
"""Generate or submit a raw-logistic model-selection Slurm array."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hmda_seconds.hmda_conversion import submit_slurm
from hmda_seconds.model_selection_cluster import (
    COARSE_STAGE,
    REFINEMENT_STAGE,
    coarse_jobs,
    refinement_jobs,
    write_slurm_array,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=(COARSE_STAGE, REFINEMENT_STAGE), required=True)
    parser.add_argument("--coarse-summary", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument(
        "--data-dir", default="data/intermediate/logistic_selection"
    )
    parser.add_argument(
        "--output-root", default="output/raw_logistic_selection"
    )
    parser.add_argument("--activate", help="Optional virtual-environment activation script.")
    parser.add_argument("--account", default="torch_pr_609_general")
    parser.add_argument("--time", default="8:00:00")
    parser.add_argument("--memory", default="32G")
    parser.add_argument("--max-concurrent", type=int)
    parser.add_argument("--submit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = args.destination or Path("output/slurm/logistic_selection") / args.stage
    if not destination.is_absolute():
        destination = REPOSITORY_ROOT / destination
    if args.stage == COARSE_STAGE:
        if args.coarse_summary is not None:
            raise ValueError("--coarse-summary is only valid for refinement")
        jobs = coarse_jobs(data_dir=args.data_dir, output_root=args.output_root)
    else:
        if args.coarse_summary is None:
            raise ValueError("--coarse-summary is required for refinement")
        jobs = refinement_jobs(
            pd.read_csv(args.coarse_summary),
            data_dir=args.data_dir,
            output_root=args.output_root,
        )
    manifest, script = write_slurm_array(
        jobs,
        destination=destination,
        repo_dir=REPOSITORY_ROOT,
        activate=args.activate,
        account=args.account,
        time_limit=args.time,
        memory=args.memory,
        max_concurrent=args.max_concurrent,
    )
    print(f"Wrote {manifest} ({len(jobs)} jobs)")
    print(f"Wrote {script}")
    if args.submit:
        print(submit_slurm(script))
    else:
        print("No jobs were submitted. Pass --submit to submit automatically.")


if __name__ == "__main__":
    main()
