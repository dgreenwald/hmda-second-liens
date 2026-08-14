#!/usr/bin/env python3
"""Generate or submit a raw-logistic model-selection Slurm array."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from py_tools.cluster import submit_slurm

from hmda_seconds import config
from hmda_seconds.logistic_features import CORE_FEATURE_SET, FEATURE_SETS
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
    parser.add_argument(
        "--stage", choices=(COARSE_STAGE, REFINEMENT_STAGE), required=True
    )
    parser.add_argument("--coarse-summary", type=Path)
    parser.add_argument("--feature-set", choices=FEATURE_SETS, default=CORE_FEATURE_SET)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--data-dir", default=config.SELECTION_DATA_DIR)
    parser.add_argument("--output-root", default=config.RAW_LOGISTIC_CLUSTER_DIR)
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
    parser.add_argument("--submit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    restricted = args.feature_set != CORE_FEATURE_SET
    data_dir = (
        config.HMDA_ONLY_SELECTION_DATA_DIR
        if restricted and args.data_dir == config.SELECTION_DATA_DIR
        else args.data_dir
    )
    output_root = (
        config.HMDA_ONLY_RAW_LOGISTIC_CLUSTER_DIR
        if restricted and args.output_root == config.RAW_LOGISTIC_CLUSTER_DIR
        else args.output_root
    )
    destination = args.destination or (
        config.OUTPUT_DIR
        / "slurm"
        / ("hmda_only_logistic_selection" if restricted else "logistic_selection")
        / args.stage
    )
    if not destination.is_absolute():
        destination = REPOSITORY_ROOT / destination
    if args.stage == COARSE_STAGE:
        if args.coarse_summary is not None:
            raise ValueError("--coarse-summary is only valid for refinement")
        jobs = coarse_jobs(
            data_dir=data_dir,
            output_root=output_root,
            feature_set=args.feature_set,
        )
    else:
        if args.coarse_summary is None:
            raise ValueError("--coarse-summary is required for refinement")
        jobs = refinement_jobs(
            pd.read_csv(args.coarse_summary),
            data_dir=data_dir,
            output_root=output_root,
            feature_set=args.feature_set,
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
        submission = submit_slurm(script)
        print(f"Submitted batch job {submission.job_id}")
    else:
        print("No jobs were submitted. Pass --submit to submit automatically.")


if __name__ == "__main__":
    main()
