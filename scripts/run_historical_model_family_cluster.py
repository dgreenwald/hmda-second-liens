#!/usr/bin/env python3
"""Prepare or submit the canonical Step 10 historical comparison workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from hmda_seconds import config, historical_model_family_cluster

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument(
        "--comparison-manifest",
        type=Path,
        default=config.OUTPUT_DIR
        / "slurm"
        / "model_family_comparison"
        / "density_ratio_jobs.json",
    )
    parser.add_argument("--selection-data-dir", type=Path, default=config.SELECTION_DATA_DIR)
    parser.add_argument("--hmda-data-dir", type=Path, default=config.HMDA_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=config.OUTPUT_DIR / "historical_model_family")
    parser.add_argument("--table-dir", type=Path, default=config.TABLE_DIR)
    parser.add_argument("--figure-dir", type=Path, default=config.FIGURE_DIR)
    parser.add_argument("--activate", default=config.SLURM_ACTIVATE)
    parser.add_argument("--account", default=config.SLURM_ACCOUNT)
    parser.add_argument("--time", default=config.SLURM_TIME)
    parser.add_argument("--memory", default=config.SLURM_MEMORY)
    parser.add_argument("--aggregate-time", default="1:00:00")
    parser.add_argument("--aggregate-memory", default="16G")
    parser.add_argument("--max-concurrent", type=int, default=config.SLURM_MAX_CONCURRENT)
    parser.add_argument("--submit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepared = historical_model_family_cluster.prepare_workflow(
        repository_root=args.repository_root,
        comparison_manifest=args.comparison_manifest,
        selection_data_dir=args.selection_data_dir,
        hmda_data_dir=args.hmda_data_dir,
        output_root=args.output_root,
        table_dir=args.table_dir,
        figure_dir=args.figure_dir,
        activate=args.activate,
        account=args.account,
        time_limit=args.time,
        memory=args.memory,
        aggregate_time=args.aggregate_time,
        aggregate_memory=args.aggregate_memory,
        max_concurrent=args.max_concurrent,
    )
    print(f"Prepared workflow in {prepared.orchestration_dir}")
    print(f"Year array: {prepared.array_script}")
    print(f"Dependent aggregation: {prepared.aggregate_script}")
    if args.submit:
        submitted = historical_model_family_cluster.submit_workflow(prepared)
        print(f"Submitted year array {submitted['year_array']['job_id']}")
        print(f"Submitted aggregate job {submitted['aggregate']['job_id']}")
    else:
        print("No jobs were submitted. Pass --submit to launch the workflow.")


if __name__ == "__main__":
    main()
