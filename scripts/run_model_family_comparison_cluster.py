#!/usr/bin/env python3
"""Prepare or submit the complete model-family diagnostic workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from hmda_seconds import config, model_family_cluster

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=config.OUTPUT_DIR / "slurm" / "model_family_comparison_runs",
    )
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--data-dir", type=Path, default=config.SELECTION_DATA_DIR)
    parser.add_argument(
        "--output-root", type=Path, default=config.MODEL_FAMILY_COMPARISON_DIR
    )
    parser.add_argument("--table-dir", type=Path, default=config.TABLE_DIR)
    parser.add_argument("--figure-dir", type=Path, default=config.FIGURE_DIR)
    parser.add_argument("--activate", default=config.SLURM_ACTIVATE)
    parser.add_argument("--account", default=config.SLURM_ACCOUNT)
    parser.add_argument("--fit-time", default=config.SLURM_TIME)
    parser.add_argument("--fit-memory", default=config.SLURM_MEMORY)
    parser.add_argument("--aggregate-time", default="1:00:00")
    parser.add_argument("--aggregate-memory", default="16G")
    parser.add_argument("--max-concurrent", type=int, default=config.SLURM_MAX_CONCURRENT)
    parser.add_argument("--logistic-model", type=Path, default=config.SELECTED_LOGISTIC_MODEL_FILE)
    parser.add_argument("--hmda-logistic-model", type=Path, default=config.HMDA_ONLY_SELECTED_LOGISTIC_MODEL_FILE)
    parser.add_argument("--boosting-model", type=Path, default=config.SELECTED_BOOSTING_MODEL_FILE)
    parser.add_argument("--hmda-boosting-model", type=Path, default=config.HMDA_ONLY_SELECTED_BOOSTING_MODEL_FILE)
    parser.add_argument("--submit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepared = model_family_cluster.prepare_run(
        repository_root=args.repository_root,
        run_root=args.run_root,
        run_id=args.run_id,
        data_dir=args.data_dir,
        output_root=args.output_root,
        table_dir=args.table_dir,
        figure_dir=args.figure_dir,
        activate=args.activate,
        account=args.account,
        fit_time=args.fit_time,
        fit_memory=args.fit_memory,
        aggregate_time=args.aggregate_time,
        aggregate_memory=args.aggregate_memory,
        max_concurrent=args.max_concurrent,
        logistic_file=args.logistic_model,
        hmda_logistic_file=args.hmda_logistic_model,
        boosting_file=args.boosting_model,
        hmda_boosting_file=args.hmda_boosting_model,
    )
    print(f"Prepared run in {prepared.run_dir}")
    print(f"Fit array: {prepared.array_script}")
    print(f"Dependent aggregation: {prepared.aggregate_script}")
    if args.submit:
        submitted = model_family_cluster.submit_run(prepared)
        print(f"Submitted fit array {submitted['fit_array']['job_id']}")
        print(f"Submitted aggregate job {submitted['aggregate']['job_id']}")
    else:
        print("No jobs were submitted. Pass --submit to launch the workflow.")


if __name__ == "__main__":
    main()
