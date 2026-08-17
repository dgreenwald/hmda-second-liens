#!/usr/bin/env python3
"""Generate or submit the selected HMDA-only boosting final-refit job."""

from __future__ import annotations

import argparse
from pathlib import Path

from py_tools.cluster import submit_slurm

from hmda_seconds import config
from hmda_seconds.hmda_only_boosting_selection import write_finalize_slurm

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path)
    parser.add_argument(
        "--decision",
        type=Path,
        default=config.TABLE_DIR / "hmda_only_boosting_decision.csv",
    )
    parser.add_argument("--data-dir", type=Path, default=config.SELECTION_DATA_DIR)
    parser.add_argument(
        "--model-output",
        type=Path,
        default=config.HMDA_ONLY_SELECTED_BOOSTING_MODEL_FILE,
    )
    parser.add_argument("--activate", default=config.SLURM_ACTIVATE)
    parser.add_argument("--account", default=config.SLURM_ACCOUNT)
    parser.add_argument("--time", default=config.SLURM_TIME)
    parser.add_argument("--memory", default=config.SLURM_MEMORY)
    parser.add_argument("--submit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = args.destination or (
        config.OUTPUT_DIR / "slurm" / "hmda_only_boosting" / "final"
    )
    script = write_finalize_slurm(
        destination=destination,
        repo_dir=REPOSITORY_ROOT,
        decision_file=args.decision,
        data_dir=args.data_dir,
        model_output=args.model_output,
        activate=args.activate,
        account=args.account,
        time_limit=args.time,
        memory=args.memory,
    )
    print(f"Wrote {script}")
    if args.submit:
        submission = submit_slurm(script)
        print(f"Submitted batch job {submission.job_id}")
    else:
        print("No job was submitted. Pass --submit to submit automatically.")


if __name__ == "__main__":
    main()
