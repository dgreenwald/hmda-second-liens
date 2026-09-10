#!/usr/bin/env python3
"""Generate or submit the four-finalist diagnostic Slurm array."""

from __future__ import annotations

import argparse
from pathlib import Path

from py_tools.cluster import submit_slurm

from hmda_seconds import config, model_family_comparison
from hmda_seconds.density_ratio.cluster import write_slurm_array

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=config.OUTPUT_DIR / "slurm" / "model_family_comparison",
    )
    parser.add_argument("--repo-dir", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--data-dir", type=Path, default=config.SELECTION_DATA_DIR)
    parser.add_argument(
        "--output-root", type=Path, default=config.MODEL_FAMILY_COMPARISON_DIR
    )
    parser.add_argument("--logistic-model", type=Path, default=config.SELECTED_LOGISTIC_MODEL_FILE)
    parser.add_argument("--hmda-logistic-model", type=Path, default=config.HMDA_ONLY_SELECTED_LOGISTIC_MODEL_FILE)
    parser.add_argument("--boosting-model", type=Path, default=config.SELECTED_BOOSTING_MODEL_FILE)
    parser.add_argument("--hmda-boosting-model", type=Path, default=config.HMDA_ONLY_SELECTED_BOOSTING_MODEL_FILE)
    parser.add_argument("--activate", default=config.SLURM_ACTIVATE)
    parser.add_argument("--account", default=config.SLURM_ACCOUNT)
    parser.add_argument("--time", default=config.SLURM_TIME)
    parser.add_argument("--memory", default=config.SLURM_MEMORY)
    parser.add_argument("--max-concurrent", type=int, default=config.SLURM_MAX_CONCURRENT)
    parser.add_argument("--submit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = model_family_comparison.comparison_jobs(
        data_dir=args.data_dir,
        output_root=args.output_root,
        logistic_file=args.logistic_model,
        hmda_logistic_file=args.hmda_logistic_model,
        boosting_file=args.boosting_model,
        hmda_boosting_file=args.hmda_boosting_model,
    )
    manifest, script = write_slurm_array(
        jobs,
        destination=args.destination,
        repo_dir=args.repo_dir,
        activate=args.activate,
        account=args.account,
        time_limit=args.time,
        memory=args.memory,
        job_name="hmda-model-family-diagnostics",
        max_concurrent=args.max_concurrent,
    )
    print(f"Wrote {manifest}")
    print(f"Wrote {script}")
    if args.submit:
        submission = submit_slurm(script)
        print(f"Submitted batch job {submission.job_id}")
    else:
        print("No jobs were submitted. Pass --submit to submit automatically.")


if __name__ == "__main__":
    main()
