#!/usr/bin/env python3
"""Run one raw-logistic model-selection manifest task."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from hmda_seconds.model_selection_cluster import execute_job, read_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--job-index", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index = args.job_index
    if index is None:
        value = os.environ.get("SLURM_ARRAY_TASK_ID")
        if value is None:
            raise ValueError("--job-index or SLURM_ARRAY_TASK_ID is required")
        index = int(value)
    jobs = read_manifest(args.manifest)
    if not 0 <= index < len(jobs):
        raise IndexError(f"Job index {index} outside manifest range 0-{len(jobs) - 1}")
    print(execute_job(jobs[index]))


if __name__ == "__main__":
    main()
