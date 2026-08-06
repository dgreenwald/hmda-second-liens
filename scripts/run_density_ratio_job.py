#!/usr/bin/env python3
"""Run one immutable density-ratio job locally or as a Slurm array task."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from hmda_seconds import config
from hmda_seconds.density_ratio.cluster import (
    configurations_from_json,
    execute_planned_job,
    expand_job_paths,
    make_job,
)
from hmda_seconds.density_ratio.shards import read_manifest, shard_path

ENVIRONMENT = {
    "stage": "HMDA_DENSITY_RATIO_STAGE",
    "family": "HMDA_DENSITY_RATIO_FAMILY",
    "specification": "HMDA_DENSITY_RATIO_SPEC",
    "train_start": "HMDA_DENSITY_RATIO_TRAIN_START",
    "output_root": "HMDA_DENSITY_RATIO_OUTPUT_ROOT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--job-index", type=int)
    parser.add_argument("--stage")
    parser.add_argument("--family")
    parser.add_argument("--specification")
    parser.add_argument("--train-start", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--configuration",
        action="append",
        default=[],
        metavar="JSON",
        help='Repeat for each candidate, e.g. --configuration \'{"C": 0.1}\'',
    )
    return parser.parse_args()


def planned_job(args: argparse.Namespace):
    """Resolve manifest mode or explicit CLI-over-environment mode."""
    if args.manifest is not None:
        index = args.job_index
        if index is None:
            value = os.environ.get("SLURM_ARRAY_TASK_ID")
            if value is None:
                raise ValueError("--job-index or SLURM_ARRAY_TASK_ID is required")
            index = int(value)
        jobs = read_manifest(args.manifest)
        if not 0 <= index < len(jobs):
            raise IndexError(f"job index {index} is outside 0-{len(jobs) - 1}")
        return expand_job_paths(jobs[index])

    stage = _resolve(args.stage, "stage")
    family = _resolve(args.family, "family")
    specification = _resolve(args.specification, "specification")
    train_start = int(_resolve(args.train_start, "train_start"))
    output_root = Path(_resolve(args.output_root, "output_root"))
    data_dir = args.data_dir or config.SELECTION_DATA_DIR
    configurations = configurations_from_json(
        family, specification, args.configuration
    )
    return make_job(
        stage=stage,
        family=family,
        specification=specification,
        train_start=train_start,
        configurations=configurations,
        data_dir=data_dir,
        output_root=output_root,
    )


def main() -> None:
    planned = planned_job(parse_args())
    completed = execute_planned_job(planned)
    print(shard_path(completed.job))


def _resolve(cli_value: object, name: str) -> object:
    if cli_value is not None:
        return cli_value
    value = os.environ.get(ENVIRONMENT[name])
    if value is None:
        option = name.replace("_", "-")
        raise ValueError(f"--{option} or {ENVIRONMENT[name]} is required")
    return value


if __name__ == "__main__":
    main()
