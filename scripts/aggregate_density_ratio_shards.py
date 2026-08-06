#!/usr/bin/env python3
"""Validate and aggregate every density-ratio shard planned by a manifest."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from hmda_seconds.density_ratio.cluster import expand_job_paths
from hmda_seconds.density_ratio.shards import (
    aggregate_shards,
    read_manifest,
    shard_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    planned = [expand_job_paths(item) for item in read_manifest(args.manifest)]
    paths = [shard_path(item.job) for item in planned]
    aggregated = aggregate_shards(planned, paths)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("cells", "horizons", "summary"):
        destination = args.output_dir / f"density_ratio_{name}.csv"
        _atomic_csv(getattr(aggregated, name), destination)
        print(destination)


def _atomic_csv(frame, destination: Path) -> None:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
