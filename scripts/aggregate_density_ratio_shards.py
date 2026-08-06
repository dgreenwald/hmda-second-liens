#!/usr/bin/env python3
"""Validate and aggregate every density-ratio shard planned by a manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from hmda_seconds.density_ratio.cluster import aggregate_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for destination in aggregate_manifest(args.manifest, args.output_dir):
        print(destination)


if __name__ == "__main__":
    main()
