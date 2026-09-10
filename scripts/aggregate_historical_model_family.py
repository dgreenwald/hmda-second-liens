#!/usr/bin/env python3
"""Validate and aggregate all annual Step 10 historical comparison shards."""

from __future__ import annotations

import argparse
from pathlib import Path

from hmda_seconds import config, historical_model_family


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=config.TABLE_DIR)
    parser.add_argument("--figure-dir", type=Path, default=config.FIGURE_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in historical_model_family.aggregate_historical(
        args.manifest,
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
    ):
        print(path)


if __name__ == "__main__":
    main()
