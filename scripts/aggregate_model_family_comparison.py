#!/usr/bin/env python3
"""Validate and aggregate the common four-finalist diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

from hmda_seconds import config, model_family_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=config.TABLE_DIR)
    parser.add_argument("--figure-dir", type=Path, default=config.FIGURE_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in model_family_comparison.aggregate_comparison(
        args.manifest, output_dir=args.output_dir, figure_dir=args.figure_dir
    ):
        print(path)


if __name__ == "__main__":
    main()
