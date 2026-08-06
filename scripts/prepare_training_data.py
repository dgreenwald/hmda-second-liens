#!/usr/bin/env python3
"""Build the concatenated HMDA training extract for TRAIN_YEARS."""

from __future__ import annotations

import argparse
from pathlib import Path

from hmda_seconds import clean, config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=config.TRAIN_PARQUET)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = clean.prepare_training_data(args.output)

    print(f"\nWrote {len(df):,} rows to {args.output}")
    print(df.groupby("year")[config.LABEL_VAR].value_counts())


if __name__ == "__main__":
    main()
