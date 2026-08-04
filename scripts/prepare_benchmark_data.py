#!/usr/bin/env python3
"""Build the preserved estimator-benchmark data with county-value features."""

from __future__ import annotations

import argparse
from pathlib import Path

from hmda_seconds import clean, config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=config.BENCHMARK_TRAIN_PARQUET)
    parser.add_argument("--yearly-dir", type=Path, default=config.HMDA_YEARLY_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = clean.build_training_data(yearly_dir=args.yearly_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    print(f"Wrote {len(frame):,} benchmark training rows to {args.output}")
    print(frame.groupby("year")[config.LABEL_VAR].value_counts())


if __name__ == "__main__":
    main()
