#!/usr/bin/env python3
"""Apply the fitted Random Forest to every HMDA year, 1990-2016."""

from __future__ import annotations

import argparse
from pathlib import Path

from hmda_seconds import classify, config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=config.MODEL_FILE)
    parser.add_argument("--output", type=Path, default=config.CLASSIFY_PARQUET)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = classify.run_classification(args.model, args.output)
    print(
        f"Classified {len(df):,} rows across {len(config.APPLY_YEARS)} years "
        f"and wrote {args.output}"
    )


if __name__ == "__main__":
    main()
