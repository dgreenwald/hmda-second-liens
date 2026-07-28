#!/usr/bin/env python3
"""Apply the fitted Random Forest to every HMDA year, 1990-2016."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from hmda_seconds import classify, config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=config.MODEL_FILE)
    parser.add_argument("--output", type=Path, default=config.CLASSIFY_PARQUET)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()

    df = classify.classify_all_years(model_file=args.model)
    elapsed = time.time() - start
    print(
        f"Classified {len(df):,} rows across {len(config.APPLY_YEARS)} years "
        f"({elapsed:.0f}s)"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
