#!/usr/bin/env python3
"""Fit the logistic lien-status classifier on the 2004-2007 training extract."""

from __future__ import annotations

import argparse
from pathlib import Path

from hmda_seconds import config, logistic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=config.TRAIN_PARQUET)
    parser.add_argument("--output", type=Path, default=config.LOGISTIC_MODEL_FILE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logistic.run_training(args.input, args.output)
    print(f"Saved fitted logistic model to {args.output}")


if __name__ == "__main__":
    main()
