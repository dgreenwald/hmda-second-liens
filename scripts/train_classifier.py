#!/usr/bin/env python3
"""Fit the Random Forest lien-status classifier on the 2004-2007 training extract."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hmda_seconds import config, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=config.TRAIN_PARQUET)
    parser.add_argument("--output", type=Path, default=config.MODEL_FILE)
    parser.add_argument(
        "--plot", type=Path, default=config.FIGURE_DIR / "rf_importances.pdf"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.input)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.plot.parent.mkdir(parents=True, exist_ok=True)

    train.fit(df, outfile=str(args.output), plotpath=str(args.plot))

    print(f"Saved fitted model to {args.output}")
    print(f"Saved feature-importance plot to {args.plot}")


if __name__ == "__main__":
    main()
