#!/usr/bin/env python3
"""Run reverse-fold density-ratio estimation of second-lien shares."""

from __future__ import annotations

import argparse
from pathlib import Path

from hmda_seconds import config, mixture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=config.SELECTION_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.TABLE_DIR)
    parser.add_argument(
        "--model-input", type=Path, default=config.SELECTED_LOGISTIC_MODEL_FILE
    )
    parser.add_argument(
        "--train-start",
        type=int,
        action="append",
        help="Run only folds with this first training year (repeatable)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = mixture.run_reverse_mixture_validation(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_file=args.model_input,
        train_starts=args.train_start,
    )
    print(results["summary"].to_string(index=False))


if __name__ == "__main__":
    main()
