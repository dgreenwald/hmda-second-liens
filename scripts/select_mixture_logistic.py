#!/usr/bin/env python3
"""Reselect logistic features and ridge strength for mixture-adjusted Brier."""

from __future__ import annotations

import argparse
from pathlib import Path

from hmda_seconds import config, mixture_logistic_selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=config.SELECTION_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.TABLE_DIR)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=config.MIXTURE_LOGISTIC_SELECTION_MODEL_DIR,
    )
    parser.add_argument(
        "--incumbent", type=Path, default=config.SELECTED_LOGISTIC_MODEL_FILE
    )
    parser.add_argument(
        "--final-model",
        type=Path,
        default=config.MIXTURE_SELECTED_LOGISTIC_MODEL_FILE,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = mixture_logistic_selection.run_mixture_logistic_selection(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        incumbent_file=args.incumbent,
        final_model_file=args.final_model,
    )
    print(results["survivors"].to_string(index=False))
    print(results["decision"].to_string(index=False))


if __name__ == "__main__":
    main()
