#!/usr/bin/env python3
"""Evaluate the fixed Random Forest with annual mixture adjustment."""

from __future__ import annotations

import argparse
from pathlib import Path

from hmda_seconds import config, random_forest_mixture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=config.SELECTION_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.TABLE_DIR)
    parser.add_argument("--figure-dir", type=Path, default=config.FIGURE_DIR)
    parser.add_argument(
        "--fold-model-dir", type=Path, default=config.RF_MIXTURE_FOLD_MODEL_DIR
    )
    parser.add_argument(
        "--final-model", type=Path, default=config.RF_MIXTURE_MODEL_FILE
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = random_forest_mixture.run_random_forest_mixture(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
        fold_model_dir=args.fold_model_dir,
        final_model_file=args.final_model,
    )
    print(results["reverse_summary"].to_string(index=False))
    print(results["forward_summary"].to_string(index=False))


if __name__ == "__main__":
    main()
