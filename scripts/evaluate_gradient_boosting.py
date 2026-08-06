#!/usr/bin/env python3
"""Evaluate the Step 9 gradient-boosting density-ratio challenger."""

from __future__ import annotations

import argparse
from pathlib import Path

from hmda_seconds import config, gradient_boosting


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=config.SELECTION_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.TABLE_DIR)
    parser.add_argument("--figure-dir", type=Path, default=config.FIGURE_DIR)
    parser.add_argument(
        "--fold-model-dir", type=Path, default=config.BOOSTING_FOLD_MODEL_DIR
    )
    parser.add_argument(
        "--final-model", type=Path, default=config.SELECTED_BOOSTING_MODEL_FILE
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = gradient_boosting.run_boosting_challenger(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        fold_model_dir=args.fold_model_dir,
        final_model_file=args.final_model,
        figure_dir=args.figure_dir,
    )
    print(results["decision"].to_string(index=False))


if __name__ == "__main__":
    main()
