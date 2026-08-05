#!/usr/bin/env python3
"""Run frozen Step 7 threshold, PR, and subgroup diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

from hmda_seconds import config, threshold_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=config.SELECTION_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.TABLE_DIR)
    parser.add_argument("--figure-dir", type=Path, default=config.FIGURE_DIR)
    parser.add_argument(
        "--model-input", type=Path, default=config.SELECTED_LOGISTIC_MODEL_FILE
    )
    parser.add_argument(
        "--fold-model-dir",
        type=Path,
        default=config.MIXTURE_FOLD_MODEL_DIR,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = threshold_diagnostics.run_threshold_diagnostics(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
        model_file=args.model_input,
        fold_model_dir=args.fold_model_dir,
    )
    print("Reverse threshold summary")
    print(results["reverse_threshold_summary"].to_string(index=False))
    print("Forward threshold summary")
    print(results["forward_threshold_summary"].to_string(index=False))


if __name__ == "__main__":
    main()
