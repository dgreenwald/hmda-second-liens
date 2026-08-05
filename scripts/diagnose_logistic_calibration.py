#!/usr/bin/env python3
"""Run Step 6 calibration diagnostics for the selected logistic model."""

from __future__ import annotations

import argparse
from pathlib import Path

from hmda_seconds import calibration, config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=config.SELECTION_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.TABLE_DIR)
    parser.add_argument("--figure-dir", type=Path, default=config.FIGURE_DIR)
    parser.add_argument(
        "--model-input", type=Path, default=config.SELECTED_LOGISTIC_MODEL_FILE
    )
    parser.add_argument(
        "--bins", type=int, default=calibration.DEFAULT_N_BINS
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = calibration.run_calibration_diagnostics(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
        model_file=args.model_input,
        n_bins=args.bins,
    )
    print("Reverse-validation calibration summary")
    print(results["reverse_summary"].to_string(index=False))
    print("Forward-regime calibration metrics")
    print(results["forward_metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
