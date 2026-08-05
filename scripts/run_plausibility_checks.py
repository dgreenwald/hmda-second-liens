#!/usr/bin/env python3
"""Run Step 8 historical application and internal plausibility checks."""

from __future__ import annotations

import argparse
from pathlib import Path

from hmda_seconds import config, plausibility


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection-data-dir", type=Path, default=config.SELECTION_DATA_DIR
    )
    parser.add_argument("--yearly-dir", type=Path, default=config.HMDA_YEARLY_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.TABLE_DIR)
    parser.add_argument("--figure-dir", type=Path, default=config.FIGURE_DIR)
    parser.add_argument(
        "--raw-model", type=Path, default=config.SELECTED_LOGISTIC_MODEL_FILE
    )
    parser.add_argument(
        "--fold-model-dir", type=Path, default=config.MIXTURE_FOLD_MODEL_DIR
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = plausibility.run_historical_plausibility(
        selection_data_dir=args.selection_data_dir,
        yearly_dir=args.yearly_dir,
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
        raw_model_file=args.raw_model,
        fold_model_dir=args.fold_model_dir,
    )
    print(results["annual"].to_string(index=False))
    print("Boundary continuity")
    print(results["continuity"].to_string(index=False))
    print("External-source definition mapping")
    print(results["external_sources"].to_string(index=False))


if __name__ == "__main__":
    main()
