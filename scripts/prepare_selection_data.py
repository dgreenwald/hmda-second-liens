#!/usr/bin/env python3
"""Prepare narrow cleaned 2004-2016 files for logistic model selection."""

from __future__ import annotations

import argparse
from pathlib import Path

from hmda_seconds import config, model_selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=config.SELECTION_DATA_DIR)
    parser.add_argument("--hmda-data-dir", type=Path, default=config.HMDA_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.TABLE_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = model_selection.prepare_selection_data(
        data_dir=args.data_dir,
        hmda_data_dir=args.hmda_data_dir,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = args.output_dir / "logistic_selection_data_summary.csv"
    summary.to_csv(summary_file, index=False)
    print(summary.to_string(index=False))
    print(f"Wrote selection-data summary to {summary_file}")


if __name__ == "__main__":
    main()
