#!/usr/bin/env python3
"""Prepare narrow cleaned 2004-2016 files for logistic model selection."""

from __future__ import annotations

import argparse
from pathlib import Path

from hmda_seconds import config, model_selection
from hmda_seconds.logistic_features import CORE_FEATURE_SET, FEATURE_SETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=config.SELECTION_DATA_DIR)
    parser.add_argument("--feature-set", choices=FEATURE_SETS, default=CORE_FEATURE_SET)
    parser.add_argument("--hmda-data-dir", type=Path, default=config.HMDA_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.TABLE_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = (
        config.HMDA_ONLY_SELECTION_DATA_DIR
        if args.feature_set != CORE_FEATURE_SET
        and args.data_dir == config.SELECTION_DATA_DIR
        else args.data_dir
    )
    summary = model_selection.prepare_selection_data(
        data_dir=data_dir,
        hmda_data_dir=args.hmda_data_dir,
        feature_set=args.feature_set,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_name = (
        "logistic_selection_data_summary.csv"
        if args.feature_set == CORE_FEATURE_SET
        else f"{args.feature_set}_logistic_selection_data_summary.csv"
    )
    summary_file = args.output_dir / summary_name
    summary.to_csv(summary_file, index=False)
    print(summary.to_string(index=False))
    print(f"Wrote selection-data summary to {summary_file}")


if __name__ == "__main__":
    main()
