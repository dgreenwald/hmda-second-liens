#!/usr/bin/env python3
"""Evaluate purchaser-specific log-LTI spline interactions."""

from __future__ import annotations

import argparse
from pathlib import Path

from hmda_seconds import config, model_selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=config.SELECTION_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.TABLE_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = model_selection.run_spline_purchaser_challenger(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )
    print(results["comparison"].to_string(index=False))


if __name__ == "__main__":
    main()
