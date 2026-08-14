#!/usr/bin/env python3
"""Combine the eligible HMDA-only boosting stages and declare a winner."""

from __future__ import annotations

import argparse
from pathlib import Path

from hmda_seconds import config
from hmda_seconds.hmda_only_boosting_selection import finalize_selection_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survivor-dir", type=Path, required=True)
    parser.add_argument("--refinement-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=config.TABLE_DIR)
    parser.add_argument(
        "--model-output",
        type=Path,
        default=config.HMDA_ONLY_SELECTED_BOOSTING_MODEL_FILE,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for destination in finalize_selection_tables(
        args.survivor_dir,
        args.refinement_dir,
        args.output_dir,
        model_output=args.model_output,
    ):
        print(destination)


if __name__ == "__main__":
    main()
