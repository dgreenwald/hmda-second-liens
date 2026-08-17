#!/usr/bin/env python3
"""Refit the selected unrestricted boosting model on 2004--2007."""

from __future__ import annotations

import argparse
from pathlib import Path

from hmda_seconds import config
from hmda_seconds.boosting_selection_cluster import fit_final_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decision",
        type=Path,
        default=config.TABLE_DIR / "boosting_challenger_decision.csv",
    )
    parser.add_argument("--data-dir", type=Path, default=config.SELECTION_DATA_DIR)
    parser.add_argument(
        "--model-output", type=Path, default=config.SELECTED_BOOSTING_MODEL_FILE
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        fit_final_model(
            args.decision,
            data_dir=args.data_dir,
            model_output=args.model_output,
            overwrite=args.overwrite,
        )
    )


if __name__ == "__main__":
    main()
