#!/usr/bin/env python3
"""Run reverse-temporal logistic specification and ridge selection."""

from __future__ import annotations

import argparse
from pathlib import Path

from hmda_seconds import config, model_selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=config.SELECTION_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.TABLE_DIR)
    parser.add_argument(
        "--model-output", type=Path, default=config.SELECTED_LOGISTIC_MODEL_FILE
    )
    parser.add_argument("--skip-geography", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = model_selection.run_model_selection(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_file=args.model_output,
        include_geography=not args.skip_geography,
    )
    print(results["decision"].to_string(index=False))
    print(f"Wrote logistic selection outputs to {args.output_dir}")
    print(f"Wrote selected logistic model to {args.model_output}")


if __name__ == "__main__":
    main()
