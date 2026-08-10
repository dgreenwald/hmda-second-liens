#!/usr/bin/env python3
"""Validate and aggregate a raw-logistic model-selection manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from hmda_seconds import config
from hmda_seconds.model_selection_cluster import aggregate_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=config.TABLE_DIR)
    parser.add_argument("--coarse-cells", type=Path)
    parser.add_argument("--model-output", type=Path, default=config.SELECTED_LOGISTIC_MODEL_FILE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for output in aggregate_manifest(
        args.manifest,
        args.output_dir,
        coarse_cells=args.coarse_cells,
        model_output=args.model_output,
    ):
        print(output)


if __name__ == "__main__":
    main()
