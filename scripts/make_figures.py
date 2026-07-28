#!/usr/bin/env python3
"""Render the LTI diagnostic histograms from binned cells."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hmda_seconds import config, diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=config.HISTOGRAM_CELLS_PARQUET)
    parser.add_argument("--output-dir", type=Path, default=config.FIGURE_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cells = pd.read_parquet(args.input)
    paths = diagnostics.render_all(cells, args.output_dir)
    print(f"Rendered {len(paths)} figures to {args.output_dir}")


if __name__ == "__main__":
    main()
