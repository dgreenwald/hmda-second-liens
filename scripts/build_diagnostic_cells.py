#!/usr/bin/env python3
"""Build binned LTI histogram cells from the classified extract."""

from __future__ import annotations

import argparse
from pathlib import Path

from hmda_seconds import config, diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=config.CLASSIFY_PARQUET)
    parser.add_argument("--output", type=Path, default=config.HISTOGRAM_CELLS_PARQUET)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cells = diagnostics.run_build_cells(args.input, args.output)
    print(f"Wrote {len(cells):,} histogram cells to {args.output}")


if __name__ == "__main__":
    main()
