#!/usr/bin/env python3
"""Compare aggregate coverage of source-specific annual HMDA parquet files."""

from __future__ import annotations

import argparse
from pathlib import Path

from py_tools.datasets import hmda

from hmda_seconds.hmda_source_comparison import compare_sources, write_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2017)
    parser.add_argument(
        "--source", action="append", dest="sources", default=None
    )
    parser.add_argument("--data-dir", type=Path, default=Path(hmda.default_dir))
    parser.add_argument("--batch-size", type=int, default=250_000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = tuple(args.sources or ("cfpb", "ffiec_three_year"))
    report = compare_sources(
        args.year, sources, data_dir=args.data_dir, batch_size=args.batch_size
    )
    print(write_report(report, args.output), end="")


if __name__ == "__main__":
    main()
