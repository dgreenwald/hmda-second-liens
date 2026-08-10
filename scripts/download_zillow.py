#!/usr/bin/env python3
"""Download the pinned Zillow county ZHVI input."""

from __future__ import annotations

import argparse
from pathlib import Path

from hmda_seconds import config, county_values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vintage", default=config.ZILLOW_VINTAGE)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        county_values.download_zillow_county_zhvi(
            args.vintage,
            data_dir=args.data_dir,
            overwrite=args.overwrite,
        )
    )


if __name__ == "__main__":
    main()
