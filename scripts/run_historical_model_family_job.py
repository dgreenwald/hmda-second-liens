#!/usr/bin/env python3
"""Run one immutable annual Step 10 historical comparison job."""

from __future__ import annotations

import argparse
from pathlib import Path

from hmda_seconds import historical_model_family


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--job-index", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(historical_model_family.run_manifest_year(args.manifest, args.job_index))


if __name__ == "__main__":
    main()
