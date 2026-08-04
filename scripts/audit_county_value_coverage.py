#!/usr/bin/env python3
"""Audit exact HMDA-loan coverage of the Zillow-scaled FHFA panel."""

from __future__ import annotations

import argparse
from pathlib import Path

from hmda_seconds import audit, clean, config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yearly-dir", type=Path, default=config.HMDA_YEARLY_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.TABLE_DIR)
    parser.add_argument("--batch-size", type=int, default=1_000_000)
    parser.add_argument("--start-year", type=int, default=min(config.APPLY_YEARS))
    parser.add_argument("--end-year", type=int, default=max(config.APPLY_YEARS))
    parser.add_argument("--method", default="geometric")
    parser.add_argument(
        "--min-overlap-years", type=int, default=config.ZILLOW_MIN_OVERLAP_YEARS
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    years = range(args.start_year, args.end_year + 1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel = clean.build_county_value_panel(
        years=years,
        method=args.method,
        min_overlap_years=args.min_overlap_years,
    )
    coverage, classes = audit.run_county_value_coverage_audit(
        panel,
        years=years,
        yearly_dir=args.yearly_dir,
        batch_size=args.batch_size,
    )
    output_file = args.output_dir / "county_value_coverage_by_year.csv"
    coverage.to_csv(output_file, index=False)
    class_file = args.output_dir / "county_value_coverage_by_lien_status.csv"
    classes.to_csv(class_file, index=False)
    print(coverage.to_string(index=False))
    print(f"Wrote county-value coverage audits to {output_file} and {class_file}")


if __name__ == "__main__":
    main()
