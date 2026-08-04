#!/usr/bin/env python3
"""Audit HMDA sample attrition and FHFA county-HPI coverage, 1990-2016."""

from __future__ import annotations

import argparse
from pathlib import Path

from hmda_seconds import audit, config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yearly-dir", type=Path, default=config.HMDA_YEARLY_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.TABLE_DIR)
    parser.add_argument("--batch-size", type=int, default=1_000_000)
    parser.add_argument("--start-year", type=int, default=min(config.APPLY_YEARS))
    parser.add_argument("--end-year", type=int, default=max(config.APPLY_YEARS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    years = range(args.start_year, args.end_year + 1)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stages, classes, coverage, hpi = audit.run_sample_audit(
        years=years,
        yearly_dir=args.yearly_dir,
        batch_size=args.batch_size,
    )
    stages.to_csv(args.output_dir / "sample_attrition_by_year.csv", index=False)
    classes.to_csv(
        args.output_dir / "sample_attrition_by_lien_status.csv", index=False
    )
    coverage.to_csv(args.output_dir / "fhfa_coverage_by_year.csv", index=False)
    hpi.to_csv(args.output_dir / "fhfa_hpi_level_comparability.csv", index=False)

    print(f"Wrote sample audit tables to {args.output_dir}")


if __name__ == "__main__":
    main()
