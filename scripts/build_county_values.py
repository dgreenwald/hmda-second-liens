#!/usr/bin/env python3
"""Scale annual FHFA county HPIs to Zillow dollar levels and audit fit quality."""

from __future__ import annotations

import argparse
from pathlib import Path

from hmda_seconds import clean, config, county_values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=config.TABLE_DIR)
    parser.add_argument("--vintage", default=config.ZILLOW_VINTAGE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fhfa = clean.load_fhfa_county_hpi()
    zillow = county_values.load_zillow_county_zhvi(vintage=args.vintage)
    zillow_annual = county_values.annualize_zillow_county(zillow)
    scales, overlap = county_values.estimate_county_scales(fhfa, zillow_annual)
    support = county_values.support_summary(scales)

    scales.to_csv(args.output_dir / "zillow_fhfa_county_scales.csv", index=False)
    overlap.to_csv(args.output_dir / "zillow_fhfa_annual_overlap.csv", index=False)
    support.to_csv(args.output_dir / "zillow_fhfa_support_summary.csv", index=False)

    print(support.to_string(index=False))
    print(f"Wrote Zillow/FHFA scaling diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
