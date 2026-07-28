#!/usr/bin/env python3
"""Build the concatenated HMDA training extract for TRAIN_YEARS.

Wraps hmda_seconds.clean: loads the FHFA county HPI, builds the balanced
panel over the full application window, cleans each training year, and
writes one concatenated parquet.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from hmda_seconds import clean, config


def build_training_data() -> pd.DataFrame:
    df_fhfa = clean.load_fhfa_county_hpi()
    df_fhfa_balanced = clean.build_balanced_fhfa_panel(df_fhfa, config.APPLY_YEARS)

    df_list = []
    for year in config.TRAIN_YEARS:
        print(f"Cleaning {year}...")
        start = time.time()
        df_year = clean.load_and_clean_year(year, df_fhfa_balanced)
        df_list.append(df_year)
        print(f"  {len(df_year):,} rows ({time.time() - start:.1f}s)")

    return pd.concat(df_list, axis=0, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=config.TRAIN_PARQUET)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_training_data()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)

    print(f"\nWrote {len(df):,} rows to {args.output}")
    print(df.groupby("year")[config.LABEL_VAR].value_counts())


if __name__ == "__main__":
    main()
