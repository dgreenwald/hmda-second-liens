#!/usr/bin/env python3
"""Export the frozen logistic predictor with saved annual mixture intercepts."""

import argparse
from pathlib import Path

from hmda_seconds import config
from hmda_seconds.portable_export import export_portable_logistic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument(
        "--annual-file",
        type=Path,
        default=config.TABLE_DIR / "step8_annual_plausibility.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.MODEL_DIR / "portable_logistic",
    )
    parser.add_argument(
        "--years", nargs="+", type=int, default=tuple(config.APPLY_YEARS)
    )
    args = parser.parse_args()
    print(
        export_portable_logistic(
            model_file=args.model,
            annual_file=args.annual_file,
            output_dir=args.output_dir,
            years=args.years,
        )
    )


if __name__ == "__main__":
    main()
