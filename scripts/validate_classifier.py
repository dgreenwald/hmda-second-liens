#!/usr/bin/env python3
"""Validate the fitted RF classifier.

Runs baselines, an out-of-bag check, a feature ablation, and a small
hyperparameter grid from the training extract alone. Out-of-time metrics,
the 2004-boundary continuity check, and out-of-time baseline comparison
additionally require the full classified extract (classify_all_years.py's
output); if that file isn't present yet, this script writes everything it
can and reports what it skipped rather than failing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hmda_seconds import config, validate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", type=Path, default=config.TRAIN_PARQUET)
    parser.add_argument("--classify-input", type=Path, default=config.CLASSIFY_PARQUET)
    parser.add_argument("--output-dir", type=Path, default=config.TABLE_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = validate.run_validation_workflow(
        train_input=args.train_input,
        classify_input=args.classify_input,
        output_dir=args.output_dir,
    )
    print(f"OOB score: {results['oob_score']:.4f}")
    print(results["feature_ablation"].to_string(index=False))
    print(results["hyperparameter_robustness"].to_string(index=False))
    if results["out_of_time_skipped"]:
        print(
            f"{args.classify_input} not found -- skipped out-of-time metrics, "
            "the continuity check, and out-of-time model comparison. Run "
            "classify_all_years.py, then re-run this script to fill those in."
        )
        return
    print(results["out_of_time_metrics"].to_string())
    print(results["rf_vs_logistic_mcnemar"].to_string())
    print(f"Wrote validation tables to {args.output_dir}")


if __name__ == "__main__":
    main()
