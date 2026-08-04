#!/usr/bin/env python3
"""Run the fair full-sample RF-versus-logistic estimator benchmark."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import pandas as pd

from hmda_seconds import benchmark, clean, config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-input", type=Path, default=config.BENCHMARK_TRAIN_PARQUET
    )
    parser.add_argument("--yearly-dir", type=Path, default=config.HMDA_YEARLY_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.TABLE_DIR)
    parser.add_argument(
        "--rf-output", type=Path, default=config.BENCHMARK_RF_MODEL_FILE
    )
    parser.add_argument(
        "--logistic-output", type=Path, default=config.BENCHMARK_LOGISTIC_MODEL_FILE
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading common training sample from {args.train_input}...")
    training = pd.read_parquet(args.train_input)
    forest, logit, model_summary = benchmark.fit_estimators(
        training,
        rf_output=args.rf_output,
        logistic_output=args.logistic_output,
    )
    del training
    gc.collect()

    print("Evaluating both estimators on 2008-2016...")
    county_values = clean.build_county_value_panel(config.APPLY_YEARS)
    results = benchmark.evaluate_estimators(
        forest,
        logit,
        county_values,
        yearly_dir=args.yearly_dir,
    )
    model_summary = model_summary.merge(
        results.pop("prediction_timing"), on="model", validate="one_to_one"
    )

    model_summary.to_csv(args.output_dir / "benchmark_model_summary.csv", index=False)
    for name, frame in results.items():
        frame.to_csv(args.output_dir / f"benchmark_{name}.csv", index=False)

    print(model_summary.to_string(index=False))
    print(results["metrics_pooled"].to_string(index=False))
    print(f"Wrote estimator benchmark tables to {args.output_dir}")


if __name__ == "__main__":
    main()
