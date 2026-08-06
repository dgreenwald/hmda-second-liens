#!/usr/bin/env python3
"""Run the fair full-sample RF-versus-logistic estimator benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from hmda_seconds import benchmark, config


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
    results = benchmark.run_benchmark(
        train_input=args.train_input,
        yearly_dir=args.yearly_dir,
        output_dir=args.output_dir,
        rf_output=args.rf_output,
        logistic_output=args.logistic_output,
    )
    print(results["model_summary"].to_string(index=False))
    print(results["metrics_pooled"].to_string(index=False))
    print(f"Wrote estimator benchmark tables to {args.output_dir}")


if __name__ == "__main__":
    main()
