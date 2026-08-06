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

import pandas as pd

from hmda_seconds import config, logistic, validate

# A lighter forest than RF_KWARGS (n_estimators=50) for the ablation grid:
# ablation refits once per feature (7 fits total), so this keeps the whole
# sweep to a few minutes instead of scaling up the production training time.
ABLATION_KWARGS = {"n_estimators": 30, "max_depth": 10, "random_state": 17, "n_jobs": -1}

HYPERPARAMETER_GRID = [
    {"n_estimators": 50, "max_depth": 10},  # config.RF_KWARGS
    {"n_estimators": 50, "max_depth": 8},
    {"n_estimators": 50, "max_depth": 15},
    {"n_estimators": 200, "max_depth": 10},
]

# Ablation and the hyperparameter grid each fit several forests; run them on
# a subsample of the full training extract so the sweep stays fast. This is
# a robustness/sensitivity check, not the headline model.
ROBUSTNESS_SUBSAMPLE = 500_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", type=Path, default=config.TRAIN_PARQUET)
    parser.add_argument("--classify-input", type=Path, default=config.CLASSIFY_PARQUET)
    parser.add_argument("--output-dir", type=Path, default=config.TABLE_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_parquet(args.train_input)
    df_robust = df_train.sample(n=min(ROBUSTNESS_SUBSAMPLE, len(df_train)), random_state=17)

    print("Fitting logistic comparator and threshold baseline...")
    logit = logistic.fit(df_train)
    logistic.save(
        logit, config.LEGACY_VALIDATION_MODEL_DIR / "logistic_comparator.pkl"
    )
    threshold_baseline = validate.fit_log_lti_threshold_baseline(
        df_train,
        config.LEGACY_VALIDATION_MODEL_DIR / "log_lti_threshold.pkl",
    )
    print(f"  log_lti threshold baseline: {threshold_baseline.threshold:.4f}")

    print("Computing out-of-bag score...")
    oob = validate.oob_score(
        df_train, model_dir=config.LEGACY_VALIDATION_MODEL_DIR
    )
    pd.Series({"oob_score": oob}).to_csv(args.output_dir / "oob_score.csv")
    print(f"  OOB score: {oob:.4f}")

    print(f"Running feature ablation on a {len(df_robust):,}-row subsample...")
    ablation = validate.feature_ablation(
        df_robust,
        model_dir=config.LEGACY_VALIDATION_MODEL_DIR,
        **ABLATION_KWARGS,
    )
    ablation.to_csv(args.output_dir / "feature_ablation.csv", index=False)
    print(ablation.to_string(index=False))

    print(f"\nRunning hyperparameter grid on a {len(df_robust):,}-row subsample...")
    hyperparams = validate.hyperparameter_robustness(
        df_robust,
        HYPERPARAMETER_GRID,
        model_dir=config.LEGACY_VALIDATION_MODEL_DIR,
        n_jobs=-1,
    )
    hyperparams.to_csv(args.output_dir / "hyperparameter_robustness.csv", index=False)
    print(hyperparams.to_string(index=False))

    if not args.classify_input.exists():
        print(
            f"\n{args.classify_input} not found -- skipping out-of-time metrics, "
            "the continuity check, and out-of-time model comparison. Run "
            "classify_all_years.py, then re-run this script to fill those in."
        )
        return

    print(f"\nLoading {args.classify_input} for out-of-time validation...")
    df_classified = pd.read_parquet(args.classify_input)

    oot = validate.out_of_time_metrics(df_classified)
    oot.to_csv(args.output_dir / "out_of_time_metrics.csv")
    print(oot.to_string())

    continuity = validate.continuity_check(df_classified)
    continuity.to_csv(args.output_dir / "continuity_check.csv")

    print("\nEvaluating comparison estimators out-of-time...")
    logit_pred = logistic.predict(logit, df_classified)
    logit_prob = logistic.predict_proba_second_lien(logit, df_classified)
    logit_oot = validate.evaluate_by_year(df_classified, logit_pred, y_prob=logit_prob)
    logit_oot.to_csv(args.output_dir / "out_of_time_metrics_logistic.csv")

    threshold_pred = threshold_baseline.predict(df_classified)
    threshold_oot = validate.evaluate_by_year(df_classified, threshold_pred)
    threshold_oot.to_csv(args.output_dir / "out_of_time_metrics_threshold_baseline.csv")

    print("\nFormal RF-vs-logistic comparison (McNemar's test, out-of-time)...")
    rf_pred = df_classified[config.PREDICTED_LABEL_VAR].to_numpy()
    comparison = validate.compare_classifiers_by_year(df_classified, rf_pred, logit_pred)
    comparison.to_csv(args.output_dir / "rf_vs_logistic_mcnemar.csv")
    print(comparison.to_string())

    print(f"\nWrote validation tables to {args.output_dir}")


if __name__ == "__main__":
    main()
