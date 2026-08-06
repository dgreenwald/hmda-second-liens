"""Fair full-sample Random Forest versus logistic-regression benchmark."""

from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
from py_tools.econometrics.machine_learning import RandomForestWrapper
from sklearn.linear_model import LogisticRegression

from . import clean, config, logistic, train, validate

MODEL_RF = "random_forest_full"
MODEL_LOGIT = "logistic_full"


def run_benchmark(
    train_input: str | Path = config.BENCHMARK_TRAIN_PARQUET,
    yearly_dir: str | Path = config.HMDA_YEARLY_DIR,
    output_dir: str | Path = config.TABLE_DIR,
    rf_output: str | Path = config.BENCHMARK_RF_MODEL_FILE,
    logistic_output: str | Path = config.BENCHMARK_LOGISTIC_MODEL_FILE,
) -> dict[str, pd.DataFrame]:
    """Run and persist the fair full-sample estimator benchmark."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training = pd.read_parquet(train_input)
    forest, logit, model_summary = fit_estimators(
        training, rf_output=rf_output, logistic_output=logistic_output
    )
    del training
    gc.collect()
    county_values = clean.build_county_value_panel(config.APPLY_YEARS)
    results = evaluate_estimators(
        forest, logit, county_values, yearly_dir=yearly_dir
    )
    model_summary = model_summary.merge(
        results.pop("prediction_timing"), on="model", validate="one_to_one"
    )
    model_summary.to_csv(output_dir / "benchmark_model_summary.csv", index=False)
    for name, frame in results.items():
        frame.to_csv(output_dir / f"benchmark_{name}.csv", index=False)
    return {"model_summary": model_summary, **results}


def fit_estimators(
    df_train: pd.DataFrame,
    rf_output: str | Path = config.BENCHMARK_RF_MODEL_FILE,
    logistic_output: str | Path = config.BENCHMARK_LOGISTIC_MODEL_FILE,
) -> tuple[RandomForestWrapper, LogisticRegression, pd.DataFrame]:
    """Fit both estimators on exactly the same full training frame."""
    rf_output = Path(rf_output)
    logistic_output = Path(logistic_output)
    rf_output.parent.mkdir(parents=True, exist_ok=True)
    logistic_output.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    logit = logistic.fit(df_train)
    logistic_fit_seconds = time.perf_counter() - start
    logistic.save(logit, logistic_output)

    start = time.perf_counter()
    forest = train.fit_full(df_train, outfile=str(rf_output))
    rf_fit_seconds = time.perf_counter() - start

    summary = pd.DataFrame(
        [
            {
                "model": MODEL_RF,
                "n_training": len(df_train),
                "fit_seconds": rf_fit_seconds,
                "artifact_bytes": rf_output.stat().st_size,
                "artifact_path": str(rf_output),
            },
            {
                "model": MODEL_LOGIT,
                "n_training": len(df_train),
                "fit_seconds": logistic_fit_seconds,
                "artifact_bytes": logistic_output.stat().st_size,
                "artifact_path": str(logistic_output),
            },
        ]
    )

    # The fitted forest does not need to retain its multi-gigabyte encoded
    # training arrays during the streamed validation pass.
    for attribute in (
        "data",
        "labels",
        "features",
        "train_labels",
        "train_features",
    ):
        setattr(forest, attribute, None)
    gc.collect()
    return forest, logit, summary


def evaluate_estimators(
    forest: RandomForestWrapper,
    logit: LogisticRegression,
    df_county_values: pd.DataFrame,
    years=None,
    yearly_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Evaluate both fitted models on identical yearly HMDA observations."""
    if years is None:
        years = config.VALIDATE_YEARS

    metric_rows = []
    comparison_rows = []
    prediction_seconds = {MODEL_RF: 0.0, MODEL_LOGIT: 0.0}
    pooled_y = []
    pooled_rf_probability = []
    pooled_logit_probability = []

    for year in years:
        frame = clean.load_and_clean_year(
            year, df_county_values, yearly_dir=yearly_dir
        )
        features = logistic.feature_matrix(frame, None, None)
        y_true = frame[config.LABEL_VAR].to_numpy(copy=True)

        start = time.perf_counter()
        rf_probability = forest.rf.predict_proba(features)[
            :, list(forest.rf.classes_).index(config.SECOND_LIEN_CLASS)
        ]
        prediction_seconds[MODEL_RF] += time.perf_counter() - start

        start = time.perf_counter()
        logit_probability = logit.predict_proba(features)[
            :, list(logit.classes_).index(config.SECOND_LIEN_CLASS)
        ]
        prediction_seconds[MODEL_LOGIT] += time.perf_counter() - start

        rf_prediction = _hard_classification(rf_probability)
        logit_prediction = _hard_classification(logit_probability)
        for model, prediction, probability in (
            (MODEL_RF, rf_prediction, rf_probability),
            (MODEL_LOGIT, logit_prediction, logit_probability),
        ):
            row = validate.classification_metrics(y_true, prediction, probability)
            row.update({"year": year, "model": model})
            metric_rows.append(row)

        comparison = validate.mcnemar_test(
            y_true, rf_prediction, logit_prediction
        )
        comparison.update(
            {
                "year": year,
                "model_a": MODEL_RF,
                "model_b": MODEL_LOGIT,
            }
        )
        comparison_rows.append(comparison)
        pooled_y.append(y_true)
        pooled_rf_probability.append(rf_probability)
        pooled_logit_probability.append(logit_probability)

        del frame, features
        gc.collect()

    y_true = np.concatenate(pooled_y)
    rf_probability = np.concatenate(pooled_rf_probability)
    logit_probability = np.concatenate(pooled_logit_probability)
    rf_prediction = _hard_classification(rf_probability)
    logit_prediction = _hard_classification(logit_probability)

    pooled_rows = []
    for model, prediction, probability in (
        (MODEL_RF, rf_prediction, rf_probability),
        (MODEL_LOGIT, logit_prediction, logit_probability),
    ):
        row = validate.classification_metrics(y_true, prediction, probability)
        row["model"] = model
        pooled_rows.append(row)

    pooled_comparison = validate.mcnemar_test(
        y_true, rf_prediction, logit_prediction
    )
    pooled_comparison.update({"model_a": MODEL_RF, "model_b": MODEL_LOGIT})

    timing = pd.DataFrame(
        [
            {"model": model, "prediction_seconds": seconds}
            for model, seconds in prediction_seconds.items()
        ]
    )
    return {
        "metrics_by_year": pd.DataFrame(metric_rows),
        "metrics_pooled": pd.DataFrame(pooled_rows),
        "mcnemar_by_year": pd.DataFrame(comparison_rows),
        "mcnemar_pooled": pd.DataFrame([pooled_comparison]),
        "prediction_timing": timing,
    }


def _hard_classification(probability: np.ndarray) -> np.ndarray:
    return np.where(
        probability >= 0.5,
        config.SECOND_LIEN_CLASS,
        config.FIRST_LIEN_CLASS,
    )
