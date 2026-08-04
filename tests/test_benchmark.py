import pandas as pd

from hmda_seconds import benchmark


def test_full_sample_benchmark_uses_common_rows(
    training_frame, tmp_path, monkeypatch
):
    forest, logit, summary = benchmark.fit_estimators(
        training_frame,
        rf_output=tmp_path / "rf.pkl",
        logistic_output=tmp_path / "logit.pkl",
    )
    validation = training_frame.copy()
    validation["year"] = 2008
    monkeypatch.setattr(
        benchmark.clean,
        "load_and_clean_year",
        lambda *args, **kwargs: validation,
    )

    results = benchmark.evaluate_estimators(
        forest,
        logit,
        pd.DataFrame(),
        years=[2008],
    )

    assert set(summary["model"]) == {
        benchmark.MODEL_RF,
        benchmark.MODEL_LOGIT,
    }
    assert (summary["n_training"] == len(training_frame)).all()
    assert set(results["metrics_by_year"]["model"]) == set(summary["model"])
    assert (results["metrics_by_year"]["n"] == len(validation)).all()
    assert results["mcnemar_pooled"]["n"].item() == len(validation)
