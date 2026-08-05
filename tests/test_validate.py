import numpy as np
import pandas as pd
import pytest

from hmda_seconds import config, validate


def test_classification_metrics_reports_accuracy_and_confusion_counts():
    y_true = np.array([1, 1, 1, 2, 2])
    y_pred = np.array([1, 1, 2, 2, 1])

    metrics = validate.classification_metrics(y_true, y_pred)

    assert metrics["n"] == 5
    assert metrics["accuracy"] == pytest.approx(3 / 5)
    assert metrics["true_negative"] == 2  # actual 1, predicted 1
    assert metrics["false_positive"] == 1  # actual 1, predicted 2
    assert metrics["false_negative"] == 1  # actual 2, predicted 1
    assert metrics["true_positive"] == 1  # actual 2, predicted 2
    assert "roc_auc" not in metrics


def test_classification_metrics_includes_roc_auc_when_probabilities_given():
    y_true = np.array([1, 1, 2, 2])
    y_pred = np.array([1, 1, 2, 2])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = validate.classification_metrics(y_true, y_pred, y_prob=y_prob)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["average_precision"] == pytest.approx(1.0)
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert metrics["log_loss"] > 0.0
    assert metrics["observed_second_share"] == pytest.approx(0.5)
    assert metrics["mean_predicted_second_share"] == pytest.approx(0.5)


def test_calibration_coefficients_recover_calibrated_probabilities():
    probability = np.linspace(0.02, 0.98, 49)
    counts = 100 * np.ones_like(probability, dtype=int)
    successes = np.rint(probability * counts).astype(int)
    y = np.concatenate(
        [
            np.r_[np.ones(success), np.zeros(count - success)]
            for success, count in zip(successes, counts, strict=True)
        ]
    )
    p = np.repeat(probability, counts)

    intercept, slope = validate.calibration_coefficients(y, p)

    assert intercept == pytest.approx(0.0, abs=0.02)
    assert slope == pytest.approx(1.0, abs=0.02)


def test_calibration_coefficients_handle_large_prevalence_shift():
    rng = np.random.default_rng(17)
    probability = np.linspace(1e-5, 0.95, 20_000)
    score = np.log(probability / (1.0 - probability))
    shifted = 1.5 + 1.1 * score
    outcome_probability = 1.0 / (1.0 + np.exp(-shifted))
    y = rng.random(len(probability)) < outcome_probability

    intercept, slope = validate.calibration_coefficients(y, probability)

    assert intercept == pytest.approx(1.5, abs=0.06)
    assert slope == pytest.approx(1.1, abs=0.04)


def test_evaluate_by_year_skips_years_with_no_labels():
    df = pd.DataFrame(
        {
            "year": [2004, 2004, 1995, 1995],
            config.LABEL_VAR: [1, 2, np.nan, np.nan],
        }
    )
    y_pred = np.array([1, 2, 1, 1])

    out = validate.evaluate_by_year(df, y_pred, years=[1995, 2004])

    assert list(out.index) == [2004]


def test_out_of_time_metrics_reads_wrapper_prediction_columns():
    df = pd.DataFrame(
        {
            "year": [2008, 2008, 2009, 2009],
            config.LABEL_VAR: [1, 2, 1, 2],
            config.PREDICTED_LABEL_VAR: [1, 2, 1, 1],
            config.PROB_SECOND_LIEN_VAR: [0.1, 0.9, 0.2, 0.4],
        }
    )

    out = validate.out_of_time_metrics(df, years=[2008, 2009])

    assert list(out.index) == [2008, 2009]
    assert out.loc[2008, "accuracy"] == pytest.approx(1.0)
    assert out.loc[2009, "accuracy"] == pytest.approx(0.5)


def test_mcnemar_test_reports_discordant_counts_and_accuracies():
    y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    y_pred_a = np.array([1, 1, 1, 1, 1, 1, 1, 1, 2, 2])  # 8/10 correct
    y_pred_b = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2])  # 6/10 correct

    result = validate.mcnemar_test(y_true, y_pred_a, y_pred_b)

    assert result["n"] == 10
    assert result["accuracy_a"] == pytest.approx(0.8)
    assert result["accuracy_b"] == pytest.approx(0.6)
    # a is right and b is wrong on indices 6, 7; b is right and a is wrong nowhere
    assert result["n_a_only_correct"] == 2
    assert result["n_b_only_correct"] == 0
    assert 0.0 <= result["p_value"] <= 1.0


def test_mcnemar_test_p_value_is_high_when_disagreement_is_balanced():
    rng = np.random.default_rng(0)
    y_true = rng.integers(1, 3, size=200)
    y_pred_a = y_true.copy()
    y_pred_b = y_true.copy()
    # a is wrong (only) on 5 cases, b is wrong (only) on a disjoint set of 5
    # cases -- equally balanced disagreement, no systematic difference in
    # accuracy between the two models.
    y_pred_a[:5] = 3 - y_pred_a[:5]
    y_pred_b[5:10] = 3 - y_pred_b[5:10]

    result = validate.mcnemar_test(y_true, y_pred_a, y_pred_b)
    assert result["n_a_only_correct"] == 5
    assert result["n_b_only_correct"] == 5
    assert result["p_value"] > 0.5  # not significant -- disagreement is symmetric


def test_compare_classifiers_by_year_skips_years_with_no_labels():
    df = pd.DataFrame(
        {
            "year": [2008, 2008, 1995, 1995],
            config.LABEL_VAR: [1, 2, np.nan, np.nan],
        }
    )
    pred_a = np.array([1, 2, 1, 1])
    pred_b = np.array([1, 1, 1, 1])

    out = validate.compare_classifiers_by_year(df, pred_a, pred_b, years=[1995, 2008])

    assert list(out.index) == [2008]
    assert out.loc[2008, "accuracy_a"] == pytest.approx(1.0)
    assert out.loc[2008, "accuracy_b"] == pytest.approx(0.5)


def test_continuity_check_reports_nan_actual_share_for_unlabeled_years():
    df = pd.DataFrame(
        {
            "year": [1995, 1995, 2005, 2005],
            config.LABEL_VAR: [np.nan, np.nan, 1, 2],
            config.PREDICTED_LABEL_VAR: [1, 2, 1, 2],
        }
    )

    out = validate.continuity_check(df)

    assert out.loc[1995, "predicted_second_lien_share"] == pytest.approx(0.5)
    assert np.isnan(out.loc[1995, "actual_second_lien_share"])
    assert out.loc[2005, "predicted_second_lien_share"] == pytest.approx(0.5)
    assert out.loc[2005, "actual_second_lien_share"] == pytest.approx(0.5)


def test_log_lti_threshold_baseline_uses_midpoint_of_class_means():
    df = pd.DataFrame(
        {
            config.LABEL_VAR: [1, 1, 2, 2],
            "log_lti": [1.0, 1.0, -1.0, -1.0],
        }
    )
    baseline = validate.fit_log_lti_threshold_baseline(df)
    assert baseline.threshold == pytest.approx(0.0)

    pred = baseline.predict(df)
    assert list(pred) == [1, 1, 2, 2]


def test_oob_score_is_a_valid_probability(training_frame):
    score = validate.oob_score(training_frame, n_estimators=100, max_depth=3, random_state=0)
    assert 0.0 <= score <= 1.0


def test_feature_ablation_covers_every_feature_plus_baseline(training_frame):
    out = validate.feature_ablation(training_frame, n_estimators=5, max_depth=2, random_state=0)

    all_vars = config.CONTINUOUS_VARS + config.CATEGORY_VARS
    assert set(out["dropped_feature"]) == {"(none)", *all_vars}
    assert out["err_rate"].between(0.0, 1.0).all()


def test_hyperparameter_robustness_returns_one_row_per_grid_point(training_frame):
    grid = [
        {"n_estimators": 5, "max_depth": 2},
        {"n_estimators": 10, "max_depth": 3},
    ]
    out = validate.hyperparameter_robustness(training_frame, grid)

    assert len(out) == 2
    assert set(out["n_estimators"]) == {5, 10}
    assert out["err_rate"].between(0.0, 1.0).all()
