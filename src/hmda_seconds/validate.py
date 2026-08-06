"""Out-of-time validation, baselines, and richer metrics for the RF classifier.

This is the letter's actual validation section (MIGRATION_PLAN.md,
"Methodological improvements" items 1-6). The original script only
spot-checked a single year (2006) against a random in-sample split. Here
every labeled year the model was NOT trained on (2008-2016) is evaluated
against its predictions, alongside a logistic comparator and a simple
threshold baseline, richer per-class metrics, an out-of-bag check, a feature
ablation, and a small hyperparameter grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from py_tools.econometrics.machine_learning import get_labels_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from statsmodels.stats.contingency_tables import mcnemar

from . import config
from .density_ratio import artifacts
from .density_ratio.protocols import ModelConfiguration

CLASS_LABELS = [config.FIRST_LIEN_CLASS, config.SECOND_LIEN_CLASS]


# --------------------------------------------------------------------------
# Shared metrics primitive
# --------------------------------------------------------------------------


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None
) -> dict:
    """Accuracy, per-class precision/recall/F1, confusion counts, and (if
    probabilities are given) ranking, scoring-rule, and calibration metrics."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_LABELS, zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS).ravel()

    metrics = {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_first": precision[0],
        "recall_first": recall[0],
        "f1_first": f1[0],
        "precision_second": precision[1],
        "recall_second": recall[1],
        "f1_second": f1[1],
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }
    if y_prob is not None:
        y_second = np.asarray(y_true) == config.SECOND_LIEN_CLASS
        y_prob = np.asarray(y_prob, dtype=float)
        calibration_intercept, calibration_slope = calibration_coefficients(
            y_second, y_prob
        )
        metrics.update(
            {
                "roc_auc": roc_auc_score(y_second, y_prob),
                "average_precision": average_precision_score(y_second, y_prob),
                "log_loss": log_loss(y_second, y_prob, labels=[False, True]),
                "brier_score": brier_score_loss(y_second, y_prob),
                "observed_second_share": y_second.mean(),
                "mean_predicted_second_share": y_prob.mean(),
                "calibration_mean_error": y_prob.mean() - y_second.mean(),
                "calibration_intercept": calibration_intercept,
                "calibration_slope": calibration_slope,
            }
        )
    return metrics


def calibration_coefficients(
    y_true_second: np.ndarray,
    y_prob: np.ndarray,
    tolerance: float = 1e-10,
    max_iter: int = 100,
) -> tuple[float, float]:
    """Fit the calibration model ``y ~ intercept + slope * logit(p)``.

    The two-parameter Newton iteration avoids constructing a general-purpose
    regression object for validation samples containing millions of loans.
    """
    y = np.asarray(y_true_second, dtype=float)
    probability = np.clip(np.asarray(y_prob, dtype=float), 1e-12, 1 - 1e-12)
    score = np.log(probability / (1.0 - probability))
    prevalence = np.clip(y.mean(), 1e-12, 1 - 1e-12)
    coefficients = np.array([np.log(prevalence / (1.0 - prevalence)), 0.0])

    def objective(candidate: np.ndarray) -> float:
        linear = candidate[0] + candidate[1] * score
        return float(np.sum(y * linear - np.logaddexp(0.0, linear)))

    for _ in range(max_iter):
        linear = coefficients[0] + coefficients[1] * score
        fitted = np.empty_like(linear)
        positive = linear >= 0
        fitted[positive] = 1.0 / (1.0 + np.exp(-linear[positive]))
        exp_linear = np.exp(linear[~positive])
        fitted[~positive] = exp_linear / (1.0 + exp_linear)
        weight = fitted * (1.0 - fitted)
        gradient = np.array(
            [(y - fitted).sum(), np.dot(score, y - fitted)]
        )
        information = np.array(
            [
                [weight.sum(), np.dot(weight, score)],
                [np.dot(weight, score), np.dot(weight, score * score)],
            ]
        )
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError:
            return np.nan, np.nan
        current_objective = objective(coefficients)
        step_scale = 1.0
        while step_scale >= 2.0**-20:
            candidate = coefficients + step_scale * step
            if objective(candidate) >= current_objective - 1e-8:
                break
            step_scale /= 2.0
        else:
            return np.nan, np.nan
        accepted_step = step_scale * step
        coefficients = candidate
        if np.max(np.abs(accepted_step)) < tolerance:
            break

    return float(coefficients[0]), float(coefficients[1])


def evaluate_by_year(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    years=None,
    y_prob: np.ndarray | None = None,
    label_var: str = config.LABEL_VAR,
) -> pd.DataFrame:
    """classification_metrics per year, given predictions already row-aligned to df.

    Years with no non-null label_var (nothing to validate against) are
    skipped rather than reported as an empty/degenerate row.
    """
    if years is None:
        years = config.VALIDATE_YEARS

    work = df[["year", label_var]].copy()
    work["_pred"] = np.asarray(y_pred)
    if y_prob is not None:
        work["_prob"] = np.asarray(y_prob)

    rows = []
    for year in years:
        sub = work.loc[work["year"] == year].dropna(subset=[label_var])
        if sub.empty:
            continue
        metrics = classification_metrics(
            sub[label_var].to_numpy(),
            sub["_pred"].to_numpy(),
            sub["_prob"].to_numpy() if y_prob is not None else None,
        )
        metrics["year"] = year
        rows.append(metrics)

    return pd.DataFrame(rows).set_index("year")


# --------------------------------------------------------------------------
# Out-of-time validation (item 1) and continuity check (item 2)
# --------------------------------------------------------------------------


def out_of_time_metrics(df: pd.DataFrame, years=None) -> pd.DataFrame:
    """evaluate_by_year using the RF's own prediction/probability columns.

    Expects df to already contain PREDICTED_LABEL_VAR and PROB_SECOND_LIEN_VAR
    (e.g. classify.py's combined output).
    """
    return evaluate_by_year(
        df,
        df[config.PREDICTED_LABEL_VAR].to_numpy(),
        years=years,
        y_prob=df[config.PROB_SECOND_LIEN_VAR].to_numpy(),
    )


def continuity_check(df: pd.DataFrame) -> pd.DataFrame:
    """Predicted vs. actual second-lien share by year (the 2004-boundary check).

    Actual share is NaN for years without any non-null lien_status (i.e.
    pre-2004), rather than silently treated as zero.
    """
    gb = df.groupby("year")
    predicted_share = gb[config.PREDICTED_LABEL_VAR].apply(
        lambda s: (s == config.SECOND_LIEN_CLASS).mean()
    )
    actual_share = gb[config.LABEL_VAR].apply(
        lambda s: (s == config.SECOND_LIEN_CLASS).mean() if s.notna().any() else np.nan
    )
    return pd.DataFrame(
        {
            "predicted_second_lien_share": predicted_share,
            "actual_second_lien_share": actual_share,
        }
    )


# --------------------------------------------------------------------------
# Simple threshold baseline (item 3)
# --------------------------------------------------------------------------


@dataclass
class LogLtiThresholdBaseline:
    """The simplest possible baseline: a single threshold on log_lti.

    Second liens/piggybacks carry much smaller balances relative to income
    than first liens (Appendix sec:hmda-data), so a threshold on log_lti
    alone should already separate the classes reasonably well. The threshold
    is the midpoint of the two classes' training-sample mean log_lti, not
    tuned against a held-out set.
    """

    threshold: float

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.where(
            df["log_lti"] >= self.threshold,
            config.FIRST_LIEN_CLASS,
            config.SECOND_LIEN_CLASS,
        )


def fit_log_lti_threshold_baseline(
    df: pd.DataFrame, model_file: str | Path | None = None
) -> LogLtiThresholdBaseline:
    means = df.groupby(config.LABEL_VAR)["log_lti"].mean()
    fitted = LogLtiThresholdBaseline(threshold=float(means.mean()))
    if model_file is not None:
        _save_validation_model(
            fitted,
            df,
            df[config.LABEL_VAR].to_numpy(),
            ("log_lti",),
            "log_lti_threshold",
            {"threshold": fitted.threshold},
            model_file,
        )
    return fitted


# --------------------------------------------------------------------------
# Formal RF-vs-baseline comparison: McNemar's test
# --------------------------------------------------------------------------


def mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> dict:
    """McNemar's test comparing two classifiers' correctness on the same sample.

    Uses only the discordant pairs (cases where exactly one model is
    correct) to test whether model_a and model_b differ systematically in
    accuracy, rather than just reporting each model's accuracy separately
    and eyeballing the difference. With the sample sizes here (millions of
    loans per year), the chi-squared approximation with continuity
    correction is used rather than the exact binomial test, and even a
    tiny, substantively unimportant accuracy difference will usually be
    "significant" -- report n_a_only_correct/n_b_only_correct (or the
    accuracy difference) alongside the p-value, not the p-value alone.
    """
    correct_a = np.asarray(y_pred_a) == np.asarray(y_true)
    correct_b = np.asarray(y_pred_b) == np.asarray(y_true)

    n_both_correct = int(np.sum(correct_a & correct_b))
    n_a_only = int(np.sum(correct_a & ~correct_b))  # a right, b wrong
    n_b_only = int(np.sum(~correct_a & correct_b))  # b right, a wrong
    n_both_wrong = int(np.sum(~correct_a & ~correct_b))

    table = [[n_both_correct, n_a_only], [n_b_only, n_both_wrong]]
    result = mcnemar(table, exact=False, correction=True)

    n = len(y_true)
    return {
        "n": n,
        "accuracy_a": (n_both_correct + n_a_only) / n,
        "accuracy_b": (n_both_correct + n_b_only) / n,
        "n_a_only_correct": n_a_only,
        "n_b_only_correct": n_b_only,
        "statistic": result.statistic,
        "p_value": result.pvalue,
    }


def compare_classifiers_by_year(
    df: pd.DataFrame,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    years=None,
    label_var: str = config.LABEL_VAR,
) -> pd.DataFrame:
    """mcnemar_test per year, given two sets of predictions row-aligned to df."""
    if years is None:
        years = config.VALIDATE_YEARS

    work = df[["year", label_var]].copy()
    work["_pred_a"] = np.asarray(y_pred_a)
    work["_pred_b"] = np.asarray(y_pred_b)

    rows = []
    for year in years:
        sub = work.loc[work["year"] == year].dropna(subset=[label_var])
        if sub.empty:
            continue
        result = mcnemar_test(
            sub[label_var].to_numpy(), sub["_pred_a"].to_numpy(), sub["_pred_b"].to_numpy()
        )
        result["year"] = year
        rows.append(result)

    return pd.DataFrame(rows).set_index("year")


# --------------------------------------------------------------------------
# Out-of-bag score (item 4)
# --------------------------------------------------------------------------


def oob_score(
    df: pd.DataFrame,
    train_size: float = config.TRAIN_SIZE,
    test_size: float = config.TEST_SIZE,
    random_state: int = 17,
    model_dir: str | Path | None = None,
    **rf_kwargs,
) -> float:
    """Fit an RF with oob_score=True on the same train split as the headline
    model, as a cheap cross-check against its held-out error rate.

    Deliberately uses the same train_size/test_size/random_state split as
    train.fit() rather than the full frame: OOB is itself an unbiased
    generalization estimate computed from each tree's bootstrap sample, so
    the point is to cross-check the *deployed* model's held-out error rate
    against a second, independent estimate -- not to fit a different, much
    larger model that takes several times as long for a number that doesn't
    describe what's actually shipped.
    """
    if not rf_kwargs:
        rf_kwargs = config.RF_KWARGS
    labels, features, feature_names = get_labels_features(
        df, config.LABEL_VAR, config.CONTINUOUS_VARS, config.CATEGORY_VARS
    )
    train_features, _, train_labels, _ = train_test_split(
        features, labels, train_size=train_size, test_size=test_size, random_state=random_state
    )
    rf = RandomForestClassifier(oob_score=True, **rf_kwargs)
    rf.fit(train_features, train_labels)
    if model_dir is not None:
        _save_validation_model(
            rf,
            df,
            train_labels,
            tuple(str(name) for name in feature_names),
            "oob",
            _artifact_rf_parameters(rf),
            Path(model_dir) / "random_forest__oob.pkl",
        )
    return rf.oob_score_


# --------------------------------------------------------------------------
# Feature ablation (item 5) and hyperparameter robustness (item 6)
# --------------------------------------------------------------------------


def feature_ablation(
    df: pd.DataFrame,
    train_size: float = config.TRAIN_SIZE,
    test_size: float = config.TEST_SIZE,
    random_state: int = 17,
    model_dir: str | Path | None = None,
    **rf_kwargs,
) -> pd.DataFrame:
    """Held-out error rate with each feature dropped in turn, vs. the full set.

    All rows share one train/test split (fixed random_state) so error-rate
    differences reflect the dropped feature rather than split noise.
    """
    if not rf_kwargs:
        rf_kwargs = config.RF_KWARGS

    all_vars = config.CONTINUOUS_VARS + config.CATEGORY_VARS
    rows = []
    for dropped in [None, *all_vars]:
        continuous_vars = [v for v in config.CONTINUOUS_VARS if v != dropped]
        category_vars = [v for v in config.CATEGORY_VARS if v != dropped]
        labels, features, feature_names = get_labels_features(
            df, config.LABEL_VAR, continuous_vars, category_vars
        )
        train_features, test_features, train_labels, test_labels = train_test_split(
            features,
            labels,
            train_size=train_size,
            test_size=test_size,
            random_state=random_state,
        )
        rf = RandomForestClassifier(**rf_kwargs)
        rf.fit(train_features, train_labels)
        if model_dir is not None:
            label = dropped or "none"
            _save_validation_model(
                rf,
                df,
                train_labels,
                tuple(str(name) for name in feature_names),
                f"ablation_drop_{label}",
                _artifact_rf_parameters(rf),
                Path(model_dir) / f"random_forest__ablation_drop_{label}.pkl",
            )
        err_rate = 1.0 - rf.score(test_features, test_labels)
        rows.append({"dropped_feature": dropped or "(none)", "err_rate": err_rate})

    return pd.DataFrame(rows)


def hyperparameter_robustness(
    df: pd.DataFrame,
    grid: list[dict],
    train_size: float = config.TRAIN_SIZE,
    test_size: float = config.TEST_SIZE,
    random_state: int = 17,
    model_dir: str | Path | None = None,
    **fixed_kwargs,
) -> pd.DataFrame:
    """Held-out error rate for each hyperparameter combination in grid.

    All combinations share one train/test split (fixed random_state) so
    error-rate differences reflect the hyperparameters rather than split
    noise. fixed_kwargs (e.g. n_jobs) apply to every fit but are not swept
    and are not included in the output columns.
    """
    labels, features, feature_names = get_labels_features(
        df, config.LABEL_VAR, config.CONTINUOUS_VARS, config.CATEGORY_VARS
    )
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, train_size=train_size, test_size=test_size, random_state=random_state
    )

    rows = []
    for params in grid:
        rf = RandomForestClassifier(random_state=random_state, **params, **fixed_kwargs)
        rf.fit(train_features, train_labels)
        if model_dir is not None:
            parameter_label = "__".join(
                f"{name}_{str(value).replace('.', 'p')}"
                for name, value in sorted(params.items())
            )
            _save_validation_model(
                rf,
                df,
                train_labels,
                tuple(str(name) for name in feature_names),
                f"hyperparameters_{parameter_label}",
                _artifact_rf_parameters(rf),
                Path(model_dir)
                / f"random_forest__hyperparameters_{parameter_label}.pkl",
            )
        err_rate = 1.0 - rf.score(test_features, test_labels)
        rows.append({**params, "err_rate": err_rate})

    return pd.DataFrame(rows)


def _save_validation_model(
    model: object,
    source: pd.DataFrame,
    fitted_labels: np.ndarray,
    feature_names: tuple[str, ...],
    specification: str,
    parameters: dict,
    model_file: str | Path,
) -> None:
    train_years = tuple(int(year) for year in sorted(pd.unique(source["year"])))
    fitted_labels = np.asarray(fitted_labels)
    counts = (
        len(fitted_labels),
        int(np.sum(fitted_labels == config.FIRST_LIEN_CLASS)),
        int(np.sum(fitted_labels == config.SECOND_LIEN_CLASS)),
    )
    path = Path(model_file)
    model_id = (
        f"legacy_validation__{specification}"
        f"__train_{min(train_years)}_{max(train_years)}"
    )
    metadata = artifacts.build_metadata(
        model_id=model_id,
        configuration=ModelConfiguration.from_mapping(
            "legacy_validation", specification, parameters
        ),
        train_years=train_years,
        counts=counts,
        feature_names=feature_names,
        weighting="observed_source_distribution",
        source_prior="observed",
        artifact_path=path,
    )
    artifacts.save_pickle_artifact(model, path, metadata)


def _artifact_rf_parameters(model: RandomForestClassifier) -> dict:
    parameters = model.get_params()
    return {
        name: parameters[name]
        for name in (
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
            "oob_score",
        )
    }
