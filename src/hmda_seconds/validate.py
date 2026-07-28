"""Out-of-time validation, baselines, and richer metrics for the RF classifier.

This is the letter's actual validation section (MIGRATION_PLAN.md,
"Methodological improvements" items 1-6). The original script only
spot-checked a single year (2006) against a random in-sample split. Here
every labeled year the model was NOT trained on (2008-2016) is evaluated
against its predictions, alongside two simple baselines, richer per-class
metrics, an out-of-bag check, a feature ablation, and a small hyperparameter
grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from py_tools.econometrics.machine_learning import get_labels_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from . import config

CLASS_LABELS = [config.FIRST_LIEN_CLASS, config.SECOND_LIEN_CLASS]


# --------------------------------------------------------------------------
# Shared metrics primitive
# --------------------------------------------------------------------------


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None
) -> dict:
    """Accuracy, per-class precision/recall/F1, confusion counts, and (if
    probabilities are given) ROC-AUC for the second-lien class."""
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
        metrics["roc_auc"] = roc_auc_score(y_true == config.SECOND_LIEN_CLASS, y_prob)
    return metrics


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
# Baselines (item 3)
# --------------------------------------------------------------------------


def fit_logistic_baseline(df: pd.DataFrame, **kwargs) -> LogisticRegression:
    """Logistic regression on the same encoded features as the RF."""
    labels, features, _ = get_labels_features(
        df, config.LABEL_VAR, config.CONTINUOUS_VARS, config.CATEGORY_VARS
    )
    model = LogisticRegression(max_iter=1000, **kwargs)
    model.fit(features, labels)
    return model


def predict_logistic_baseline(model: LogisticRegression, df: pd.DataFrame) -> np.ndarray:
    _, features, _ = get_labels_features(
        df, config.LABEL_VAR, config.CONTINUOUS_VARS, config.CATEGORY_VARS, features_only=True
    )
    return model.predict(features)


def predict_proba_logistic_baseline(model: LogisticRegression, df: pd.DataFrame) -> np.ndarray:
    _, features, _ = get_labels_features(
        df, config.LABEL_VAR, config.CONTINUOUS_VARS, config.CATEGORY_VARS, features_only=True
    )
    classes = list(model.classes_)
    return model.predict_proba(features)[:, classes.index(config.SECOND_LIEN_CLASS)]


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


def fit_log_lti_threshold_baseline(df: pd.DataFrame) -> LogLtiThresholdBaseline:
    means = df.groupby(config.LABEL_VAR)["log_lti"].mean()
    return LogLtiThresholdBaseline(threshold=float(means.mean()))


# --------------------------------------------------------------------------
# Out-of-bag score (item 4)
# --------------------------------------------------------------------------


def oob_score(
    df: pd.DataFrame,
    train_size: float = config.TRAIN_SIZE,
    test_size: float = config.TEST_SIZE,
    random_state: int = 17,
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
    labels, features, _ = get_labels_features(
        df, config.LABEL_VAR, config.CONTINUOUS_VARS, config.CATEGORY_VARS
    )
    train_features, _, train_labels, _ = train_test_split(
        features, labels, train_size=train_size, test_size=test_size, random_state=random_state
    )
    rf = RandomForestClassifier(oob_score=True, **rf_kwargs)
    rf.fit(train_features, train_labels)
    return rf.oob_score_


# --------------------------------------------------------------------------
# Feature ablation (item 5) and hyperparameter robustness (item 6)
# --------------------------------------------------------------------------


def feature_ablation(
    df: pd.DataFrame,
    train_size: float = config.TRAIN_SIZE,
    test_size: float = config.TEST_SIZE,
    random_state: int = 17,
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
        labels, features, _ = get_labels_features(
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
        err_rate = 1.0 - rf.score(test_features, test_labels)
        rows.append({"dropped_feature": dropped or "(none)", "err_rate": err_rate})

    return pd.DataFrame(rows)


def hyperparameter_robustness(
    df: pd.DataFrame,
    grid: list[dict],
    train_size: float = config.TRAIN_SIZE,
    test_size: float = config.TEST_SIZE,
    random_state: int = 17,
    **fixed_kwargs,
) -> pd.DataFrame:
    """Held-out error rate for each hyperparameter combination in grid.

    All combinations share one train/test split (fixed random_state) so
    error-rate differences reflect the hyperparameters rather than split
    noise. fixed_kwargs (e.g. n_jobs) apply to every fit but are not swept
    and are not included in the output columns.
    """
    labels, features, _ = get_labels_features(
        df, config.LABEL_VAR, config.CONTINUOUS_VARS, config.CATEGORY_VARS
    )
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, train_size=train_size, test_size=test_size, random_state=random_state
    )

    rows = []
    for params in grid:
        rf = RandomForestClassifier(random_state=random_state, **params, **fixed_kwargs)
        rf.fit(train_features, train_labels)
        err_rate = 1.0 - rf.score(test_features, test_labels)
        rows.append({**params, "err_rate": err_rate})

    return pd.DataFrame(rows)
