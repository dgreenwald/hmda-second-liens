"""Fit and apply the logistic lien-status classifier.

The logistic model uses the same continuous and categorical features as the
Random Forest.  It lives in its own module because out-of-time validation
shows that it is a competitive estimator, not merely a disposable baseline.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from py_tools.econometrics.machine_learning import get_labels_features
from sklearn.linear_model import LogisticRegression

from . import config


def fit(
    df: pd.DataFrame,
    label_var: str = config.LABEL_VAR,
    continuous_vars: list[str] | None = None,
    category_vars: list[str] | None = None,
    **logistic_kwargs,
) -> LogisticRegression:
    """Fit a logistic classifier on all rows of a labeled frame.

    Keyword arguments override ``config.LOGISTIC_KWARGS``.  Using the full
    2004--2007 extract is intentional: model selection is evaluated on the
    separate 2008--2016 out-of-time sample rather than a random split.
    """
    if continuous_vars is None:
        continuous_vars = config.CONTINUOUS_VARS
    if category_vars is None:
        category_vars = config.CATEGORY_VARS

    encoded = _pin_category_levels(df, category_vars)
    labels, features, _ = get_labels_features(
        encoded, label_var, continuous_vars, category_vars
    )
    kwargs = {**config.LOGISTIC_KWARGS, **logistic_kwargs}
    model = LogisticRegression(**kwargs)
    model.fit(features, labels)
    return model


def predict(
    model: LogisticRegression,
    df: pd.DataFrame,
    continuous_vars: list[str] | None = None,
    category_vars: list[str] | None = None,
) -> np.ndarray:
    """Predict first- versus second-lien labels for a cleaned frame."""
    features = _feature_matrix(df, continuous_vars, category_vars)
    return model.predict(features)


def predict_proba_second_lien(
    model: LogisticRegression,
    df: pd.DataFrame,
    continuous_vars: list[str] | None = None,
    category_vars: list[str] | None = None,
) -> np.ndarray:
    """Return each observation's estimated probability of a second lien."""
    features = _feature_matrix(df, continuous_vars, category_vars)
    classes = list(model.classes_)
    return model.predict_proba(features)[:, classes.index(config.SECOND_LIEN_CLASS)]


def save(model: LogisticRegression, outfile: str | Path) -> None:
    """Serialize a fitted logistic classifier."""
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with outfile.open("wb") as file:
        pickle.dump(model, file)


def load(infile: str | Path) -> LogisticRegression:
    """Load a classifier written by :func:`save`.

    Pickle files must only be loaded from trusted sources.
    """
    with Path(infile).open("rb") as file:
        model = pickle.load(file)
    if not isinstance(model, LogisticRegression):
        raise TypeError(f"Expected LogisticRegression, got {type(model).__name__}")
    return model


def _feature_matrix(
    df: pd.DataFrame,
    continuous_vars: list[str] | None,
    category_vars: list[str] | None,
) -> np.ndarray:
    if continuous_vars is None:
        continuous_vars = config.CONTINUOUS_VARS
    if category_vars is None:
        category_vars = config.CATEGORY_VARS
    encoded = _pin_category_levels(df, category_vars)
    _, features, _ = get_labels_features(
        encoded,
        config.LABEL_VAR,
        continuous_vars,
        category_vars,
        features_only=True,
    )
    return features


def _pin_category_levels(
    df: pd.DataFrame, category_vars: list[str]
) -> pd.DataFrame:
    """Ensure encoded batches have the same dummy columns as training data.

    Persisted parquet files and caller-created subsets do not necessarily
    retain pandas categorical metadata. Reapply the canonical levels here so
    prediction remains safe even when a batch contains only some categories.
    The shallow copy shares all unmodified columns with the input frame.
    """
    encoded = df.copy(deep=False)
    for var in category_vars:
        if var in config.CATEGORY_LEVELS:
            encoded[var] = pd.Categorical(
                df[var], categories=config.CATEGORY_LEVELS[var]
            )
    return encoded
