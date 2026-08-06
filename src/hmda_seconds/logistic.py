"""Fit and apply the logistic lien-status classifier.

The logistic model uses the same continuous and categorical features as the
Random Forest.  It lives in its own module because out-of-time validation
shows that it is a competitive estimator, not merely a disposable baseline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from py_tools.econometrics.machine_learning import get_labels_features
from sklearn.linear_model import LogisticRegression

from . import clean, config
from .density_ratio import artifacts
from .density_ratio.protocols import ModelConfiguration


def run_training(
    input_file: str | Path = config.TRAIN_PARQUET,
    output_file: str | Path = config.LOGISTIC_MODEL_FILE,
) -> LogisticRegression:
    """Load the training extract, fit the legacy logistic model, and save it."""
    model = fit(pd.read_parquet(input_file))
    save(model, output_file)
    return model


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

    encoded = clean.pin_category_levels(df, category_vars)
    labels, features, feature_names = get_labels_features(
        encoded, label_var, continuous_vars, category_vars
    )
    kwargs = {**config.LOGISTIC_KWARGS, **logistic_kwargs}
    model = LogisticRegression(**kwargs)
    model.fit(features, labels)
    counts = artifacts.training_counts(
        df,
        label_var=label_var,
        first_lien_class=config.FIRST_LIEN_CLASS,
        second_lien_class=config.SECOND_LIEN_CLASS,
    )
    model.hmda_train_years_ = tuple(
        int(year) for year in sorted(pd.unique(df["year"]))
    )
    model.hmda_training_counts_ = counts
    model.hmda_feature_names_ = tuple(str(name) for name in feature_names)
    model.hmda_configuration_ = {
        "continuous_vars": tuple(continuous_vars),
        "category_vars": tuple(category_vars),
        "kwargs": kwargs,
    }
    return model


def predict(
    model: LogisticRegression,
    df: pd.DataFrame,
    continuous_vars: list[str] | None = None,
    category_vars: list[str] | None = None,
) -> np.ndarray:
    """Predict first- versus second-lien labels for a cleaned frame."""
    features = feature_matrix(df, continuous_vars, category_vars)
    return model.predict(features)


def predict_proba_second_lien(
    model: LogisticRegression,
    df: pd.DataFrame,
    continuous_vars: list[str] | None = None,
    category_vars: list[str] | None = None,
) -> np.ndarray:
    """Return each observation's estimated probability of a second lien."""
    features = feature_matrix(df, continuous_vars, category_vars)
    classes = list(model.classes_)
    return model.predict_proba(features)[:, classes.index(config.SECOND_LIEN_CLASS)]


def save(model: LogisticRegression, outfile: str | Path) -> None:
    """Serialize a fitted logistic classifier."""
    outfile = Path(outfile)
    train_years = tuple(model.hmda_train_years_)
    model_id = (
        "legacy_logistic__primitive_features"
        f"__train_{min(train_years)}_{max(train_years)}"
    )
    metadata = artifacts.build_metadata(
        model_id=model_id,
        configuration=ModelConfiguration.from_mapping(
            "legacy_logistic",
            "primitive_features",
            {
                "C": float(model.C),
                "solver": str(model.solver),
            },
            random_seed=model.random_state,
        ),
        train_years=train_years,
        counts=tuple(model.hmda_training_counts_),
        feature_names=tuple(model.hmda_feature_names_),
        weighting="observed_source_distribution",
        source_prior="observed",
        artifact_path=outfile,
    )
    artifacts.save_pickle_artifact(model, outfile, metadata)


def load(infile: str | Path) -> LogisticRegression:
    """Load a classifier written by :func:`save`.

    Pickle files must only be loaded from trusted sources.
    """
    model, metadata = artifacts.load_pickle_artifact(infile, LogisticRegression)
    if metadata is not None:
        train_years = tuple(model.hmda_train_years_)
        expected_id = (
            "legacy_logistic__primitive_features"
            f"__train_{min(train_years)}_{max(train_years)}"
        )
        artifacts.validate_metadata_identity(
            metadata, model_id=expected_id, train_years=train_years
        )
    return model


def feature_matrix(
    df: pd.DataFrame,
    continuous_vars: list[str] | None,
    category_vars: list[str] | None,
) -> np.ndarray:
    """Encode the configured continuous and categorical model features."""
    if continuous_vars is None:
        continuous_vars = config.CONTINUOUS_VARS
    if category_vars is None:
        category_vars = config.CATEGORY_VARS
    encoded = clean.pin_category_levels(df, category_vars)
    _, features, _ = get_labels_features(
        encoded,
        config.LABEL_VAR,
        continuous_vars,
        category_vars,
        features_only=True,
    )
    return features
