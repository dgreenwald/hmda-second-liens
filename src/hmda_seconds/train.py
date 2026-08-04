"""Fit the Random Forest lien-status classifier.

Ports the 'estimate' stage of the original classify_seconds.py run_list,
using py_tools.econometrics.machine_learning as-is rather than
reimplementing a Random Forest wrapper.
"""

from __future__ import annotations

import pandas as pd
from py_tools.econometrics.machine_learning import (
    RandomForestWrapper,
    complete_estimation,
)

from . import config


def fit(
    df: pd.DataFrame,
    label_var: str = config.LABEL_VAR,
    continuous_vars: list[str] | None = None,
    category_vars: list[str] | None = None,
    train_size: float = config.TRAIN_SIZE,
    test_size: float = config.TEST_SIZE,
    outfile: str | None = None,
    plotpath: str | None = None,
    **rf_kwargs,
) -> RandomForestWrapper:
    """Fit the Random Forest on a train/test split and evaluate on the holdout.

    This is the in-sample (random-split) evaluation the original script
    performed. It is a sanity check, not the letter's validation result --
    see validate.py for out-of-time evaluation across 2008-2016.
    """
    if continuous_vars is None:
        continuous_vars = config.CONTINUOUS_VARS
    if category_vars is None:
        category_vars = config.CATEGORY_VARS
    if not rf_kwargs:
        rf_kwargs = config.RF_KWARGS

    return complete_estimation(
        df,
        label_var,
        continuous_vars,
        category_vars,
        train_size,
        test_size,
        outfile=outfile,
        evaluate=True,
        plot=plotpath is not None,
        plotpath=plotpath,
        **rf_kwargs,
    )


def fit_full(
    df: pd.DataFrame,
    label_var: str = config.LABEL_VAR,
    continuous_vars: list[str] | None = None,
    category_vars: list[str] | None = None,
    outfile: str | None = None,
    **rf_kwargs,
) -> RandomForestWrapper:
    """Fit a Random Forest on every row, for fair temporal comparison."""
    if continuous_vars is None:
        continuous_vars = config.CONTINUOUS_VARS
    if category_vars is None:
        category_vars = config.CATEGORY_VARS
    if not rf_kwargs:
        rf_kwargs = config.RF_KWARGS

    encoded = _pin_category_levels(df, category_vars)
    wrapper = RandomForestWrapper(data=encoded, **rf_kwargs)
    wrapper.set_labels_features(label_var, continuous_vars, category_vars)
    wrapper.train_test_split(train_size=None, test_size=None)
    wrapper.fit()
    if outfile is not None:
        wrapper.save(outfile)
    return wrapper


def _pin_category_levels(
    df: pd.DataFrame, category_vars: list[str]
) -> pd.DataFrame:
    encoded = df.copy(deep=False)
    for var in category_vars:
        if var in config.CATEGORY_LEVELS:
            encoded[var] = pd.Categorical(
                df[var], categories=config.CATEGORY_LEVELS[var]
            )
    return encoded
