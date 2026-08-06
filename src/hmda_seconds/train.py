"""Fit the Random Forest lien-status classifier.

Ports the 'estimate' stage of the original classify_seconds.py run_list,
using py_tools.econometrics.machine_learning as-is rather than
reimplementing a Random Forest wrapper.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from py_tools.econometrics.machine_learning import (
    RandomForestWrapper,
    complete_estimation,
)

from . import clean, config
from .density_ratio import artifacts
from .density_ratio.protocols import ModelConfiguration


def run_training(
    input_file: str | Path = config.TRAIN_PARQUET,
    output_file: str | Path = config.MODEL_FILE,
    plot_file: str | Path = config.FIGURE_DIR / "rf_importances.pdf",
) -> RandomForestWrapper:
    """Load the training extract, fit the legacy forest, and persist outputs."""
    output_file = Path(output_file)
    plot_file = Path(plot_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plot_file.parent.mkdir(parents=True, exist_ok=True)
    return fit(
        pd.read_parquet(input_file),
        outfile=str(output_file),
        plotpath=str(plot_file),
    )


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

    fitted = complete_estimation(
        df,
        label_var,
        continuous_vars,
        category_vars,
        train_size,
        test_size,
        outfile=None,
        evaluate=True,
        plot=plotpath is not None,
        plotpath=plotpath,
        **rf_kwargs,
    )
    if outfile is not None:
        _save_random_forest(fitted, df, outfile, "random_split")
    return fitted


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

    encoded = clean.pin_category_levels(df, category_vars)
    wrapper = RandomForestWrapper(data=encoded, **rf_kwargs)
    wrapper.set_labels_features(label_var, continuous_vars, category_vars)
    wrapper.train_test_split(train_size=None, test_size=None)
    wrapper.fit()
    if outfile is not None:
        _save_random_forest(wrapper, df, outfile, "full_sample")
    return wrapper


def _save_random_forest(
    wrapper: RandomForestWrapper,
    source: pd.DataFrame,
    outfile: str,
    specification: str,
) -> None:
    path = Path(outfile)
    train_years = tuple(int(year) for year in sorted(pd.unique(source["year"])))
    labels = wrapper.train_labels
    n_first = int((labels == config.FIRST_LIEN_CLASS).sum())
    n_second = int((labels == config.SECOND_LIEN_CLASS).sum())
    parameters = wrapper.rf.get_params()
    selected_parameters = {
        name: parameters[name]
        for name in (
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
        )
    }
    model_id = (
        f"legacy_random_forest__{specification}"
        f"__train_{min(train_years)}_{max(train_years)}"
    )
    metadata = artifacts.build_metadata(
        model_id=model_id,
        configuration=ModelConfiguration.from_mapping(
            "legacy_random_forest",
            specification,
            selected_parameters,
            random_seed=parameters["random_state"],
        ),
        train_years=train_years,
        counts=(len(labels), n_first, n_second),
        feature_names=_feature_names(wrapper.continuous_vars, wrapper.category_vars),
        weighting="observed_source_distribution",
        source_prior="observed",
        artifact_path=path,
    )
    artifacts.save_pickle_artifact(wrapper.rf, path, metadata)


def _feature_names(
    continuous_vars: list[str], category_vars: list[str]
) -> tuple[str, ...]:
    names = list(continuous_vars)
    for variable in category_vars:
        names.extend(
            f"{variable}_{level}" for level in config.CATEGORY_LEVELS[variable]
        )
    return tuple(names)
