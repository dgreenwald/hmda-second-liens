"""Apply the fitted Random Forest to every year in APPLY_YEARS.

Ports the 'classify' + 'combine' stages of the original run_list into one
pass that writes a single combined parquet, and adds predicted class
probabilities the original script never computed (RandomForestWrapper only
exposes hard-label predict(); probabilities come from the underlying
sklearn classifier's predict_proba).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from py_tools.econometrics.machine_learning import RandomForestWrapper

from . import clean, config


def classify_frame(df: pd.DataFrame, rfw: RandomForestWrapper) -> pd.DataFrame:
    """Add predicted lien status and P(second lien) columns to a cleaned frame."""
    rfw.set_data(df)
    rfw.set_labels_features(
        config.LABEL_VAR, config.CONTINUOUS_VARS, config.CATEGORY_VARS, features_only=True
    )
    rfw.predict()

    df = df.copy()
    df[config.PREDICTED_LABEL_VAR] = rfw.predictions

    classes = list(rfw.rf.classes_)
    proba = rfw.rf.predict_proba(rfw.features)
    df[config.PROB_SECOND_LIEN_VAR] = proba[:, classes.index(config.SECOND_LIEN_CLASS)]

    return df


def classify_all_years(
    model_file: str | Path | None = None,
    years=None,
    yearly_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Clean and classify every year in ``years``; return one combined frame."""
    if model_file is None:
        model_file = config.MODEL_FILE
    if years is None:
        years = config.APPLY_YEARS

    rfw = RandomForestWrapper(infile=str(model_file))

    df_fhfa = clean.load_fhfa_county_hpi()
    df_fhfa_balanced = clean.build_balanced_fhfa_panel(df_fhfa, config.APPLY_YEARS)

    df_list = []
    for year in years:
        df_year = clean.load_and_clean_year(year, df_fhfa_balanced, yearly_dir=yearly_dir)
        df_list.append(classify_frame(df_year, rfw))

    return pd.concat(df_list, axis=0, ignore_index=True)
