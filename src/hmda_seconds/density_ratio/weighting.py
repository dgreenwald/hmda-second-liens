"""Shared source-prior weighting conventions for density-ratio families."""

from __future__ import annotations

import numpy as np
import pandas as pd


def equal_source_prior_weights(
    training: pd.DataFrame, y_second: np.ndarray
) -> np.ndarray:
    """Give both lien classes half of each source year's total weight."""
    weights = np.empty(len(training), dtype=float)
    years = training["year"].to_numpy()
    for year in np.unique(years):
        in_year = years == year
        share = float(y_second[in_year].mean())
        if not 0 < share < 1:
            raise ValueError(f"Source year {year} must contain both lien classes")
        weights[in_year & y_second] = 0.5 / share
        weights[in_year & ~y_second] = 0.5 / (1.0 - share)
    return weights
