"""Canonical probability-metric and reliability-bin diagnostic cells."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import evaluation

DEFAULT_N_BINS = 10


@dataclass(frozen=True)
class DiagnosticCell:
    """Common outputs plus optional family-specific diagnostic tables."""

    metrics: pd.DataFrame
    bins: pd.DataFrame
    extensions: dict[str, pd.DataFrame]


def evaluate_cell(
    y_second: np.ndarray,
    probability: np.ndarray,
    *,
    metadata: Mapping[str, object],
    n_bins: int = DEFAULT_N_BINS,
    metrics: Mapping[str, float | int] | None = None,
    additional_metrics: Mapping[str, object] | None = None,
    extension_records: Mapping[str, Mapping[str, object]] | None = None,
) -> DiagnosticCell:
    """Build one canonical metric row, reliability table, and extensions."""
    if metrics is None:
        metrics = evaluation.evaluate_sample(y_second, probability)
    metric_record = evaluation.metric_record_from_metrics(
        metrics, metadata=metadata, additional=additional_metrics
    )
    bins = reliability_bins(y_second, probability, n_bins).assign(**metadata)
    extensions = {
        name: pd.DataFrame([{**metadata, **record}])
        for name, record in (extension_records or {}).items()
    }
    return DiagnosticCell(pd.DataFrame([metric_record]), bins, extensions)


def reliability_bins(
    y_second: np.ndarray,
    probability: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
) -> pd.DataFrame:
    """Build approximate equal-count validation-probability bins."""
    y = np.asarray(y_second, dtype=bool)
    probability = np.asarray(probability, dtype=float)
    evaluation.validate_probability_inputs(y, probability)
    if n_bins < 2:
        raise ValueError("n_bins must be at least two")

    quantiles = np.quantile(probability, np.linspace(0.0, 1.0, n_bins + 1))
    interior = np.unique(quantiles[1:-1])
    assignment = np.searchsorted(interior, probability, side="right")
    effective_bins = len(interior) + 1
    count = np.bincount(assignment, minlength=effective_bins)
    probability_sum = np.bincount(
        assignment, weights=probability, minlength=effective_bins
    )
    observed_sum = np.bincount(
        assignment, weights=y.astype(float), minlength=effective_bins
    )
    minimum = np.full(effective_bins, np.inf)
    maximum = np.full(effective_bins, -np.inf)
    np.minimum.at(minimum, assignment, probability)
    np.maximum.at(maximum, assignment, probability)
    keep = count > 0
    return pd.DataFrame(
        {
            "probability_bin": np.arange(1, effective_bins + 1)[keep],
            "n": count[keep].astype(np.int64),
            "min_probability": minimum[keep],
            "max_probability": maximum[keep],
            "mean_predicted_probability": probability_sum[keep] / count[keep],
            "observed_second_share": observed_sum[keep] / count[keep],
        }
    )
