"""Shared aggregation rules for reverse-time evaluation cells."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def two_stage_horizon_means(
    cells: pd.DataFrame,
    *,
    candidate_columns: Sequence[str],
    metric_columns: Sequence[str],
    count_column: str,
    expected_horizons: Sequence[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average within horizons, then give every horizon equal weight."""
    if cells.empty:
        raise ValueError("cannot aggregate an empty cell table")
    if not metric_columns:
        raise ValueError("metric_columns cannot be empty")
    required = [*candidate_columns, "horizon", *metric_columns, count_column]
    missing = [column for column in required if column not in cells]
    if missing:
        raise KeyError(f"cell table is missing aggregation columns: {missing}")

    observed_horizons = set(cells["horizon"].unique())
    expected = observed_horizons if expected_horizons is None else set(expected_horizons)
    if not expected:
        raise ValueError("expected horizons cannot be empty")
    if candidate_columns:
        horizon_sets = cells.groupby(list(candidate_columns), dropna=False)[
            "horizon"
        ].agg(lambda values: set(values))
        incomplete = horizon_sets.loc[horizon_sets != expected]
        if not incomplete.empty:
            raise ValueError(
                "candidate horizon coverage is incomplete: "
                f"expected {sorted(expected)}"
            )
    elif observed_horizons != expected:
        raise ValueError(
            f"horizon coverage is incomplete: expected {sorted(expected)}"
        )

    horizon_groups = [*candidate_columns, "horizon"]
    horizons = cells.groupby(horizon_groups, as_index=False).agg(
        **{column: (column, "mean") for column in metric_columns},
        n_cells=(count_column, "size"),
    )
    summary = horizons.groupby(list(candidate_columns), as_index=False).agg(
        **{column: (column, "mean") for column in metric_columns},
        n_horizons=("horizon", "nunique"),
        n_cells=("n_cells", "sum"),
    )
    return horizons, summary
