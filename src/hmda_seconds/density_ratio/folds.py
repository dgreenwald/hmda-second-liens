"""Authoritative temporal folds shared by every density-ratio family."""

from __future__ import annotations

from collections.abc import Iterable

from .protocols import Direction, TemporalFold


def reverse_folds(
    first_labeled_year: int = 2004,
    last_labeled_year: int = 2016,
    training_window: int = 4,
) -> list[TemporalFold]:
    """Return the frozen triangular backward-validation design."""
    if training_window <= 0:
        raise ValueError("training_window must be positive")
    first_training_year = first_labeled_year + 1
    last_training_year = last_labeled_year - training_window + 1
    if last_training_year < first_training_year:
        raise ValueError("labeled-year range is too short for a reverse fold")
    return [
        temporal_fold(
            train_years=range(start, start + training_window),
            target_years=range(first_labeled_year, start),
            direction="reverse",
        )
        for start in range(first_training_year, last_training_year + 1)
    ]


def forward_fold(
    train_years: Iterable[int], target_years: Iterable[int]
) -> TemporalFold:
    """Return one forward-robustness fold from explicit year sets."""
    return temporal_fold(train_years, target_years, direction="forward")


def temporal_fold(
    train_years: Iterable[int],
    target_years: Iterable[int],
    *,
    direction: Direction | None = None,
) -> TemporalFold:
    """Validate explicit source/target years and construct a stable fold."""
    train_years = tuple(train_years)
    target_years = tuple(target_years)
    if not train_years or not target_years:
        raise ValueError("train_years and target_years cannot be empty")
    if direction is None:
        direction = _infer_direction(train_years, target_years)
    boundary = min(train_years) if direction == "reverse" else max(train_years)
    horizons = tuple(
        boundary - year if direction == "reverse" else year - boundary
        for year in target_years
    )
    fold_id = (
        f"{direction}__train_{min(train_years)}_{max(train_years)}"
        f"__target_{min(target_years)}_{max(target_years)}"
    )
    return TemporalFold(
        fold_id=fold_id,
        train_years=train_years,
        target_years=target_years,
        direction=direction,
        horizons=horizons,
    )


def _infer_direction(
    train_years: tuple[int, ...], target_years: tuple[int, ...]
) -> Direction:
    if max(target_years) < min(train_years):
        return "reverse"
    if min(target_years) > max(train_years):
        return "forward"
    raise ValueError("Cannot infer direction from overlapping or interleaved years")
