"""Small, atomic CSV checkpoint operations for resumable table workflows."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd


def read_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV checkpoint, or return an empty frame when it is absent."""
    path = Path(path)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def write_csv(frame: pd.DataFrame, path: str | Path) -> None:
    """Atomically replace a CSV table."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def rows_present(
    frame: pd.DataFrame,
    key: Mapping[str, object],
    *,
    required_non_null: Sequence[str] = (),
) -> bool:
    """Return whether a checkpoint contains a complete row for a logical key."""
    required = [*key, *required_non_null]
    if frame.empty or any(column not in frame for column in required):
        return False
    matching = pd.Series(True, index=frame.index)
    for column, value in key.items():
        matching &= frame[column] == value
    for column in required_non_null:
        matching &= frame[column].notna()
    return bool(matching.any())


def append_rows(
    existing: pd.DataFrame, new: pd.DataFrame, path: str | Path
) -> pd.DataFrame:
    """Append rows in memory and atomically persist the combined table."""
    combined = new.copy() if existing.empty else pd.concat(
        [existing, new], ignore_index=True
    )
    write_csv(combined, path)
    return combined


def replace_rows(
    existing: pd.DataFrame,
    new: pd.DataFrame,
    path: str | Path,
    *,
    key_columns: Sequence[str],
) -> pd.DataFrame:
    """Replace the logical key represented by ``new`` and persist atomically."""
    if new.empty:
        raise ValueError("replacement rows cannot be empty")
    if not key_columns:
        raise ValueError("key_columns cannot be empty")
    missing = [column for column in key_columns if column not in new]
    if missing:
        raise KeyError(f"replacement rows are missing key columns: {missing}")
    unique_keys = new.loc[:, list(key_columns)].drop_duplicates()
    if len(unique_keys) != 1:
        raise ValueError("replacement rows must contain exactly one logical key")

    if not existing.empty:
        missing_existing = [column for column in key_columns if column not in existing]
        if missing_existing:
            raise KeyError(
                f"existing checkpoint is missing key columns: {missing_existing}"
            )
        key = unique_keys.iloc[0]
        matching = pd.Series(True, index=existing.index)
        for column in key_columns:
            matching &= existing[column] == key[column]
        existing = existing.loc[~matching]
    combined = pd.concat([existing, new], ignore_index=True)
    write_csv(combined, path)
    return combined
