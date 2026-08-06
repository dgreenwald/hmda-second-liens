import pandas as pd
import pytest

from hmda_seconds.density_ratio import checkpoints


def test_read_csv_returns_empty_frame_for_missing_file(tmp_path):
    assert checkpoints.read_csv(tmp_path / "missing.csv").empty


def test_append_rows_preserves_existing_order_and_writes(tmp_path):
    path = tmp_path / "cells.csv"
    existing = pd.DataFrame({"year": [2003], "value": [1.0]})
    new = pd.DataFrame({"year": [2004], "value": [2.0]})

    combined = checkpoints.append_rows(existing, new, path)

    assert combined["year"].tolist() == [2003, 2004]
    pd.testing.assert_frame_equal(pd.read_csv(path), combined)


def test_rows_present_supports_composite_keys_and_required_values():
    frame = pd.DataFrame(
        {
            "train_start": [2005, 2005],
            "validation_year": [2003, 2004],
            "score": [1.0, float("nan")],
        }
    )

    assert checkpoints.rows_present(
        frame, {"train_start": 2005, "validation_year": 2003}
    )
    assert not checkpoints.rows_present(
        frame,
        {"train_start": 2005, "validation_year": 2004},
        required_non_null=("score",),
    )


def test_replace_rows_replaces_one_multicolumn_key(tmp_path):
    path = tmp_path / "cells.csv"
    existing = pd.DataFrame(
        {
            "train_start": [2005, 2005, 2006],
            "validation_year": [2004, 2003, 2004],
            "bin": [1, 1, 1],
            "value": [1.0, 2.0, 3.0],
        }
    )
    new = pd.DataFrame(
        {
            "train_start": [2005, 2005],
            "validation_year": [2004, 2004],
            "bin": [1, 2],
            "value": [4.0, 5.0],
        }
    )

    combined = checkpoints.replace_rows(
        existing,
        new,
        path,
        key_columns=("train_start", "validation_year"),
    )

    assert combined["value"].tolist() == [2.0, 3.0, 4.0, 5.0]
    pd.testing.assert_frame_equal(pd.read_csv(path), combined)


def test_replace_rows_rejects_multiple_logical_keys(tmp_path):
    new = pd.DataFrame({"year": [2003, 2004], "value": [1.0, 2.0]})

    with pytest.raises(ValueError, match="exactly one logical key"):
        checkpoints.replace_rows(
            pd.DataFrame(), new, tmp_path / "cells.csv", key_columns=("year",)
        )
