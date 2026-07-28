import numpy as np
import pandas as pd
import pytest

from hmda_seconds import config, diagnostics


def _classified_frame():
    # year 1990: no actual lien_status (pre-2004); year 2005: has it.
    rng = np.random.default_rng(0)
    n = 300
    year = np.where(np.arange(n) < 150, 1990, 2005)
    predicted = rng.integers(1, 3, size=n)
    actual = np.where(year == 2005, predicted, np.nan)
    log_lti = rng.normal(0.5, 0.3, size=n)

    return pd.DataFrame(
        {
            "year": year,
            "log_lti": log_lti,
            config.LABEL_VAR: actual,
            config.PREDICTED_LABEL_VAR: predicted,
        }
    )


def test_build_cells_includes_actual_series_only_for_labeled_years():
    df = _classified_frame()
    cells = diagnostics.build_cells(df, years=[1990, 2005])

    series_1990 = set(cells.loc[cells["year"] == 1990, "series"].unique())
    series_2005 = set(cells.loc[cells["year"] == 2005, "series"].unique())

    assert series_1990 == {"raw", "predicted_first", "predicted_second"}
    assert series_2005 == set(diagnostics.SERIES)


def test_build_cells_skips_years_with_no_rows():
    df = _classified_frame()
    cells = diagnostics.build_cells(df, years=[1990, 2005, 1991])
    assert 1991 not in set(cells["year"].unique())


def test_build_cells_bin_counts_sum_to_series_row_count():
    df = _classified_frame()
    cells = diagnostics.build_cells(df, years=[2005])

    n_predicted_first = (
        (df["year"] == 2005) & (df[config.PREDICTED_LABEL_VAR] == config.FIRST_LIEN_CLASS)
    ).sum()
    cell_total = cells.loc[
        (cells["year"] == 2005) & (cells["series"] == "predicted_first"), "count"
    ].sum()
    assert cell_total == n_predicted_first


def test_render_histogram_writes_a_file(tmp_path):
    df = _classified_frame()
    cells = diagnostics.build_cells(df, years=[2005])

    output_path = tmp_path / "test.pdf"
    diagnostics.render_histogram(cells, 2005, "raw", output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_render_histogram_raises_for_missing_series(tmp_path):
    df = _classified_frame()
    cells = diagnostics.build_cells(df, years=[2005])

    with pytest.raises(ValueError, match="No cells"):
        diagnostics.render_histogram(cells, 2005, "not_a_real_series", tmp_path / "x.pdf")


def test_render_histogram_raises_for_year_with_no_actual_data(tmp_path):
    df = _classified_frame()
    cells = diagnostics.build_cells(df, years=[1990])

    with pytest.raises(ValueError, match="No cells"):
        diagnostics.render_histogram(cells, 1990, "actual_first", tmp_path / "x.pdf")


def test_render_all_writes_one_file_per_year_series_combo(tmp_path):
    df = _classified_frame()
    cells = diagnostics.build_cells(df, years=[1990, 2005])

    paths = diagnostics.render_all(cells, tmp_path)

    expected_count = len(cells[["year", "series"]].drop_duplicates())
    assert len(paths) == expected_count
    assert all(p.exists() for p in paths)
    assert (tmp_path / "hmda_lti_raw_1990.pdf").exists()
    assert (tmp_path / "hmda_lti_actual_first_2005.pdf").exists()
