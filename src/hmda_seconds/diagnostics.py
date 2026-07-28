"""Binned LTI histogram cells and diagnostic figures.

Adapted from the replication_package_proposal's
code/diagnostics/{build_hmda_lti_cells.py,hmda_lien_diagnostics.py}
templates, generalized to run over the full APPLY_YEARS range (1990-2016)
instead of the five hardcoded years those scripts used -- classify.py
already produces predictions (and, for 2004-2016, actual lien_status) for
every year, so there's no reason to hand-pick a subset here; a paper draft
can select whichever years it wants to embed from the full set this
produces.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

FIGSIZE = (6.0, 4.0)
BAR_ALPHA = 0.6
BIN_WIDTH = 0.2
BIN_COUNT = 40
LTI_MIN = 0.0
LTI_MAX = 8.0

SERIES = ("raw", "actual_first", "actual_second", "predicted_first", "predicted_second")


def _cells_for_series(lti: np.ndarray, year: int, series: str) -> pd.DataFrame:
    values = lti[np.isfinite(lti)]
    counts, _ = np.histogram(values, bins=np.linspace(LTI_MIN, LTI_MAX, BIN_COUNT + 1))
    return pd.DataFrame(
        {
            "year": year,
            "series": series,
            "lti_bin": np.arange(BIN_COUNT, dtype=np.int64),
            "count": counts.astype(np.int64),
        }
    )


def build_cells(df: pd.DataFrame, years=None) -> pd.DataFrame:
    """Bin log_lti into LTI histogram cells for every requested year/series.

    df must contain year, log_lti, PREDICTED_LABEL_VAR, and (for years where
    it's available) LABEL_VAR -- i.e. classify.py's combined output. The
    "actual_first"/"actual_second" series are only emitted for years with at
    least one non-null label (2004-2016); "raw" and the "predicted_*" series
    are emitted for every year.
    """
    if years is None:
        years = config.APPLY_YEARS

    rows = []
    for year in years:
        sub = df.loc[df["year"] == year]
        if sub.empty:
            continue
        lti = np.exp(sub["log_lti"].to_numpy())

        rows.append(_cells_for_series(lti, year, "raw"))

        actual = sub[config.LABEL_VAR]
        if actual.notna().any():
            actual_values = actual.to_numpy()
            for status, series in (
                (config.FIRST_LIEN_CLASS, "actual_first"),
                (config.SECOND_LIEN_CLASS, "actual_second"),
            ):
                rows.append(_cells_for_series(lti[actual_values == status], year, series))

        predicted = sub[config.PREDICTED_LABEL_VAR].to_numpy()
        for status, series in (
            (config.FIRST_LIEN_CLASS, "predicted_first"),
            (config.SECOND_LIEN_CLASS, "predicted_second"),
        ):
            rows.append(_cells_for_series(lti[predicted == status], year, series))

    cells = pd.concat(rows, ignore_index=True)
    return cells.sort_values(["year", "series", "lti_bin"]).reset_index(drop=True)


def render_histogram(cells: pd.DataFrame, year: int, series: str, output_path: Path) -> None:
    """Render one (year, series) LTI density histogram to output_path."""
    sample = cells.loc[(cells["year"] == year) & (cells["series"] == series)]
    if sample.empty:
        raise ValueError(f"No cells for year={year}, series={series!r}")
    sample = sample.sort_values("lti_bin")
    if not np.array_equal(sample["lti_bin"].to_numpy(), np.arange(BIN_COUNT)):
        raise ValueError(f"Missing LTI bins for year={year}, series={series!r}")

    counts = sample["count"].to_numpy(dtype=float)
    total = counts.sum()
    if total <= 0:
        raise ValueError(f"No observations for year={year}, series={series!r}")
    density = counts / (total * BIN_WIDTH)
    x = LTI_MIN + BIN_WIDTH * np.arange(BIN_COUNT)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(x, density, width=BIN_WIDTH, align="edge", alpha=BAR_ALPHA, edgecolor="black")
    ax.set_xlim(LTI_MIN, LTI_MAX)
    ax.set_xlabel("Loan-to-Income Ratio")
    ax.set_ylabel("Density")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def render_all(cells: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Render every (year, series) combination present in cells."""
    paths = []
    combos = cells[["year", "series"]].drop_duplicates().sort_values(["year", "series"])
    for year, series in combos.itertuples(index=False):
        path = output_dir / f"hmda_lti_{series}_{year}.pdf"
        render_histogram(cells, year, series, path)
        paths.append(path)
    return paths
