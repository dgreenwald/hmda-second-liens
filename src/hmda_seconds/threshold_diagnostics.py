"""Step 7 threshold, precision-recall, and subgroup diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score

from . import config, mixture, model_selection
from .density_ratio import checkpoints, evaluation
from .density_ratio import folds as temporal_folds
from .logistic_features import CENSUS_REGION_BY_STATE, REGION_LEVELS

CANONICAL_THRESHOLD = 0.5
PR_THRESHOLDS = np.unique(
    np.r_[0.0, expit(np.linspace(-12.0, 12.0, 241)), 0.5, 1.0]
)
ESTIMATORS = ("raw_logistic", "known_source_prior_mixture")
METRIC_COLUMNS = (
    "accuracy",
    "precision_second",
    "recall_second",
    "f1_second",
    "observed_second_share",
    "mean_probability",
    "hard_second_share",
    "brier_score",
    "average_precision",
)
SUBGROUP_METRIC_COLUMNS = METRIC_COLUMNS[:-1]


def threshold_metrics(
    y_second: np.ndarray,
    probability: np.ndarray,
    threshold: float = CANONICAL_THRESHOLD,
    include_average_precision: bool = True,
) -> dict[str, float | int]:
    """Return canonical hard-label and probability metrics."""
    y = np.asarray(y_second, dtype=bool)
    probability = np.asarray(probability, dtype=float)
    predicted = probability >= threshold
    true_positive = int(np.sum(y & predicted))
    false_positive = int(np.sum(~y & predicted))
    false_negative = int(np.sum(y & ~predicted))
    true_negative = int(np.sum(~y & ~predicted))
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    metrics = {
        "n": len(y),
        "threshold": threshold,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "accuracy": _safe_ratio(true_positive + true_negative, len(y)),
        "precision_second": precision,
        "recall_second": recall,
        "f1_second": _safe_ratio(2 * precision * recall, precision + recall),
        "observed_second_share": float(y.mean()),
        "mean_probability": float(probability.mean()),
        "hard_second_share": float(predicted.mean()),
        "brier_score": float(np.mean((probability - y) ** 2)),
    }
    if include_average_precision:
        metrics["average_precision"] = float(
            average_precision_score(y, probability)
        )
    return metrics


def precision_recall_counts(
    y_second: np.ndarray,
    probability: np.ndarray,
    thresholds: np.ndarray = PR_THRESHOLDS,
) -> pd.DataFrame:
    """Return confusion counts on a fixed, logit-spaced threshold grid."""
    y = np.asarray(y_second, dtype=bool)
    probability = np.asarray(probability, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    assignment = np.searchsorted(thresholds, probability, side="right")
    counts = np.bincount(assignment, minlength=len(thresholds) + 1)
    positives = np.bincount(
        assignment, weights=y.astype(float), minlength=len(thresholds) + 1
    )
    predicted = np.cumsum(counts[::-1])[::-1][1:]
    true_positive = np.cumsum(positives[::-1])[::-1][1:]
    false_positive = predicted - true_positive
    false_negative = y.sum() - true_positive
    precision = np.divide(
        true_positive,
        predicted,
        out=np.ones_like(true_positive),
        where=predicted > 0,
    )
    recall = true_positive / y.sum()
    return pd.DataFrame(
        {
            "threshold": thresholds,
            "true_positive": true_positive.astype(np.int64),
            "false_positive": false_positive.astype(np.int64),
            "false_negative": false_negative.astype(np.int64),
            "precision": precision,
            "recall": recall,
        }
    )


def subgroup_metrics(
    frame: pd.DataFrame,
    y_second: np.ndarray,
    probability: np.ndarray,
    threshold: float = CANONICAL_THRESHOLD,
) -> pd.DataFrame:
    """Return metrics for predeclared categorical and target-decile groups."""
    definitions = [
        ("loan_type", frame["loan_type"], tuple(config.CATEGORY_LEVELS["loan_type"])),
        (
            "purchaser_type",
            frame["purchaser_type"],
            tuple(config.CATEGORY_LEVELS["purchaser_type"]),
        ),
    ]
    regions = pd.to_numeric(frame["state_code"], errors="coerce").map(
        CENSUS_REGION_BY_STATE
    )
    definitions.append(("region", regions, REGION_LEVELS))
    rows = []
    for dimension, values, levels in definitions:
        rows.extend(
            _categorical_group_rows(
                dimension, values, levels, y_second, probability, threshold
            )
        )
    for variable in config.CONTINUOUS_VARS:
        rows.extend(
            _continuous_decile_rows(
                variable,
                frame[variable].to_numpy(dtype=float),
                y_second,
                probability,
                threshold,
            )
        )
    return pd.DataFrame(rows)


def run_threshold_diagnostics(
    data_dir: str | Path = config.SELECTION_DATA_DIR,
    output_dir: str | Path = config.TABLE_DIR,
    figure_dir: str | Path = config.FIGURE_DIR,
    model_file: str | Path = config.SELECTED_LOGISTIC_MODEL_FILE,
    fold_model_dir: str | Path = config.MIXTURE_FOLD_MODEL_DIR,
) -> dict[str, pd.DataFrame]:
    """Run frozen Step 7 diagnostics backward and forward."""
    output_dir = Path(output_dir)
    figure_dir = Path(figure_dir)
    fold_model_dir = Path(fold_model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    data_by_year = model_selection.load_selection_years(data_dir=data_dir)
    selected = model_selection.load_selected_model(model_file)

    reverse = _run_design(
        data_by_year,
        list(reversed(temporal_folds.reverse_folds())),
        selected,
        fold_model_dir,
        output_dir,
        "reverse",
    )
    forward_fold = temporal_folds.forward_fold(
        config.TRAIN_YEARS, config.VALIDATE_YEARS
    )
    forward = _run_design(
        data_by_year,
        [forward_fold],
        selected,
        fold_model_dir,
        output_dir,
        "forward",
    )
    outputs = {}
    for design, frames in (("reverse", reverse), ("forward", forward)):
        cells, pr_cells, subgroup_cells = frames
        horizons, summary = aggregate_threshold_metrics(cells)
        pr_summary = aggregate_precision_recall(pr_cells)
        subgroup_horizons, subgroup_summary = aggregate_subgroup_metrics(
            subgroup_cells
        )
        design_outputs = {
            "threshold_cells": cells,
            "threshold_horizons": horizons,
            "threshold_summary": summary,
            "precision_recall": pr_summary,
            "subgroup_cells": subgroup_cells,
            "subgroup_horizons": subgroup_horizons,
            "subgroup_summary": subgroup_summary,
        }
        for name, result in design_outputs.items():
            key = f"{design}_{name}"
            outputs[key] = result
            result.to_csv(output_dir / f"step7_{key}.csv", index=False)
        render_precision_recall_panels(
            pr_summary,
            panel="horizon" if design == "reverse" else "validation_year",
            output_file=figure_dir / f"step7_{design}_precision_recall.pdf",
        )
    return outputs


def aggregate_threshold_metrics(
    cells: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average cells within horizon, then horizons equally by estimator."""
    horizons = (
        cells.groupby(["estimator", "horizon"], as_index=False)
        .agg(
            **{column: (column, "mean") for column in METRIC_COLUMNS},
            n_cells=("validation_year", "size"),
            n_loans=("n", "sum"),
        )
        .sort_values(["estimator", "horizon"])
    )
    summary = (
        horizons.groupby("estimator", as_index=False)
        .agg(
            **{column: (column, "mean") for column in METRIC_COLUMNS},
            n_horizons=("horizon", "nunique"),
            n_cells=("n_cells", "sum"),
        )
        .sort_values("estimator")
    )
    return horizons, summary


def aggregate_precision_recall(cells: pd.DataFrame) -> pd.DataFrame:
    """Pool fixed-grid confusion counts within each evaluation panel."""
    panel = "horizon" if cells["evaluation_design"].iloc[0] == "reverse" else "validation_year"
    grouped = (
        cells.groupby(["estimator", panel, "threshold"], as_index=False)
        .agg(
            true_positive=("true_positive", "sum"),
            false_positive=("false_positive", "sum"),
            false_negative=("false_negative", "sum"),
            n_cells=("validation_year", "nunique"),
        )
        .sort_values([panel, "estimator", "threshold"])
    )
    grouped["precision"] = grouped["true_positive"] / (
        grouped["true_positive"] + grouped["false_positive"]
    ).replace(0, np.nan)
    grouped["precision"] = grouped["precision"].fillna(1.0)
    grouped["recall"] = grouped["true_positive"] / (
        grouped["true_positive"] + grouped["false_negative"]
    )
    return grouped


def aggregate_subgroup_metrics(
    cells: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average subgroup cells within horizon and then horizons equally."""
    groups = ["estimator", "subgroup_dimension", "subgroup_value", "horizon"]
    horizons = (
        cells.groupby(groups, as_index=False)
        .agg(
            **{column: (column, "mean") for column in SUBGROUP_METRIC_COLUMNS},
            n_cells=("validation_year", "size"),
            mean_n=("n", "mean"),
            min_n=("n", "min"),
        )
        .sort_values(groups)
    )
    summary_groups = ["estimator", "subgroup_dimension", "subgroup_value"]
    summary = (
        horizons.groupby(summary_groups, as_index=False)
        .agg(
            **{column: (column, "mean") for column in SUBGROUP_METRIC_COLUMNS},
            n_horizons=("horizon", "nunique"),
            min_cell_n=("min_n", "min"),
        )
        .sort_values(summary_groups)
    )
    return horizons, summary


def render_precision_recall_panels(
    curves: pd.DataFrame, panel: str, output_file: str | Path
) -> None:
    """Render raw and adjusted fixed-grid precision-recall curves."""
    values = sorted(curves[panel].unique())
    fig, axes = plt.subplots(3, 3, figsize=(9.0, 9.0), squeeze=False)
    colors = {"raw_logistic": "0.45", "known_source_prior_mixture": "C0"}
    labels = {"raw_logistic": "Raw logistic", "known_source_prior_mixture": "Mixture"}
    for axis, value in zip(axes.flat, values, strict=False):
        sample = curves.loc[curves[panel] == value]
        for estimator in ESTIMATORS:
            line = sample.loc[sample["estimator"] == estimator].sort_values("recall")
            axis.plot(
                line["recall"],
                line["precision"],
                color=colors[estimator],
                label=labels[estimator],
            )
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_title(f"{panel.replace('_', ' ').title()} {value}")
        axis.set_xlabel("Second-lien recall")
        axis.set_ylabel("Second-lien precision")
    axes.flat[0].legend(loc="lower left")
    for axis in axes.flat[len(values) :]:
        axis.set_visible(False)
    fig.tight_layout()
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file)
    plt.close(fig)


def _run_design(
    data_by_year: dict[int, pd.DataFrame],
    folds: list[temporal_folds.TemporalFold],
    selected: model_selection.SelectedLogisticModel,
    fold_model_dir: Path,
    output_dir: Path,
    design: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prefix = f"step7_{design}"
    cells_file = output_dir / f"{prefix}_threshold_cells.csv"
    pr_file = output_dir / f"{prefix}_precision_recall_cells.csv"
    subgroup_file = output_dir / f"{prefix}_subgroup_cells.csv"
    cells, pr_cells, subgroup_cells = map(
        checkpoints.read_csv, (cells_file, pr_file, subgroup_file)
    )
    for fold in folds:
        missing = [
            year
            for year in fold.validation_years
            if not _cell_complete(
                cells, pr_cells, subgroup_cells, fold.train_start, year
            )
        ]
        known_path = mixture.known_source_prior_model_path(
            fold.train_years,
            selected.specification,
            selected.regularization_c,
            fold_model_dir,
        )
        known = mixture.load_known_source_prior_model(known_path)
        raw_path = _raw_model_path(
            fold.train_years,
            selected.specification.name,
            selected.regularization_c,
            fold_model_dir,
        )
        if raw_path.exists():
            raw = model_selection.load_selected_model(raw_path)
        elif tuple(fold.train_years) == tuple(config.TRAIN_YEARS):
            raw = selected
            model_selection.save_selected_model(
                raw, raw_path, train_years=fold.train_years
            )
        else:
            training = pd.concat(
                [data_by_year[year] for year in fold.train_years],
                ignore_index=True,
            )
            raw = model_selection.fit_selected_model(
                training, selected.specification, selected.regularization_c
            )
            model_selection.save_selected_model(raw, raw_path)
        if not missing:
            continue
        if raw.transformer.feature_names_ != known.transformer.feature_names_:
            raise RuntimeError("Raw and known-prior fold feature columns differ")
        for validation_year in missing:
            target = data_by_year[validation_year]
            features = known.transformer.transform(target)
            evaluated = evaluation.evaluate_target(
                known,
                target,
                fold,
                label_var=config.LABEL_VAR,
                second_lien_class=config.SECOND_LIEN_CLASS,
            )
            adjusted_probability = evaluated.probability
            second_column = list(raw.classifier.classes_).index(
                config.SECOND_LIEN_CLASS
            )
            raw_probability = raw.classifier.predict_proba(features)[:, second_column]
            y_second = (
                target[config.LABEL_VAR].to_numpy() == config.SECOND_LIEN_CLASS
            )
            horizon = fold.horizon_for(validation_year)
            metadata = {
                "evaluation_design": design,
                "specification": selected.specification.name,
                "regularization_c": selected.regularization_c,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "validation_year": validation_year,
                "horizon": horizon,
            }
            for estimator, probability in (
                ("raw_logistic", raw_probability),
                ("known_source_prior_mixture", adjusted_probability),
            ):
                estimator_metadata = {**metadata, "estimator": estimator}
                metric_row = pd.DataFrame(
                    [
                        {
                            **estimator_metadata,
                            **threshold_metrics(y_second, probability),
                        }
                    ]
                )
                pr_rows = precision_recall_counts(y_second, probability).assign(
                    **estimator_metadata
                )
                subgroup_rows = subgroup_metrics(
                    target, y_second, probability
                ).assign(**estimator_metadata)
                key_columns = (
                    "train_start",
                    "validation_year",
                    "estimator",
                )
                cells = checkpoints.replace_rows(
                    cells, metric_row, cells_file, key_columns=key_columns
                )
                pr_cells = checkpoints.replace_rows(
                    pr_cells, pr_rows, pr_file, key_columns=key_columns
                )
                subgroup_cells = checkpoints.replace_rows(
                    subgroup_cells,
                    subgroup_rows,
                    subgroup_file,
                    key_columns=key_columns,
                )
    return cells, pr_cells, subgroup_cells


def _categorical_group_rows(
    dimension: str,
    values: pd.Series,
    levels: tuple,
    y_second: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> list[dict]:
    rows = []
    array = np.asarray(values)
    for level in levels:
        keep = array == level
        if keep.any():
            rows.append(
                {
                    "subgroup_dimension": dimension,
                    "subgroup_value": str(level),
                    **threshold_metrics(
                        y_second[keep],
                        probability[keep],
                        threshold,
                        include_average_precision=False,
                    ),
                }
            )
    return rows


def _continuous_decile_rows(
    variable: str,
    values: np.ndarray,
    y_second: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> list[dict]:
    cuts = np.unique(np.quantile(values, np.linspace(0, 1, 11))[1:-1])
    codes = np.searchsorted(cuts, values, side="right")
    rows = []
    for code in range(len(cuts) + 1):
        keep = codes == code
        rows.append(
            {
                "subgroup_dimension": f"{variable}_target_decile",
                "subgroup_value": str(code + 1),
                **threshold_metrics(
                    y_second[keep],
                    probability[keep],
                    threshold,
                    include_average_precision=False,
                ),
            }
        )
    return rows


def _raw_model_path(
    train_years: tuple[int, ...],
    specification: str,
    regularization_c: float,
    model_dir: Path,
) -> Path:
    c_label = format(regularization_c, ".12g").replace(".", "p")
    return model_dir / (
        f"raw_logistic__{specification}__c_{c_label}"
        f"__train_{min(train_years)}_{max(train_years)}.pkl"
    )


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _cell_complete(
    cells: pd.DataFrame,
    pr_cells: pd.DataFrame,
    subgroup_cells: pd.DataFrame,
    train_start: int,
    validation_year: int,
) -> bool:
    for frame in (cells, pr_cells, subgroup_cells):
        if frame.empty:
            return False
        matching = (
            (frame["train_start"] == train_start)
            & (frame["validation_year"] == validation_year)
        )
        if set(frame.loc[matching, "estimator"]) != set(ESTIMATORS):
            return False
    return True
