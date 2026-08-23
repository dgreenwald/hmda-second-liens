"""Common mixture-adjusted diagnostics for the four frozen finalists."""

from __future__ import annotations

import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config, model_selection
from .density_ratio import artifacts
from .density_ratio.cluster import expand_job_paths
from .density_ratio.families.gradient_boosting import load_boosting_model
from .density_ratio.folds import forward_fold, reverse_folds
from .density_ratio.protocols import JobSpecification, ModelConfiguration
from .density_ratio.shards import (
    PlannedJob,
    aggregate_shards,
    read_manifest,
    read_shard,
    shard_path,
)

REVERSE_STAGE = "model_family_comparison_reverse"
FORWARD_STAGE = "model_family_comparison_forward"
METRICS = (
    "brier_score",
    "log_loss",
    "calibration_mean_error",
    "calibration_intercept",
    "calibration_slope",
    "share_error",
    "absolute_share_error",
    "hard_share_error_050",
)
CELL_KEYS = (
    "evaluation_design",
    "predictor_set",
    "train_start",
    "train_end",
    "target_year",
    "horizon",
)


def selected_configurations(
    *,
    logistic_file: str | Path = config.SELECTED_LOGISTIC_MODEL_FILE,
    hmda_logistic_file: str | Path = config.HMDA_ONLY_SELECTED_LOGISTIC_MODEL_FILE,
    boosting_file: str | Path = config.SELECTED_BOOSTING_MODEL_FILE,
    hmda_boosting_file: str | Path = config.HMDA_ONLY_SELECTED_BOOSTING_MODEL_FILE,
) -> tuple[ModelConfiguration, ...]:
    """Translate the four selected artifacts into density-ratio configurations."""
    paths = (logistic_file, hmda_logistic_file, boosting_file, hmda_boosting_file)
    for path in paths:
        artifacts.validate_existing_artifact(path, allow_legacy=False)

    configurations = []
    for path in (logistic_file, hmda_logistic_file):
        selected = model_selection.load_selected_model(path)
        if selected.train_years != tuple(config.TRAIN_YEARS):
            raise ValueError("Selected logistic finalist has the wrong training years")
        configurations.append(
            ModelConfiguration.from_mapping(
                "logistic",
                selected.specification.name,
                {"C": selected.regularization_c},
            )
        )
    for path in (boosting_file, hmda_boosting_file):
        selected = load_boosting_model(path)
        if selected.train_years != tuple(config.TRAIN_YEARS):
            raise ValueError("Selected boosting finalist has the wrong training years")
        configurations.append(
            ModelConfiguration.from_mapping(
                "hist_gradient_boosting",
                selected.specification,
                asdict(selected.parameters),
                random_seed=config.BOOSTING_RANDOM_STATE,
            )
        )
    if len(set(configurations)) != 4:
        raise ValueError("Selected finalist configurations must be distinct")
    return tuple(configurations)


def comparison_jobs(
    *,
    data_dir: str | Path = config.SELECTION_DATA_DIR,
    output_root: str | Path = config.MODEL_FAMILY_COMPARISON_DIR,
    configurations: tuple[ModelConfiguration, ...] | None = None,
    **artifact_files,
) -> list[PlannedJob]:
    """Plan one immutable fit per finalist and temporal source window."""
    configurations = configurations or selected_configurations(**artifact_files)
    folds = [*reverse_folds(), forward_fold(config.TRAIN_YEARS, config.VALIDATE_YEARS)]
    jobs = []
    for configuration in configurations:
        for fold in folds:
            stage = REVERSE_STAGE if fold.direction == "reverse" else FORWARD_STAGE
            job = JobSpecification(
                stage=stage,
                family=configuration.family,
                specification=configuration.specification,
                train_years=fold.train_years,
                configurations=(configuration,),
                input_paths=(("selection_data_dir", str(data_dir)),),
                output_root=str(output_root),
            )
            jobs.append(PlannedJob(job, fold))
    return jobs


def aggregate_comparison(
    manifest: str | Path,
    *,
    output_dir: str | Path = config.TABLE_DIR,
    figure_dir: str | Path = config.FIGURE_DIR,
) -> list[Path]:
    """Validate all comparison shards and publish matched diagnostic outputs."""
    planned = [expand_job_paths(item) for item in read_manifest(manifest)]
    parts = []
    for direction in ("reverse", "forward"):
        selected = [item for item in planned if item.fold.direction == direction]
        if len(selected) != (36 if direction == "reverse" else 4):
            raise ValueError(f"Expected complete four-model {direction} job design")
        _validate_artifacts(selected)
        aggregated = aggregate_shards(
            selected, [shard_path(item.job) for item in selected]
        )
        frame = aggregated.cells.copy()
        frame["evaluation_design"] = direction
        parts.append(frame)
    cells = _decorate_cells(pd.concat(parts, ignore_index=True))
    _validate_cells(cells)
    reverse_horizons, summary = _aggregate_metrics(cells)
    paired_cells = _paired_cells(cells)
    paired_summary = _paired_summary(paired_cells)

    output_dir = Path(output_dir)
    figure_dir = Path(figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "cells": cells,
        "reverse_horizons": reverse_horizons,
        "summary": summary,
        "paired_cells": paired_cells,
        "paired_summary": paired_summary,
    }
    destinations = []
    for name, frame in outputs.items():
        destination = output_dir / f"model_family_comparison_{name}.csv"
        _atomic_csv(frame, destination)
        destinations.append(destination)
    for design in ("reverse", "forward"):
        destination = figure_dir / f"model_family_comparison_{design}.pdf"
        _render_metrics(cells, design, destination)
        destinations.append(destination)
    return destinations


def _decorate_cells(cells: pd.DataFrame) -> pd.DataFrame:
    result = cells.copy()
    result["model_family"] = result["family"].map(
        {"logistic": "logistic", "hist_gradient_boosting": "boosting"}
    )
    result["predictor_set"] = np.where(
        result["specification"].str.startswith("hmda_only"),
        "hmda_only",
        "unrestricted",
    )
    result["share_error"] = result["mixture_share"] - result["actual_second_share"]
    result["absolute_share_error"] = result["share_error"].abs()
    result["hard_share_error_050"] = (
        result["hard_share_050"] - result["actual_second_share"]
    )
    order = [
        "evaluation_design",
        "predictor_set",
        "model_family",
        "configuration_id",
        "family",
        "specification",
        "hyperparameters",
        "random_seed",
        "model_id",
        "fold_id",
        "train_start",
        "train_end",
        "target_year",
        "horizon",
        "n_observations",
        "actual_second_share",
        "mixture_share",
        "share_error",
        "absolute_share_error",
        "mean_probability",
        "hard_share_050",
        "hard_share_error_050",
        "brier_score",
        "log_loss",
        "calibration_mean_error",
        "calibration_intercept",
        "calibration_slope",
        "optimizer_converged",
        "mixture_at_boundary",
        "mixture_em_difference",
        "schema_version",
    ]
    return result[order].sort_values(
        ["evaluation_design", "predictor_set", "model_family", "train_start", "target_year"]
    ).reset_index(drop=True)


def _validate_cells(cells: pd.DataFrame) -> None:
    counts = cells.groupby(["predictor_set", "model_family", "evaluation_design"]).size()
    for predictor_set in ("unrestricted", "hmda_only"):
        for family in ("logistic", "boosting"):
            if counts.get((predictor_set, family, "reverse"), 0) != 45:
                raise ValueError("Every finalist must contain 45 reverse cells")
            if counts.get((predictor_set, family, "forward"), 0) != 9:
                raise ValueError("Every finalist must contain nine forward cells")
    support = ["n_observations", "actual_second_share"]
    grouped = cells.groupby(["evaluation_design", "train_start", "target_year"])
    for column in support:
        if (grouped[column].nunique() != 1).any():
            raise ValueError(f"Finalists do not share identical {column} by cell")
    if not np.allclose(cells["mean_probability"], cells["mixture_share"], atol=1e-7):
        raise ValueError("Adjusted probability means do not equal mixture shares")
    if not cells["optimizer_converged"].all():
        raise ValueError("Comparison contains a non-converged mixture estimate")
    if cells["mixture_at_boundary"].any():
        raise ValueError("Comparison contains a mixture estimate at the boundary")
    if cells.duplicated([*CELL_KEYS, "model_family"]).any():
        raise ValueError("Comparison contains duplicate model cells")


def _validate_artifacts(planned: list[PlannedJob]) -> None:
    """Verify every shard-referenced fit and its metadata identity."""
    seen: set[Path] = set()
    for item in planned:
        shard = read_shard(shard_path(item.job))
        for model in shard.models:
            path = Path(model.artifact_path)
            if path in seen:
                raise ValueError(f"Comparison reuses fitted artifact {path}")
            seen.add(path)
            metadata = artifacts.validate_existing_artifact(path, allow_legacy=False)
            artifacts.validate_metadata_identity(
                metadata,
                model_id=model.model_id,
                train_years=item.fold.train_years,
            )
            if metadata is None or metadata.configuration != model.configuration:
                raise ValueError(
                    f"Artifact configuration does not match shard model {model.model_id}"
                )


def _aggregate_metrics(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group = ["predictor_set", "model_family", "configuration_id"]
    reverse = cells.loc[cells["evaluation_design"] == "reverse"]
    horizons = reverse.groupby([*group, "horizon"], as_index=False).agg(
        **{metric: (metric, "mean") for metric in METRICS},
        n_cells=("brier_score", "size"),
        n_loans=("n_observations", "sum"),
    )
    reverse_summary = horizons.groupby(group, as_index=False).agg(
        **{metric: (metric, "mean") for metric in METRICS},
        n_horizons=("horizon", "nunique"),
        n_cells=("n_cells", "sum"),
    )
    reverse_summary.insert(0, "evaluation_design", "reverse")
    forward = cells.loc[cells["evaluation_design"] == "forward"]
    forward_summary = forward.groupby(group, as_index=False).agg(
        **{metric: (metric, "mean") for metric in METRICS},
        n_cells=("brier_score", "size"),
    )
    forward_summary.insert(0, "evaluation_design", "forward")
    forward_summary["n_horizons"] = np.nan
    summary = pd.concat([reverse_summary, forward_summary], ignore_index=True)
    summary["weighting"] = np.where(
        summary["evaluation_design"] == "reverse",
        "equal_within_horizon_then_equal_across_horizons",
        "equal_across_validation_years",
    )
    return horizons.sort_values([*group, "horizon"]).reset_index(drop=True), summary


def _paired_cells(cells: pd.DataFrame) -> pd.DataFrame:
    index = [*CELL_KEYS]
    keep = [*index, "model_family", *METRICS]
    wide = cells[keep].pivot(index=index, columns="model_family", values=list(METRICS))
    if wide.isna().any().any():
        raise ValueError("Logistic and boosting cells are not one-to-one")
    wide.columns = [f"{metric}_{family}" for metric, family in wide.columns]
    result = wide.reset_index()
    for metric in METRICS:
        result[f"boosting_minus_logistic_{metric}"] = (
            result[f"{metric}_boosting"] - result[f"{metric}_logistic"]
        )
    return result.sort_values(index).reset_index(drop=True)


def _paired_summary(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (design, predictor_set), frame in paired.groupby(
        ["evaluation_design", "predictor_set"], sort=True
    ):
        if design == "reverse":
            by_horizon = frame.groupby("horizon", as_index=False)[
                [f"boosting_minus_logistic_{metric}" for metric in METRICS]
            ].mean()
            differences = {metric: by_horizon[f"boosting_minus_logistic_{metric}"].mean() for metric in METRICS}
            weighting = "equal_within_horizon_then_equal_across_horizons"
        else:
            differences = {metric: frame[f"boosting_minus_logistic_{metric}"].mean() for metric in METRICS}
            weighting = "equal_across_validation_years"
        rows.append(
            {
                "evaluation_design": design,
                "predictor_set": predictor_set,
                **{f"boosting_minus_logistic_{key}": value for key, value in differences.items()},
                "n_cells_boosting_lower_brier": int((frame["boosting_minus_logistic_brier_score"] < 0).sum()),
                "n_cells": len(frame),
                "weighting": weighting,
            }
        )
    return pd.DataFrame(rows)


def _render_metrics(cells: pd.DataFrame, design: str, destination: Path) -> None:
    selected = cells.loc[cells["evaluation_design"] == design]
    panel = "horizon" if design == "reverse" else "target_year"
    if design == "reverse":
        selected = selected.groupby(
            ["predictor_set", "model_family", panel], as_index=False
        )[["brier_score", "log_loss", "share_error"]].mean()
    fig, axes = plt.subplots(3, 2, figsize=(10, 10), sharex="col")
    labels = (("brier_score", "Brier score"), ("log_loss", "Log loss"), ("share_error", "Share error"))
    for column, predictor_set in enumerate(("unrestricted", "hmda_only")):
        subset = selected.loc[selected["predictor_set"] == predictor_set]
        for row, (metric, label) in enumerate(labels):
            axis = axes[row, column]
            for family, color in (("logistic", "C0"), ("boosting", "C1")):
                values = subset.loc[subset["model_family"] == family].sort_values(panel)
                axis.plot(values[panel], values[metric], marker="o", label=family.title(), color=color)
            axis.axhline(0, color="0.7", linestyle="--", linewidth=0.8) if metric == "share_error" else None
            axis.set_ylabel(label)
            axis.set_title(predictor_set.replace("_", " ").title()) if row == 0 else None
            if row == 2:
                axis.set_xlabel(panel.replace("_", " ").title())
            if row == 0 and column == 0:
                axis.legend()
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination)
    plt.close(fig)


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
