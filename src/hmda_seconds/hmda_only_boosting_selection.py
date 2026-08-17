"""Frozen staged gradient-boosting selection using HMDA predictors only."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from py_tools import cluster as cluster_tools

from . import config, gradient_boosting, model_selection
from .density_ratio.cluster import make_job
from .density_ratio.families.gradient_boosting import (
    HMDA_ONLY_SPECIFICATION,
    BoostingParameters,
    fit_boosting_ratio_model,
    save_boosting_model,
)
from .density_ratio.folds import reverse_folds
from .density_ratio.protocols import ModelConfiguration
from .logistic_features import HMDA_ONLY_FEATURE_SET

SCREEN_STAGE = "hmda_only_boosting_screen"
SURVIVOR_STAGE = "hmda_only_boosting_survivors"
REFINEMENT_STAGE = "hmda_only_boosting_refinement"
STAGES = (SCREEN_STAGE, SURVIVOR_STAGE, REFINEMENT_STAGE)
FAMILY = "hist_gradient_boosting"


def screen_jobs(*, data_dir: str | Path, output_root: str | Path):
    """Return the six-structure screen on the latest reverse fold."""
    fold = max(reverse_folds(), key=lambda item: item.train_start)
    return [
        make_job(
            stage=SCREEN_STAGE,
            family=FAMILY,
            specification=HMDA_ONLY_SPECIFICATION,
            train_start=fold.train_start,
            configurations=_configurations(gradient_boosting.structure_grid()),
            data_dir=data_dir,
            output_root=output_root,
        )
    ]


def survivor_jobs(
    screen_summary: pd.DataFrame,
    *,
    data_dir: str | Path,
    output_root: str | Path,
):
    """Retain the two screen winners and evaluate both on all folds."""
    _validate_summary(screen_summary, expected_candidates=6, expected_cells=9)
    survivors = [
        _parameters_from_row(row)
        for _, row in screen_summary.head(config.BOOSTING_SCREEN_SURVIVORS).iterrows()
    ]
    configurations = _configurations(survivors)
    return [
        make_job(
            stage=SURVIVOR_STAGE,
            family=FAMILY,
            specification=HMDA_ONLY_SPECIFICATION,
            train_start=fold.train_start,
            configurations=configurations,
            data_dir=data_dir,
            output_root=output_root,
        )
        for fold in reverse_folds()
    ]


def refinement_jobs(
    survivor_summary: pd.DataFrame,
    *,
    data_dir: str | Path,
    output_root: str | Path,
):
    """Evaluate four one-coordinate refinements around the best survivor."""
    _validate_summary(
        survivor_summary,
        expected_candidates=config.BOOSTING_SCREEN_SURVIVORS,
        expected_cells=45,
    )
    best = _parameters_from_row(survivor_summary.iloc[0])
    configurations = _configurations(gradient_boosting.refinement_grid(best))
    if len(configurations) != 4:
        raise ValueError("The frozen boosting refinement must contain four candidates")
    return [
        make_job(
            stage=REFINEMENT_STAGE,
            family=FAMILY,
            specification=HMDA_ONLY_SPECIFICATION,
            train_start=fold.train_start,
            configurations=configurations,
            data_dir=data_dir,
            output_root=output_root,
        )
        for fold in reverse_folds()
    ]


def finalize_selection_tables(
    survivor_dir: str | Path,
    refinement_dir: str | Path,
    output_dir: str | Path,
    *,
    model_output: str | Path = config.HMDA_ONLY_SELECTED_BOOSTING_MODEL_FILE,
) -> list[Path]:
    """Combine the eligible stages and write the restricted decision table."""
    survivor_dir = Path(survivor_dir)
    refinement_dir = Path(refinement_dir)
    survivor_summary = pd.read_csv(survivor_dir / "density_ratio_summary.csv")
    refinement_summary = pd.read_csv(refinement_dir / "density_ratio_summary.csv")
    _validate_summary(
        survivor_summary,
        expected_candidates=config.BOOSTING_SCREEN_SURVIVORS,
        expected_cells=45,
    )
    _validate_summary(refinement_summary, expected_candidates=4, expected_cells=45)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destinations = []
    combined = {}
    for name in ("cells", "horizons", "summary"):
        frames = [
            pd.read_csv(survivor_dir / f"density_ratio_{name}.csv"),
            pd.read_csv(refinement_dir / f"density_ratio_{name}.csv"),
        ]
        frame = pd.concat(frames, ignore_index=True)
        key = {
            "cells": ["configuration_id", "fold_id", "target_year"],
            "horizons": ["configuration_id", "horizon"],
            "summary": ["configuration_id"],
        }[name]
        if frame.duplicated(key).any():
            raise ValueError(f"Eligible boosting {name} contain duplicate rows")
        sort = {
            "cells": ["configuration_id", "horizon", "target_year", "fold_id"],
            "horizons": ["configuration_id", "horizon"],
            "summary": ["selection_brier", "configuration_id"],
        }[name]
        frame = frame.sort_values(sort).reset_index(drop=True)
        combined[name] = frame
        destination = output_dir / f"hmda_only_boosting_{name}.csv"
        _atomic_csv(frame, destination)
        destinations.append(destination)

    summary = combined["summary"]
    _validate_summary(summary, expected_candidates=6, expected_cells=45)
    winner = summary.iloc[0]
    decision = pd.DataFrame(
        [
            {
                **winner.to_dict(),
                "screen_train_start": max(fold.train_start for fold in reverse_folds()),
                "screen_survivors": config.BOOSTING_SCREEN_SURVIVORS,
                "selection_metric": "mixture_adjusted_brier_equal_horizon_weight",
                "model_file": str(model_output),
                "training_years": "2004-2007",
            }
        ]
    )
    decision_path = output_dir / "hmda_only_boosting_decision.csv"
    _atomic_csv(decision, decision_path)
    destinations.append(decision_path)
    return destinations


def fit_final_model(
    decision_file: str | Path,
    *,
    data_dir: str | Path = config.SELECTION_DATA_DIR,
    model_output: str | Path = config.HMDA_ONLY_SELECTED_BOOSTING_MODEL_FILE,
    overwrite: bool = False,
) -> Path:
    """Refit the declared HMDA-only boosting winner on 2004--2007."""
    model_output = Path(model_output)
    if model_output.exists() and not overwrite:
        raise FileExistsError(
            f"Selected model already exists: {model_output}; "
            "pass --overwrite to replace it"
        )
    decision = pd.read_csv(decision_file)
    if len(decision) != 1:
        raise ValueError("Selection decision must contain exactly one row")
    row = decision.iloc[0]
    if row.get("family") != FAMILY:
        raise ValueError("Selection decision has the wrong model family")
    if row.get("specification") != HMDA_ONLY_SPECIFICATION:
        raise ValueError("Selection decision has the wrong boosting specification")
    parameters = _parameters_from_row(row)
    data = model_selection.load_selection_years(
        data_dir,
        config.TRAIN_YEARS,
        feature_set=HMDA_ONLY_FEATURE_SET,
    )
    training = pd.concat([data[year] for year in config.TRAIN_YEARS], ignore_index=True)
    fitted, _ = fit_boosting_ratio_model(
        training,
        parameters,
        specification=HMDA_ONLY_SPECIFICATION,
    )
    save_boosting_model(fitted, model_output)
    return model_output


def write_finalize_slurm(
    *,
    destination: str | Path,
    repo_dir: str | Path,
    decision_file: str | Path = config.TABLE_DIR / "hmda_only_boosting_decision.csv",
    data_dir: str | Path = config.SELECTION_DATA_DIR,
    model_output: str | Path = config.HMDA_ONLY_SELECTED_BOOSTING_MODEL_FILE,
    activate: str | None = None,
    account: str | None = "torch_pr_609_general",
    time_limit: str = "8:00:00",
    memory: str = "32G",
) -> Path:
    """Write the single-job Slurm script for the selected boosting refit."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    repo_dir = Path(repo_dir).resolve()
    script = destination / "finalize_hmda_only_boosting.slurm"
    cluster_tools.write_slurm_script(
        cluster_tools.SlurmJob(
            name="hmda-only-boost-final",
            command=(
                "python",
                "scripts/finalize_hmda_only_boosting.py",
                "--decision",
                Path(decision_file).resolve(),
                "--data-dir",
                Path(data_dir).resolve(),
                "--model-output",
                Path(model_output).resolve(),
            ),
            workdir=repo_dir,
            log_dir=destination,
            resources=cluster_tools.SlurmResources(
                time=time_limit,
                memory=memory,
                account=account,
            ),
            activate=activate,
        ),
        script,
    )
    return script


def _configurations(parameters):
    return tuple(
        ModelConfiguration.from_mapping(
            FAMILY,
            HMDA_ONLY_SPECIFICATION,
            asdict(candidate),
            random_seed=config.BOOSTING_RANDOM_STATE,
        )
        for candidate in parameters
    )


def _parameters_from_row(row: pd.Series) -> BoostingParameters:
    values = json.loads(str(row["hyperparameters"]))
    return BoostingParameters(
        max_leaf_nodes=int(values["max_leaf_nodes"]),
        learning_rate=float(values["learning_rate"]),
        max_iter=int(values["max_iter"]),
        l2_regularization=float(values["l2_regularization"]),
        min_samples_leaf=int(values["min_samples_leaf"]),
    )


def _validate_summary(
    summary: pd.DataFrame, *, expected_candidates: int, expected_cells: int
) -> None:
    required = {
        "configuration_id",
        "family",
        "specification",
        "hyperparameters",
        "selection_brier",
        "n_horizons",
        "n_cells",
    }
    missing = required - set(summary)
    if missing:
        raise ValueError(f"Boosting summary missing columns: {sorted(missing)}")
    if (
        len(summary) != expected_candidates
        or summary["configuration_id"].duplicated().any()
    ):
        raise ValueError("Boosting summary has an incomplete candidate set")
    if set(summary["family"]) != {FAMILY}:
        raise ValueError("Boosting summary has the wrong family")
    if set(summary["specification"]) != {HMDA_ONLY_SPECIFICATION}:
        raise ValueError("Boosting summary has the wrong specification")
    if (
        not (summary["n_horizons"] == 9).all()
        or not (summary["n_cells"] == expected_cells).all()
    ):
        raise ValueError("Boosting summary has an incomplete reverse design")


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w") as file:
            frame.to_csv(file, index=False)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
