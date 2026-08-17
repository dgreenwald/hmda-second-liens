"""Shared staged cluster selection for frozen gradient-boosting variants."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from py_tools import cluster as cluster_tools

from . import config, gradient_boosting, model_selection
from .density_ratio.cluster import make_job
from .density_ratio.families.gradient_boosting import (
    HMDA_ONLY_SPECIFICATION,
    SPECIFICATION,
    BoostingParameters,
    fit_boosting_ratio_model,
    save_boosting_model,
)
from .density_ratio.folds import reverse_folds
from .density_ratio.protocols import ModelConfiguration
from .logistic_features import CORE_FEATURE_SET, HMDA_ONLY_FEATURE_SET

FAMILY = "hist_gradient_boosting"


@dataclass(frozen=True)
class BoostingVariant:
    """Paths and identities that distinguish one frozen boosting protocol."""

    name: str
    specification: str
    feature_set: str
    stages: tuple[str, str, str]
    table_prefix: str
    model_output: Path
    expected_winner: BoostingParameters | None = None
    legacy_tables: bool = False

    def stage_label(self, stage: str) -> str:
        """Return the filesystem label for one declared stage."""
        try:
            index = self.stages.index(stage)
        except ValueError as error:
            raise ValueError(f"Unknown {self.name} boosting stage {stage!r}") from error
        return ("screen", "survivors", "refinement")[index]


CORE_FROZEN_WINNER = BoostingParameters(
    max_leaf_nodes=7,
    learning_rate=0.05,
    max_iter=200,
    l2_regularization=10.0,
    min_samples_leaf=1_000,
)

CORE_VARIANT = BoostingVariant(
    name="core",
    specification=SPECIFICATION,
    feature_set=CORE_FEATURE_SET,
    stages=("boosting_screen", "boosting_survivors", "boosting_refinement"),
    table_prefix="boosting_challenger",
    model_output=config.SELECTED_BOOSTING_MODEL_FILE,
    expected_winner=CORE_FROZEN_WINNER,
    legacy_tables=True,
)

HMDA_ONLY_VARIANT = BoostingVariant(
    name="hmda_only",
    specification=HMDA_ONLY_SPECIFICATION,
    feature_set=HMDA_ONLY_FEATURE_SET,
    stages=(
        "hmda_only_boosting_screen",
        "hmda_only_boosting_survivors",
        "hmda_only_boosting_refinement",
    ),
    table_prefix="hmda_only_boosting",
    model_output=config.HMDA_ONLY_SELECTED_BOOSTING_MODEL_FILE,
)


def screen_jobs(
    variant: BoostingVariant, *, data_dir: str | Path, output_root: str | Path
):
    """Return the six-structure screen on the latest reverse fold."""
    fold = max(reverse_folds(), key=lambda item: item.train_start)
    return [
        make_job(
            stage=variant.stages[0],
            family=FAMILY,
            specification=variant.specification,
            train_start=fold.train_start,
            configurations=_configurations(
                gradient_boosting.structure_grid(), variant.specification
            ),
            data_dir=data_dir,
            output_root=output_root,
        )
    ]


def survivor_jobs(
    variant: BoostingVariant,
    screen_summary: pd.DataFrame,
    *,
    data_dir: str | Path,
    output_root: str | Path,
):
    """Retain the two screen winners and evaluate both on all folds."""
    validate_summary(
        screen_summary,
        variant,
        expected_candidates=6,
        expected_cells=9,
    )
    survivors = [
        parameters_from_row(row)
        for _, row in screen_summary.head(config.BOOSTING_SCREEN_SURVIVORS).iterrows()
    ]
    configurations = _configurations(survivors, variant.specification)
    return [
        make_job(
            stage=variant.stages[1],
            family=FAMILY,
            specification=variant.specification,
            train_start=fold.train_start,
            configurations=configurations,
            data_dir=data_dir,
            output_root=output_root,
        )
        for fold in reverse_folds()
    ]


def refinement_jobs(
    variant: BoostingVariant,
    survivor_summary: pd.DataFrame,
    *,
    data_dir: str | Path,
    output_root: str | Path,
):
    """Evaluate four one-coordinate refinements around the best survivor."""
    validate_summary(
        survivor_summary,
        variant,
        expected_candidates=config.BOOSTING_SCREEN_SURVIVORS,
        expected_cells=45,
    )
    best = parameters_from_row(survivor_summary.iloc[0])
    configurations = _configurations(
        gradient_boosting.refinement_grid(best), variant.specification
    )
    if len(configurations) != 4:
        raise ValueError("The frozen boosting refinement must contain four candidates")
    return [
        make_job(
            stage=variant.stages[2],
            family=FAMILY,
            specification=variant.specification,
            train_start=fold.train_start,
            configurations=configurations,
            data_dir=data_dir,
            output_root=output_root,
        )
        for fold in reverse_folds()
    ]


def finalize_selection_tables(
    variant: BoostingVariant,
    survivor_dir: str | Path,
    refinement_dir: str | Path,
    output_dir: str | Path,
    *,
    model_output: str | Path | None = None,
) -> list[Path]:
    """Combine the eligible stages and write one validated decision table."""
    survivor_dir = Path(survivor_dir)
    refinement_dir = Path(refinement_dir)
    survivor_summary = pd.read_csv(survivor_dir / "density_ratio_summary.csv")
    refinement_summary = pd.read_csv(refinement_dir / "density_ratio_summary.csv")
    validate_summary(
        survivor_summary,
        variant,
        expected_candidates=config.BOOSTING_SCREEN_SURVIVORS,
        expected_cells=45,
    )
    validate_summary(
        refinement_summary,
        variant,
        expected_candidates=4,
        expected_cells=45,
    )

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
        combined[name] = frame.sort_values(sort).reset_index(drop=True)

    summary = combined["summary"]
    validate_summary(
        summary,
        variant,
        expected_candidates=6,
        expected_cells=45,
    )
    winner = summary.iloc[0]
    winner_parameters = parameters_from_row(winner)
    if (
        variant.expected_winner is not None
        and winner_parameters != variant.expected_winner
    ):
        raise ValueError(
            f"{variant.name} cluster winner {winner_parameters.identifier} does not "
            f"match frozen result {variant.expected_winner.identifier}"
        )

    output_tables = combined
    if variant.legacy_tables:
        compatibility_cells = gradient_boosting.translate_cluster_cells(
            combined["cells"]
        )
        compatibility_horizons, compatibility_summary = (
            gradient_boosting.aggregate_brier(compatibility_cells)
        )
        output_tables = {
            "cells": compatibility_cells,
            "horizons": compatibility_horizons,
            "summary": compatibility_summary,
        }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destinations = []
    for name, frame in output_tables.items():
        destination = output_dir / f"{variant.table_prefix}_{name}.csv"
        _atomic_csv(frame, destination)
        destinations.append(destination)

    declared_model = Path(model_output or variant.model_output)
    decision_values = winner.to_dict()
    if variant.legacy_tables:
        parameter_id = winner_parameters.identifier
        compatibility_winner = output_tables["summary"].loc[
            output_tables["summary"]["parameter_id"] == parameter_id
        ]
        if len(compatibility_winner) != 1:
            raise ValueError("Compatibility summary does not contain the winner")
        decision_values = {
            **compatibility_winner.iloc[0].to_dict(),
            "configuration_id": winner["configuration_id"],
            "family": winner["family"],
            "specification": winner["specification"],
            "hyperparameters": winner["hyperparameters"],
            "random_seed": winner["random_seed"],
        }
    decision = pd.DataFrame(
        [
            {
                **decision_values,
                "screen_train_start": max(fold.train_start for fold in reverse_folds()),
                "screen_survivors": config.BOOSTING_SCREEN_SURVIVORS,
                "selection_metric": "mixture_adjusted_brier_equal_horizon_weight",
                "model_file": str(declared_model),
                "training_years": "2004-2007",
            }
        ]
    )
    decision_path = output_dir / f"{variant.table_prefix}_decision.csv"
    _atomic_csv(decision, decision_path)
    destinations.append(decision_path)
    return destinations


def fit_final_model(
    variant: BoostingVariant,
    decision_file: str | Path,
    *,
    data_dir: str | Path = config.SELECTION_DATA_DIR,
    model_output: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Refit a declared boosting winner on the frozen 2004--2007 sample."""
    destination = Path(model_output or variant.model_output)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Selected model already exists: {destination}; "
            "pass --overwrite to replace it"
        )
    decision = pd.read_csv(decision_file)
    if len(decision) != 1:
        raise ValueError("Selection decision must contain exactly one row")
    row = decision.iloc[0]
    if row.get("family") != FAMILY:
        raise ValueError("Selection decision has the wrong model family")
    if row.get("specification") != variant.specification:
        raise ValueError("Selection decision has the wrong boosting specification")
    parameters = parameters_from_row(row)
    if variant.expected_winner is not None and parameters != variant.expected_winner:
        raise ValueError("Selection decision does not match the frozen boosting winner")
    data = model_selection.load_selection_years(
        data_dir,
        config.TRAIN_YEARS,
        feature_set=variant.feature_set,
    )
    training = pd.concat([data[year] for year in config.TRAIN_YEARS], ignore_index=True)
    fitted, _ = fit_boosting_ratio_model(
        training,
        parameters,
        specification=variant.specification,
    )
    save_boosting_model(fitted, destination)
    return destination


def write_finalize_slurm(
    variant: BoostingVariant,
    *,
    destination: str | Path,
    repo_dir: str | Path,
    decision_file: str | Path,
    data_dir: str | Path,
    model_output: str | Path,
    worker_script: str,
    job_name: str,
    script_name: str,
    activate: str | None = None,
    account: str | None = "torch_pr_609_general",
    time_limit: str = "8:00:00",
    memory: str = "32G",
) -> Path:
    """Write a single-job Slurm script for a selected boosting refit."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    repo_dir = Path(repo_dir).resolve()
    script = destination / script_name
    cluster_tools.write_slurm_script(
        cluster_tools.SlurmJob(
            name=job_name,
            command=(
                "python",
                worker_script,
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


def parameters_from_row(row: pd.Series) -> BoostingParameters:
    """Restore one boosting parameter object from an aggregate table row."""
    values = json.loads(str(row["hyperparameters"]))
    return BoostingParameters(
        max_leaf_nodes=int(values["max_leaf_nodes"]),
        learning_rate=float(values["learning_rate"]),
        max_iter=int(values["max_iter"]),
        l2_regularization=float(values["l2_regularization"]),
        min_samples_leaf=int(values["min_samples_leaf"]),
    )


def validate_summary(
    summary: pd.DataFrame,
    variant: BoostingVariant,
    *,
    expected_candidates: int,
    expected_cells: int,
) -> None:
    """Reject incomplete or cross-variant boosting summaries."""
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
    if set(summary["specification"]) != {variant.specification}:
        raise ValueError("Boosting summary has the wrong specification")
    if (
        not (summary["n_horizons"] == 9).all()
        or not (summary["n_cells"] == expected_cells).all()
    ):
        raise ValueError("Boosting summary has an incomplete reverse design")


def _configurations(parameters, specification: str):
    return tuple(
        ModelConfiguration.from_mapping(
            FAMILY,
            specification,
            asdict(candidate),
            random_seed=config.BOOSTING_RANDOM_STATE,
        )
        for candidate in parameters
    )


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
