"""Frozen unrestricted gradient-boosting selection on cluster shards."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config
from .staged_boosting import CORE_VARIANT
from .staged_boosting import finalize_selection_tables as _finalize_tables
from .staged_boosting import fit_final_model as _fit_final_model
from .staged_boosting import refinement_jobs as _refinement_jobs
from .staged_boosting import screen_jobs as _screen_jobs
from .staged_boosting import survivor_jobs as _survivor_jobs
from .staged_boosting import write_finalize_slurm as _write_finalize_slurm

SCREEN_STAGE, SURVIVOR_STAGE, REFINEMENT_STAGE = CORE_VARIANT.stages
STAGES = CORE_VARIANT.stages


def screen_jobs(*, data_dir: str | Path, output_root: str | Path):
    """Return the frozen unrestricted structure-screen jobs."""
    return _screen_jobs(CORE_VARIANT, data_dir=data_dir, output_root=output_root)


def survivor_jobs(
    screen_summary: pd.DataFrame,
    *,
    data_dir: str | Path,
    output_root: str | Path,
):
    """Return the frozen unrestricted survivor jobs."""
    return _survivor_jobs(
        CORE_VARIANT,
        screen_summary,
        data_dir=data_dir,
        output_root=output_root,
    )


def refinement_jobs(
    survivor_summary: pd.DataFrame,
    *,
    data_dir: str | Path,
    output_root: str | Path,
):
    """Return the frozen unrestricted refinement jobs."""
    return _refinement_jobs(
        CORE_VARIANT,
        survivor_summary,
        data_dir=data_dir,
        output_root=output_root,
    )


def finalize_selection_tables(
    survivor_dir: str | Path,
    refinement_dir: str | Path,
    output_dir: str | Path,
    *,
    model_output: str | Path = config.SELECTED_BOOSTING_MODEL_FILE,
) -> list[Path]:
    """Write baseline-compatible combined tables and decision."""
    return _finalize_tables(
        CORE_VARIANT,
        survivor_dir,
        refinement_dir,
        output_dir,
        model_output=model_output,
    )


def fit_final_model(
    decision_file: str | Path,
    *,
    data_dir: str | Path = config.SELECTION_DATA_DIR,
    model_output: str | Path = config.SELECTED_BOOSTING_MODEL_FILE,
    overwrite: bool = False,
) -> Path:
    """Refit the frozen unrestricted winner on 2004--2007."""
    return _fit_final_model(
        CORE_VARIANT,
        decision_file,
        data_dir=data_dir,
        model_output=model_output,
        overwrite=overwrite,
    )


def write_finalize_slurm(
    *,
    destination: str | Path,
    repo_dir: str | Path,
    decision_file: str | Path = config.TABLE_DIR / "boosting_challenger_decision.csv",
    data_dir: str | Path = config.SELECTION_DATA_DIR,
    model_output: str | Path = config.SELECTED_BOOSTING_MODEL_FILE,
    activate: str | None = None,
    account: str | None = "torch_pr_609_general",
    time_limit: str = "8:00:00",
    memory: str = "32G",
) -> Path:
    """Write the unrestricted selected-model Slurm script."""
    return _write_finalize_slurm(
        CORE_VARIANT,
        destination=destination,
        repo_dir=repo_dir,
        decision_file=decision_file,
        data_dir=data_dir,
        model_output=model_output,
        worker_script="scripts/finalize_boosting.py",
        job_name="hmda-boost-final",
        script_name="finalize_boosting.slurm",
        activate=activate,
        account=account,
        time_limit=time_limit,
        memory=memory,
    )
