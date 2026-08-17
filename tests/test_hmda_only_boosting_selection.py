import json
import runpy
import sys
from pathlib import Path

import pandas as pd
import pytest

from hmda_seconds import config, gradient_boosting
from hmda_seconds.density_ratio import artifacts
from hmda_seconds.density_ratio.families.gradient_boosting import (
    HMDA_ONLY_SPECIFICATION,
    BoostingParameters,
    load_boosting_model,
)
from hmda_seconds.hmda_only_boosting_selection import (
    REFINEMENT_STAGE,
    SCREEN_STAGE,
    SURVIVOR_STAGE,
    fit_final_model,
    refinement_jobs,
    screen_jobs,
    survivor_jobs,
    write_finalize_slurm,
)


def _summary(parameters, n_cells):
    return pd.DataFrame(
        [
            {
                "configuration_id": f"candidate-{index}",
                "family": "hist_gradient_boosting",
                "specification": HMDA_ONLY_SPECIFICATION,
                "hyperparameters": json.dumps(
                    {
                        "max_leaf_nodes": item.max_leaf_nodes,
                        "learning_rate": item.learning_rate,
                        "max_iter": item.max_iter,
                        "l2_regularization": item.l2_regularization,
                        "min_samples_leaf": item.min_samples_leaf,
                    },
                    sort_keys=True,
                ),
                "selection_brier": 0.1 + index,
                "n_horizons": 9,
                "n_cells": n_cells,
            }
            for index, item in enumerate(parameters)
        ]
    )


def test_screen_is_one_six_candidate_latest_fold_job():
    jobs = screen_jobs(data_dir="data", output_root="output")

    assert len(jobs) == 1
    assert jobs[0].job.stage == SCREEN_STAGE
    assert jobs[0].fold.train_start == 2013
    assert len(jobs[0].job.configurations) == 6


def test_survivors_and_refinements_are_one_job_per_fold():
    screen = _summary(gradient_boosting.structure_grid(), n_cells=9)
    survivors = survivor_jobs(screen, data_dir="data", output_root="output")

    assert len(survivors) == 9
    assert {job.job.stage for job in survivors} == {SURVIVOR_STAGE}
    assert all(
        len(job.job.configurations) == config.BOOSTING_SCREEN_SURVIVORS
        for job in survivors
    )

    survivor_summary = _summary(
        gradient_boosting.structure_grid()[: config.BOOSTING_SCREEN_SURVIVORS],
        n_cells=45,
    )
    refinements = refinement_jobs(
        survivor_summary, data_dir="data", output_root="output"
    )
    assert len(refinements) == 9
    assert {job.job.stage for job in refinements} == {REFINEMENT_STAGE}
    assert all(len(job.job.configurations) == 4 for job in refinements)


def test_incomplete_screen_is_rejected():
    screen = _summary(gradient_boosting.structure_grid(), n_cells=9).iloc[:-1]

    with pytest.raises(ValueError, match="incomplete candidate set"):
        survivor_jobs(screen, data_dir="data", output_root="output")


def test_final_refit_uses_declared_winner_and_hmda_only_schema(
    training_frame, tmp_path, monkeypatch
):
    parameters = BoostingParameters(
        max_leaf_nodes=3,
        learning_rate=0.1,
        max_iter=2,
        l2_regularization=1.0,
        min_samples_leaf=2,
    )
    decision = tmp_path / "decision.csv"
    _summary([parameters], n_cells=45).to_csv(decision, index=False)
    frames = {year: training_frame.assign(year=year) for year in config.TRAIN_YEARS}

    def fake_load(data_dir, years, *, feature_set):
        assert feature_set == "hmda_only"
        return {year: frames[year] for year in years}

    monkeypatch.setattr(
        "hmda_seconds.hmda_only_boosting_selection."
        "model_selection.load_selection_years",
        fake_load,
    )
    output = tmp_path / "boosting_hmda_only_challenger.pkl"

    assert fit_final_model(decision, data_dir="unused", model_output=output) == output
    fitted = load_boosting_model(output)
    metadata = artifacts.load_metadata(output, allow_legacy=False)
    assert fitted.parameters == parameters
    assert fitted.specification == HMDA_ONLY_SPECIFICATION
    assert fitted.train_years == tuple(config.TRAIN_YEARS)
    assert metadata.train_years == tuple(config.TRAIN_YEARS)
    assert metadata.configuration.specification == HMDA_ONLY_SPECIFICATION
    assert metadata.feature_names == ("log_lti", "purchaser_type", "loan_type")
    with pytest.raises(FileExistsError, match="--overwrite"):
        fit_final_model(decision, data_dir="unused", model_output=output)


def test_finalize_slurm_uses_explicit_paths(tmp_path):
    script = write_finalize_slurm(
        destination=tmp_path / "slurm" / "final",
        repo_dir=tmp_path / "repo",
        decision_file=tmp_path / "output" / "tables" / "decision.csv",
        data_dir=tmp_path / "data" / "selection",
        model_output=tmp_path / "output" / "model" / "selected.pkl",
        activate="/cluster/venv/bin/activate",
        account="test-account",
        time_limit="12:00:00",
        memory="24G",
    )

    contents = script.read_text()
    destination = (tmp_path / "slurm" / "final").resolve()
    assert "#SBATCH --job-name=hmda-only-boost-final" in contents
    assert "#SBATCH --account=test-account" in contents
    assert "#SBATCH --time=12:00:00" in contents
    assert "#SBATCH --mem=24G" in contents
    assert f"#SBATCH --output={destination}/%x_%j.out" in contents
    assert "source /cluster/venv/bin/activate" in contents
    assert f"cd {(tmp_path / 'repo').resolve()}" in contents
    assert "python scripts/finalize_hmda_only_boosting.py" in contents
    assert (
        f"--decision {(tmp_path / 'output' / 'tables' / 'decision.csv').resolve()}"
        in contents
    )
    assert f"--data-dir {(tmp_path / 'data' / 'selection').resolve()}" in contents
    assert (
        f"--model-output {(tmp_path / 'output' / 'model' / 'selected.pkl').resolve()}"
        in contents
    )
    assert "--overwrite" not in contents


def test_finalize_generator_defaults_follow_central_config(monkeypatch):
    script = (
        Path(__file__).parents[1]
        / "scripts"
        / "generate_finalize_hmda_only_boosting_slurm.py"
    )
    parse_args = runpy.run_path(str(script))["parse_args"]
    monkeypatch.setattr(sys, "argv", [str(script)])

    args = parse_args()

    assert args.decision == config.TABLE_DIR / "hmda_only_boosting_decision.csv"
    assert args.data_dir == config.SELECTION_DATA_DIR
    assert args.model_output == config.HMDA_ONLY_SELECTED_BOOSTING_MODEL_FILE
    assert args.activate == config.SLURM_ACTIVATE
    assert args.account == config.SLURM_ACCOUNT
    assert args.time == config.SLURM_TIME
    assert args.memory == config.SLURM_MEMORY
