import json
import runpy
import sys
from pathlib import Path

import pandas as pd
import pytest

from hmda_seconds import config, gradient_boosting
from hmda_seconds.boosting_selection_cluster import (
    REFINEMENT_STAGE,
    SCREEN_STAGE,
    SURVIVOR_STAGE,
    finalize_selection_tables,
    refinement_jobs,
    screen_jobs,
    survivor_jobs,
    write_finalize_slurm,
)
from hmda_seconds.density_ratio.families.gradient_boosting import SPECIFICATION
from hmda_seconds.staged_boosting import CORE_FROZEN_WINNER


def _hyperparameters(parameters):
    return json.dumps(
        {
            "max_leaf_nodes": parameters.max_leaf_nodes,
            "learning_rate": parameters.learning_rate,
            "max_iter": parameters.max_iter,
            "l2_regularization": parameters.l2_regularization,
            "min_samples_leaf": parameters.min_samples_leaf,
        },
        sort_keys=True,
    )


def _summary(parameters, briers=None, *, n_cells=45):
    briers = briers or [0.1 + index for index in range(len(parameters))]
    return pd.DataFrame(
        [
            {
                "configuration_id": f"candidate-{item.identifier}",
                "family": "hist_gradient_boosting",
                "specification": SPECIFICATION,
                "hyperparameters": _hyperparameters(item),
                "random_seed": config.BOOSTING_RANDOM_STATE,
                "selection_brier": brier,
                "mean_log_loss": brier,
                "n_horizons": 9,
                "n_cells": n_cells,
            }
            for item, brier in zip(parameters, briers, strict=True)
        ]
    ).sort_values("selection_brier")


def _write_stage(directory, parameters, briers):
    directory.mkdir()
    summary = _summary(parameters, briers)
    cells = []
    horizons = []
    for _, candidate in summary.iterrows():
        common = {
            "configuration_id": candidate["configuration_id"],
            "family": candidate["family"],
            "specification": candidate["specification"],
            "hyperparameters": candidate["hyperparameters"],
            "random_seed": candidate["random_seed"],
        }
        cells.append(
            {
                **common,
                "model_id": f"model-{candidate['configuration_id']}",
                "fold_id": "reverse_2005_2008",
                "target_year": 2004,
                "horizon": 1,
                "n_observations": 100,
                "actual_second_share": 0.2,
                "mixture_share": 0.19,
                "mean_probability": 0.19,
                "hard_share_050": 0.18,
                "brier_score": candidate["selection_brier"],
                "log_loss": candidate["selection_brier"],
                "calibration_mean_error": -0.01,
                "calibration_intercept": 0.0,
                "calibration_slope": 1.0,
                "optimizer_converged": True,
                "mixture_at_boundary": False,
                "mixture_em_difference": 0.0,
                "train_start": 2005,
                "train_end": 2008,
            }
        )
        horizons.append(
            {
                **common,
                "horizon": 1,
                "mean_brier": candidate["selection_brier"],
                "mean_log_loss": candidate["selection_brier"],
                "mean_mixture_share": 0.19,
                "n_cells": 1,
            }
        )
    pd.DataFrame(cells).to_csv(directory / "density_ratio_cells.csv", index=False)
    pd.DataFrame(horizons).to_csv(directory / "density_ratio_horizons.csv", index=False)
    summary.to_csv(directory / "density_ratio_summary.csv", index=False)


def test_unrestricted_stages_use_frozen_core_specification():
    screen = screen_jobs(data_dir="data", output_root="output")

    assert len(screen) == 1
    assert screen[0].job.stage == SCREEN_STAGE
    assert screen[0].job.specification == SPECIFICATION
    assert screen[0].fold.train_start == 2013
    assert len(screen[0].job.configurations) == 6

    screen_summary = _summary(gradient_boosting.structure_grid(), n_cells=9)
    survivors = survivor_jobs(screen_summary, data_dir="data", output_root="output")
    assert len(survivors) == 9
    assert {item.job.stage for item in survivors} == {SURVIVOR_STAGE}
    assert all(len(item.job.configurations) == 2 for item in survivors)

    survivor_summary = _summary(gradient_boosting.structure_grid()[:2])
    refinements = refinement_jobs(
        survivor_summary, data_dir="data", output_root="output"
    )
    assert len(refinements) == 9
    assert {item.job.stage for item in refinements} == {REFINEMENT_STAGE}
    assert all(len(item.job.configurations) == 4 for item in refinements)


def test_finalization_preserves_legacy_tables_and_verifies_frozen_winner(tmp_path):
    base = gradient_boosting.BoostingParameters(7, 0.05)
    other = gradient_boosting.BoostingParameters(15, 0.05)
    refinements = gradient_boosting.refinement_grid(base)
    refinement_briers = [
        0.01 if item == CORE_FROZEN_WINNER else 0.2 + index
        for index, item in enumerate(refinements)
    ]
    survivor_dir = tmp_path / "survivors"
    refinement_dir = tmp_path / "refinement"
    output_dir = tmp_path / "tables"
    _write_stage(survivor_dir, [base, other], [0.1, 0.11])
    _write_stage(refinement_dir, refinements, refinement_briers)

    destinations = finalize_selection_tables(survivor_dir, refinement_dir, output_dir)

    assert {path.name for path in destinations} == {
        "boosting_challenger_cells.csv",
        "boosting_challenger_horizons.csv",
        "boosting_challenger_summary.csv",
        "boosting_challenger_decision.csv",
    }
    cells = pd.read_csv(output_dir / "boosting_challenger_cells.csv")
    decision = pd.read_csv(output_dir / "boosting_challenger_decision.csv").iloc[0]
    assert "parameter_id" in cells
    assert "adjusted_brier" in cells
    assert "configuration_id" not in cells
    assert decision["parameter_id"] == CORE_FROZEN_WINNER.identifier
    assert json.loads(decision["hyperparameters"])["l2_regularization"] == 10.0


def test_finalization_rejects_a_winner_that_differs_from_frozen_result(tmp_path):
    base = gradient_boosting.BoostingParameters(7, 0.05)
    other = gradient_boosting.BoostingParameters(15, 0.05)
    survivor_dir = tmp_path / "survivors"
    refinement_dir = tmp_path / "refinement"
    output_dir = tmp_path / "tables"
    _write_stage(survivor_dir, [base, other], [0.01, 0.11])
    _write_stage(
        refinement_dir,
        gradient_boosting.refinement_grid(base),
        [0.2, 0.21, 0.22, 0.23],
    )

    with pytest.raises(ValueError, match="does not match frozen result"):
        finalize_selection_tables(survivor_dir, refinement_dir, output_dir)
    assert not output_dir.exists()


def test_finalize_slurm_uses_explicit_unrestricted_paths(tmp_path):
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
    assert "#SBATCH --job-name=hmda-boost-final" in contents
    assert "python scripts/finalize_boosting.py" in contents
    assert "#SBATCH --account=test-account" in contents
    assert "#SBATCH --time=12:00:00" in contents
    assert "#SBATCH --mem=24G" in contents
    assert "--overwrite" not in contents


@pytest.mark.parametrize(
    "script_name",
    ("generate_boosting_selection_slurm.py", "generate_finalize_boosting_slurm.py"),
)
def test_generators_follow_central_config(script_name, monkeypatch):
    script = Path(__file__).parents[1] / "scripts" / script_name
    parse_args = runpy.run_path(str(script))["parse_args"]
    argv = [str(script)]
    if script_name == "generate_boosting_selection_slurm.py":
        argv.extend(["--stage", SCREEN_STAGE])
    monkeypatch.setattr(sys, "argv", argv)

    args = parse_args()

    assert args.data_dir == config.SELECTION_DATA_DIR
    assert args.activate == config.SLURM_ACTIVATE
    assert args.account == config.SLURM_ACCOUNT
    assert args.time == config.SLURM_TIME
    assert args.memory == config.SLURM_MEMORY
    if hasattr(args, "output_root"):
        assert args.output_root == config.BOOSTING_CLUSTER_DIR
        assert args.max_concurrent == config.SLURM_MAX_CONCURRENT
    else:
        assert args.model_output == config.SELECTED_BOOSTING_MODEL_FILE
