import json

import pandas as pd
import pytest

from hmda_seconds import config, gradient_boosting
from hmda_seconds.density_ratio.families.gradient_boosting import (
    HMDA_ONLY_SPECIFICATION,
)
from hmda_seconds.hmda_only_boosting_selection import (
    REFINEMENT_STAGE,
    SCREEN_STAGE,
    SURVIVOR_STAGE,
    refinement_jobs,
    screen_jobs,
    survivor_jobs,
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
