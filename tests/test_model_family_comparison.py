from types import SimpleNamespace

import pandas as pd
import pytest

from hmda_seconds import model_family_cluster, model_family_comparison
from hmda_seconds.density_ratio.protocols import EvaluationResult, ModelConfiguration
from hmda_seconds.density_ratio.shards import (
    ResultShard,
    ShardModel,
    shard_path,
    write_manifest,
    write_shard,
)


def configurations():
    return (
        ModelConfiguration.from_mapping("logistic", "linear__none", {"C": 0.1}),
        ModelConfiguration.from_mapping(
            "logistic", "hmda_only__linear__none", {"C": 1.0}
        ),
        ModelConfiguration.from_mapping(
            "hist_gradient_boosting",
            "primitive_continuous_and_native_categories",
            {"max_leaf_nodes": 7},
            random_seed=17,
        ),
        ModelConfiguration.from_mapping(
            "hist_gradient_boosting",
            "hmda_only_primitive_and_native_categories",
            {"max_leaf_nodes": 7},
            random_seed=17,
        ),
    )


def test_comparison_job_matrix_has_four_frozen_finalists_and_ten_fits(tmp_path):
    jobs = model_family_comparison.comparison_jobs(
        data_dir=tmp_path / "data",
        output_root=tmp_path / "results",
        configurations=configurations(),
    )

    assert len(jobs) == 40
    assert sum(job.fold.direction == "reverse" for job in jobs) == 36
    assert sum(job.fold.direction == "forward" for job in jobs) == 4
    assert {job.job.configurations[0] for job in jobs} == set(configurations())
    assert all(len(job.job.configurations) == 1 for job in jobs)


def test_aggregate_comparison_builds_matched_cells_and_weighted_summaries(tmp_path):
    output_root = tmp_path / "results"
    jobs = model_family_comparison.comparison_jobs(
        data_dir=tmp_path / "data",
        output_root=output_root,
        configurations=configurations(),
    )
    manifest = write_manifest(jobs, tmp_path / "jobs.json")
    for planned in jobs:
        configuration = planned.job.configurations[0]
        family_penalty = 0.0 if configuration.family == "logistic" else -0.01
        model_id = (
            f"{configuration.family}__{configuration.specification}"
            f"__train_{planned.fold.train_start}_{planned.fold.train_end}"
        )
        results = []
        for target_year in planned.fold.target_years:
            actual = 0.2
            mixture_share = 0.18
            results.append(
                EvaluationResult(
                    model_id=model_id,
                    fold_id=planned.fold.fold_id,
                    target_year=target_year,
                    horizon=planned.fold.horizon_for(target_year),
                    n_observations=100 + target_year,
                    actual_second_share=actual,
                    mixture_share=mixture_share,
                    mean_probability=mixture_share,
                    hard_share_050=0.1,
                    brier_score=0.1 + family_penalty,
                    log_loss=0.3 + family_penalty,
                    calibration_mean_error=mixture_share - actual,
                    calibration_intercept=0.2,
                    calibration_slope=1.0,
                    optimizer_converged=True,
                    mixture_at_boundary=False,
                )
            )
        shard = ResultShard(
            job=planned.job,
            fold=planned.fold,
            models=(
                ShardModel(
                    model_id=model_id,
                    configuration=configuration,
                    artifact_path=str(tmp_path / f"{model_id}.pkl"),
                ),
            ),
            results=tuple(results),
        )
        write_shard(shard, shard_path(planned.job))

    destinations = model_family_comparison.aggregate_comparison(
        manifest,
        output_dir=tmp_path / "tables",
        figure_dir=tmp_path / "figures",
    )

    assert len(destinations) == 7
    cells = pd.read_csv(tmp_path / "tables" / "model_family_comparison_cells.csv")
    summary = pd.read_csv(tmp_path / "tables" / "model_family_comparison_summary.csv")
    paired = pd.read_csv(
        tmp_path / "tables" / "model_family_comparison_paired_summary.csv"
    )
    assert len(cells) == 216
    assert set(summary["n_cells"]) == {9, 45}
    assert set(summary.loc[summary.evaluation_design == "reverse", "n_horizons"]) == {9}
    assert paired["boosting_minus_logistic_brier_score"].tolist() == pytest.approx(
        [-0.01] * 4
    )
    assert set(paired["n_cells_boosting_lower_brier"]) == {9, 45}
    assert (tmp_path / "figures" / "model_family_comparison_reverse.pdf").exists()


def test_cluster_submission_chains_aggregation_after_array(tmp_path, monkeypatch):
    prepared = model_family_cluster.PreparedComparisonRun(
        run_dir=tmp_path,
        manifest=tmp_path / "manifest.json",
        array_script=tmp_path / "array.slurm",
        aggregate_script=tmp_path / "aggregate.slurm",
    )
    monkeypatch.setattr(
        model_family_cluster.cluster_tools,
        "submit_slurm",
        lambda path: SimpleNamespace(job_id="101"),
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="202\n")

    monkeypatch.setattr(model_family_cluster.subprocess, "run", fake_run)

    payload = model_family_cluster.submit_run(prepared)

    assert payload["fit_array"] == {"job_id": "101", "dependency": None}
    assert payload["aggregate"]["dependency"] == "afterok:101"
    assert "--dependency=afterok:101" in calls[0]
    assert "--kill-on-invalid-dep=yes" in calls[0]
    assert (tmp_path / "submission.json").is_file()
