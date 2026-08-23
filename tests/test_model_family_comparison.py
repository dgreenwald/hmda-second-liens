from types import SimpleNamespace

import pandas as pd
import pytest

from hmda_seconds import model_family_cluster, model_family_comparison
from hmda_seconds.density_ratio import artifacts
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
                    artifact_path=str(output_root / "models" / f"{model_id}.pkl"),
                ),
            ),
            results=tuple(results),
        )
        artifacts.save_fitted_model(
            {"model_id": model_id},
            shard.models[0].artifact_path,
            model_id=model_id,
            configuration=configuration,
            train_years=planned.fold.train_years,
            counts=(100, 80, 20),
            feature_names=("log_lti",),
            weighting="equal_source_year_class_priors",
            source_prior="balanced",
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


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("optimizer_converged", False, "non-converged mixture estimate"),
        ("mixture_at_boundary", True, "mixture estimate at the boundary"),
    ],
)
def test_validate_cells_rejects_invalid_mixture_estimates(field, value, message):
    rows = []
    for predictor_set in ("unrestricted", "hmda_only"):
        for family in ("logistic", "boosting"):
            for design, count in (("reverse", 45), ("forward", 9)):
                for cell in range(count):
                    rows.append(
                        {
                            "evaluation_design": design,
                            "predictor_set": predictor_set,
                            "model_family": family,
                            "train_start": cell // 5,
                            "train_end": cell // 5 + 3,
                            "target_year": cell,
                            "horizon": cell % 9 + 1,
                            "n_observations": 100,
                            "actual_second_share": 0.2,
                            "mean_probability": 0.2,
                            "mixture_share": 0.2,
                            "optimizer_converged": True,
                            "mixture_at_boundary": False,
                        }
                    )
    cells = pd.DataFrame(rows)
    cells.loc[0, field] = value

    with pytest.raises(ValueError, match=message):
        model_family_comparison._validate_cells(cells)


def test_cluster_workflow_uses_one_regenerable_canonical_directory(tmp_path):
    orchestration_dir = tmp_path / "slurm" / "model_family_comparison"
    arguments = {
        "repository_root": tmp_path,
        "orchestration_dir": orchestration_dir,
        "data_dir": tmp_path / "data",
        "output_root": tmp_path / "comparison",
        "table_dir": tmp_path / "tables",
        "figure_dir": tmp_path / "figures",
        "activate": None,
        "configurations": configurations(),
    }

    first = model_family_cluster.prepare_workflow(**arguments)
    second = model_family_cluster.prepare_workflow(**arguments)

    assert first == second
    assert first.orchestration_dir == orchestration_dir
    assert first.manifest == orchestration_dir / "density_ratio_jobs.json"
    assert first.array_script == orchestration_dir / "density_ratio_jobs.slurm"
    assert first.aggregate_script == (
        orchestration_dir / "aggregate_model_family_comparison.slurm"
    )


def test_cluster_submission_chains_aggregation_after_array(tmp_path, monkeypatch):
    prepared = model_family_cluster.PreparedComparisonWorkflow(
        orchestration_dir=tmp_path,
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

    payload = model_family_cluster.submit_workflow(prepared)

    assert payload["fit_array"] == {"job_id": "101", "dependency": None}
    assert payload["aggregate"]["dependency"] == "afterok:101"
    assert "--dependency=afterok:101" in calls[0]
    assert "--kill-on-invalid-dep=yes" in calls[0]
    assert (tmp_path / "submission.json").is_file()
