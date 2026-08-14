import json
import runpy
import sys
from pathlib import Path

import pandas as pd
import pytest

from hmda_seconds import config
from hmda_seconds.density_ratio import artifacts
from hmda_seconds.logistic_features import (
    HMDA_ONLY_FEATURE_SET,
    core_specifications,
    hmda_only_specifications,
)
from hmda_seconds.model_selection_cluster import (
    COARSE_STAGE,
    REFINEMENT_STAGE,
    coarse_jobs,
    execute_job,
    finalize_selection,
    read_manifest,
    read_shard,
    refinement_jobs,
    write_finalize_slurm,
    write_manifest,
    write_slurm_array,
)


def complete_coarse_summary() -> pd.DataFrame:
    rows = []
    for specification_index, specification in enumerate(core_specifications()):
        for c_index, regularization_c in enumerate(config.LOGISTIC_SELECTION_COARSE_C):
            rows.append(
                {
                    "specification": specification.name,
                    "continuous_form": specification.continuous_form,
                    "interactions": specification.interactions,
                    "geography": "none",
                    "regularization_c": regularization_c,
                    "selection_brier": 0.1 + specification_index + c_index,
                    "n_horizons": 9,
                    "n_cells": 45,
                }
            )
    return pd.DataFrame(rows)


def test_coarse_grid_has_one_ridge_path_per_specification_and_fold():
    jobs = coarse_jobs(data_dir="data", output_root="output")

    assert len(jobs) == 108
    assert sum(len(job.c_values) for job in jobs) == 432
    assert {job.stage for job in jobs} == {COARSE_STAGE}
    assert {job.specification for job in jobs} == {
        specification.name for specification in core_specifications()
    }
    assert {job.train_start for job in jobs} == set(range(2005, 2014))
    assert all(job.c_values == config.LOGISTIC_SELECTION_COARSE_C for job in jobs)


def test_hmda_only_coarse_grid_has_eight_specifications_and_nine_folds():
    jobs = coarse_jobs(
        data_dir="data",
        output_root="output",
        feature_set=HMDA_ONLY_FEATURE_SET,
    )

    assert len(jobs) == 72
    assert {job.specification for job in jobs} == {
        specification.name for specification in hmda_only_specifications()
    }
    assert {job.feature_set for job in jobs} == {HMDA_ONLY_FEATURE_SET}
    assert all("feature_set" in job.to_dict() for job in jobs)


def test_core_job_serialization_remains_backward_compatible():
    job = coarse_jobs(data_dir="data", output_root="output")[0]

    assert "feature_set" not in job.to_dict()


def test_refinement_grid_uses_adjacent_decades_for_each_specification():
    summary = complete_coarse_summary()
    jobs = refinement_jobs(summary, data_dir="data", output_root="output")

    assert len(jobs) == 108
    assert sum(len(job.c_values) for job in jobs) == 216
    assert {job.stage for job in jobs} == {REFINEMENT_STAGE}
    assert all(job.c_values == pytest.approx((1e-5, 1e-3)) for job in jobs)


def test_refinement_rejects_incomplete_coarse_results():
    summary = complete_coarse_summary().iloc[:-1]

    with pytest.raises(ValueError, match="12 specifications"):
        refinement_jobs(summary, data_dir="data", output_root="output")


def test_manifest_round_trip_requires_complete_grid(tmp_path):
    jobs = coarse_jobs(data_dir="$DATA/selection", output_root="$DATA/output")
    manifest = write_manifest(jobs, tmp_path / "jobs.json")

    assert read_manifest(manifest) == jobs
    values = json.loads(manifest.read_text())
    values["jobs"].pop()
    manifest.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="complete 108-job grid"):
        read_manifest(manifest)


def test_slurm_matches_conversion_generator_conventions(tmp_path):
    jobs = coarse_jobs(data_dir="data", output_root="output")
    manifest, script = write_slurm_array(
        jobs,
        destination=tmp_path / "slurm",
        repo_dir=tmp_path / "repo",
        activate="/cluster/venv/bin/activate",
        account="test-account",
        max_concurrent=7,
    )

    assert len(read_manifest(manifest)) == 108
    contents = script.read_text()
    destination = (tmp_path / "slurm").resolve()
    assert "#SBATCH --job-name=hmda-logistic-coarse" in contents
    assert "#SBATCH --account=test-account" in contents
    assert "#SBATCH --array=0-107%7" in contents
    assert f"#SBATCH --output={destination}/%x_%A_%a.out" in contents
    assert f"#SBATCH --error={destination}/%x_%A_%a.err" in contents
    assert "source /cluster/venv/bin/activate" in contents
    assert "python scripts/run_logistic_selection_job.py" in contents
    assert "/usr/bin/time" not in contents
    assert "sbatch " not in contents


def test_default_slurm_has_no_environment_specific_activation(tmp_path):
    _, script = write_slurm_array(
        coarse_jobs(data_dir="data", output_root="output"),
        destination=tmp_path / "slurm",
        repo_dir=Path("repo"),
    )

    contents = script.read_text()
    assert '\nsource "' not in contents
    assert "LABDIR" not in contents


def test_finalize_slurm_uses_explicit_pipeline_paths(tmp_path):
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
    assert "#SBATCH --job-name=hmda-logistic-final" in contents
    assert "#SBATCH --account=test-account" in contents
    assert "#SBATCH --time=12:00:00" in contents
    assert "#SBATCH --mem=24G" in contents
    assert f"#SBATCH --output={destination}/%x_%j.out" in contents
    assert "source /cluster/venv/bin/activate" in contents
    assert f"cd {(tmp_path / 'repo').resolve()}" in contents
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


def test_generator_defaults_follow_central_config(monkeypatch):
    script = (
        Path(__file__).parents[1] / "scripts" / "generate_logistic_selection_slurm.py"
    )
    parse_args = runpy.run_path(str(script))["parse_args"]
    monkeypatch.setattr(sys, "argv", [str(script), "--stage", "coarse"])

    args = parse_args()

    assert args.data_dir == config.SELECTION_DATA_DIR
    assert args.output_root == config.RAW_LOGISTIC_CLUSTER_DIR
    assert args.activate == config.SLURM_ACTIVATE
    assert args.account == config.SLURM_ACCOUNT
    assert args.time == config.SLURM_TIME
    assert args.memory == config.SLURM_MEMORY
    assert args.max_concurrent == config.SLURM_MAX_CONCURRENT


def test_finalize_generator_defaults_follow_central_config(monkeypatch):
    script = (
        Path(__file__).parents[1] / "scripts" / "generate_finalize_logistic_slurm.py"
    )
    parse_args = runpy.run_path(str(script))["parse_args"]
    monkeypatch.setattr(sys, "argv", [str(script)])

    args = parse_args()

    assert args.data_dir == config.SELECTION_DATA_DIR
    assert args.model_output == config.SELECTED_LOGISTIC_MODEL_FILE
    assert args.activate == config.SLURM_ACTIVATE
    assert args.account == config.SLURM_ACCOUNT
    assert args.time == config.SLURM_TIME
    assert args.memory == config.SLURM_MEMORY


def test_worker_publishes_models_and_resumes_from_immutable_shard(
    training_frame, tmp_path, monkeypatch
):
    frames = {year: training_frame.assign(year=year) for year in range(2004, 2009)}

    def fake_load(data_dir, years):
        return {year: frames[year] for year in years}

    monkeypatch.setattr(
        "hmda_seconds.model_selection_cluster.model_selection.load_selection_years",
        fake_load,
    )
    job = coarse_jobs(data_dir="unused", output_root=tmp_path)[0]

    path = execute_job(job)
    shard = read_shard(path)
    assert len(shard["cells"]) == 4
    assert len(shard["artifact_paths"]) == 4
    assert all(Path(path).exists() for path in shard["artifact_paths"])

    def unexpected_load(*args, **kwargs):
        raise AssertionError("completed job should not reload data")

    monkeypatch.setattr(
        "hmda_seconds.model_selection_cluster.model_selection.load_selection_years",
        unexpected_load,
    )
    assert execute_job(job) == path


def test_finalize_refits_declared_winner_on_2004_2007(
    training_frame, tmp_path, monkeypatch
):
    decision = tmp_path / "decision.csv"
    pd.DataFrame(
        [
            {
                "continuous_form": "linear",
                "interactions": "none",
                "regularization_c": 0.1,
            }
        ]
    ).to_csv(decision, index=False)
    frames = {year: training_frame.assign(year=year) for year in config.TRAIN_YEARS}
    monkeypatch.setattr(
        "hmda_seconds.model_selection_cluster.model_selection.load_selection_years",
        lambda data_dir, years: {year: frames[year] for year in years},
    )
    output = tmp_path / "selected.pkl"

    assert (
        finalize_selection(decision, data_dir="unused", model_output=output) == output
    )
    metadata = artifacts.load_metadata(output, allow_legacy=False)
    assert metadata.train_years == tuple(config.TRAIN_YEARS)
    assert metadata.configuration.specification == "linear__none"
    with pytest.raises(FileExistsError, match="--overwrite"):
        finalize_selection(decision, data_dir="unused", model_output=output)
