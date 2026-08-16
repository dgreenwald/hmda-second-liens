import json
import runpy
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from hmda_seconds import config
from hmda_seconds.density_ratio.cluster import (
    COARSE_C_VALUES,
    FIRST_ORDER_FEATURE_SPECIFICATIONS,
    FIRST_ORDER_STAGE,
    INCUMBENT_C_VALUES,
    INCUMBENT_SPECIFICATION,
    PILOT_SPECIFICATIONS,
    configurations_from_json,
    expand_job_paths,
    first_order_logistic_jobs,
    pilot_jobs,
    write_slurm_array,
)
from hmda_seconds.density_ratio.shards import read_manifest


def test_pilot_manifest_has_simple_and_spline_heavy_jobs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    jobs = pilot_jobs(data_dir="$TEST_DATA/selection", output_root="$TEST_OUT")
    repo_dir = tmp_path / "repo"
    manifest, script = write_slurm_array(
        jobs,
        destination=Path("slurm"),
        repo_dir=repo_dir,
        activate="/cluster/venv/bin/activate",
    )

    restored = read_manifest(manifest)
    assert [item.job.specification for item in restored] == list(PILOT_SPECIFICATIONS)
    assert all(
        len(item.job.configurations) == len(COARSE_C_VALUES) for item in restored
    )
    assert all(item.job.train_years == (2013, 2014, 2015, 2016) for item in restored)
    contents = script.read_text()
    assert "#SBATCH --time=8:00:00" in contents
    assert "#SBATCH --mem=32G" in contents
    assert "#SBATCH --array=0-1" in contents
    assert "python scripts/run_density_ratio_job.py" in contents
    assert "/usr/bin/time" not in contents
    assert "sbatch " not in contents
    assert f"cd {repo_dir}" in contents


def test_manifest_paths_expand_only_at_execution(monkeypatch):
    monkeypatch.setenv("CLUSTER_ROOT", "/cluster/project")
    planned = pilot_jobs(
        data_dir="$CLUSTER_ROOT/data", output_root="$CLUSTER_ROOT/output"
    )[0]

    expanded = expand_job_paths(planned)

    assert dict(expanded.job.input_paths)["selection_data_dir"] == (
        "/cluster/project/data"
    )
    assert expanded.job.output_root == "/cluster/project/output"
    assert planned.job.output_root == "$CLUSTER_ROOT/output"


def test_first_order_grid_is_the_frozen_coordinate_neighborhood():
    jobs = first_order_logistic_jobs(data_dir="data", output_root="output")

    assert len(jobs) == 63
    assert sum(len(item.job.configurations) for item in jobs) == 81
    assert {item.job.stage for item in jobs} == {FIRST_ORDER_STAGE}
    assert {item.job.specification for item in jobs} == {
        INCUMBENT_SPECIFICATION,
        *FIRST_ORDER_FEATURE_SPECIFICATIONS,
    }
    starts_by_specification = {}
    for item in jobs:
        starts_by_specification.setdefault(item.job.specification, set()).add(
            item.fold.train_start
        )
        c_values = tuple(
            configuration.parameter_dict()["C"]
            for configuration in item.job.configurations
        )
        expected = (
            INCUMBENT_C_VALUES
            if item.job.specification == INCUMBENT_SPECIFICATION
            else (0.1,)
        )
        assert c_values == expected
    assert all(
        starts == set(range(2005, 2014)) for starts in starts_by_specification.values()
    )


def test_first_order_slurm_array_has_a_concurrency_cap(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    jobs = first_order_logistic_jobs(data_dir="data", output_root="output")

    manifest, script = write_slurm_array(
        jobs,
        destination=Path("slurm/first_order"),
        repo_dir=tmp_path,
        activate="/cluster/venv/bin/activate",
        max_concurrent=3,
    )

    assert len(read_manifest(manifest)) == 63
    assert "#SBATCH --array=0-62%3" in script.read_text()
    assert "sbatch " not in script.read_text()


@pytest.mark.parametrize(
    "script_name",
    ("generate_density_ratio_slurm.py", "generate_first_order_logistic_slurm.py"),
)
def test_density_ratio_generators_follow_central_config(script_name, monkeypatch):
    script = Path(__file__).parents[1] / "scripts" / script_name
    parse_args = runpy.run_path(str(script))["parse_args"]
    monkeypatch.setattr(sys, "argv", [str(script)])

    args = parse_args()

    assert args.repo_dir == Path(__file__).parents[1]
    assert args.data_dir == config.SELECTION_DATA_DIR
    assert args.output_root == config.OUTPUT_DIR / "density_ratio"
    assert args.activate == config.SLURM_ACTIVATE
    assert args.account == config.SLURM_ACCOUNT
    assert args.time == config.SLURM_TIME
    assert args.memory == config.SLURM_MEMORY


def test_configuration_json_supports_seed_and_rejects_non_objects():
    values = configurations_from_json(
        "hist_gradient_boosting",
        "primitive_continuous_and_native_categories",
        ['{"max_iter": 10, "random_seed": 17}'],
    )
    assert values[0].parameter_dict() == {"max_iter": 10}
    assert values[0].random_seed == 17
    with pytest.raises(TypeError, match="JSON object"):
        configurations_from_json("logistic", "linear__none", [json.dumps([1])])


def test_slurm_destination_may_be_absolute(tmp_path):
    jobs = pilot_jobs(data_dir="data", output_root="output")
    manifest, script = write_slurm_array(
        jobs,
        destination=tmp_path,
        repo_dir="/cluster/repo",
        activate="/cluster/venv/bin/activate",
    )

    assert manifest.parent == tmp_path
    assert f"--manifest {manifest}" in script.read_text()


def test_worker_explicit_arguments_override_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("HMDA_DENSITY_RATIO_STAGE", "environment-stage")
    monkeypatch.setenv("HMDA_DENSITY_RATIO_TRAIN_START", "2012")
    script = Path(__file__).parents[1] / "scripts" / "run_density_ratio_job.py"
    planned_job = runpy.run_path(str(script))["planned_job"]
    args = Namespace(
        manifest=None,
        job_index=None,
        stage="cli-stage",
        family="logistic",
        specification="linear__none",
        train_start=2013,
        output_root=tmp_path,
        data_dir=tmp_path / "data",
        configuration=['{"C": 0.1}'],
    )

    planned = planned_job(args)

    assert planned.job.stage == "cli-stage"
    assert planned.job.train_years == (2013, 2014, 2015, 2016)
