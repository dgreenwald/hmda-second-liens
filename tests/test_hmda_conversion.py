import json
from subprocess import CompletedProcess

import pytest

from hmda_seconds import hmda_conversion


def test_conversion_jobs_make_one_job_per_supported_year_and_source():
    jobs = hmda_conversion.conversion_jobs(
        years=[2017, 2022], sources=["ffiec_three_year", "ffiec_snapshot"]
    )
    assert jobs == [
        {"year": 2017, "source": "ffiec_three_year"},
        {"year": 2022, "source": "ffiec_three_year"},
        {"year": 2017, "source": "ffiec_snapshot"},
        {"year": 2022, "source": "ffiec_snapshot"},
    ]


def test_conversion_jobs_reject_years_absent_from_selected_source():
    with pytest.raises(ValueError, match="1989"):
        hmda_conversion.conversion_jobs(years=[1989], sources=["cfpb"])


def test_all_source_expands_to_one_job_per_available_explicit_source():
    jobs = hmda_conversion.conversion_jobs(years=[2017], sources=["all"])

    assert jobs == [
        {"year": 2017, "source": "ffiec_three_year"},
        {"year": 2017, "source": "ffiec_snapshot"},
        {"year": 2017, "source": "cfpb"},
    ]
    assert all(job["source"] != "all" for job in jobs)


def test_write_conversion_slurm_writes_manifest_and_capped_array(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    jobs = hmda_conversion.conversion_jobs(years=[2004, 2005], sources=["nara"])
    manifest, script = hmda_conversion.write_conversion_slurm(
        jobs,
        destination="output/slurm/hmda",
        data_dir="$PY_TOOLS_DATA_DIR/hmda",
        activate="/cluster/venv/bin/activate",
        account="test-account",
        max_concurrent=2,
    )
    assert json.loads(manifest.read_text()) == jobs
    contents = script.read_text()
    assert "#SBATCH --array=0-1%2" in contents
    assert "#SBATCH --account=test-account" in contents
    expected_log_root = tmp_path / "output/slurm/hmda"
    assert f"#SBATCH --output={expected_log_root}/%x_%A_%a.out" in contents
    assert f"#SBATCH --error={expected_log_root}/%x_%A_%a.err" in contents
    assert "years=(2004 2005)" in contents
    assert "sources=(nara nara)" in contents
    assert 'year="${years[SLURM_ARRAY_TASK_ID]}"' in contents
    assert "python -m py_tools.datasets.hmda convert" in contents
    assert "/usr/bin/time" not in contents
    assert "scontrol" not in contents
    assert '--data-dir "$PY_TOOLS_DATA_DIR/hmda"' in contents
    assert "sbatch " not in contents


def test_default_slurm_has_no_environment_specific_setup(tmp_path):
    _, script = hmda_conversion.write_conversion_slurm(
        [{"year": 2003, "source": "nara"}], destination=tmp_path
    )

    contents = script.read_text()
    assert "\nsource \"" not in contents
    assert "LABDIR" not in contents
    assert "PY_TOOLS_DATA_DIR" not in contents
    assert "--data-dir" not in contents


def test_run_conversion_job_executes_only_selected_pair(tmp_path, monkeypatch):
    manifest = tmp_path / "jobs.json"
    manifest.write_text(json.dumps([{"year": 2004, "source": "nara"}, {"year": 2017, "source": "cfpb"}]))
    calls = []

    def fake_convert(year, **kwargs):
        calls.append((year, kwargs))
        return [tmp_path / "result.parquet"]

    monkeypatch.setattr(hmda_conversion.hmda, "convert_lar", fake_convert)
    output = hmda_conversion.run_conversion_job(
        manifest, 1, data_dir=tmp_path / "hmda", chunksize=25_000
    )
    assert output == tmp_path / "result.parquet"
    assert calls[0][0] == 2017
    assert calls[0][1]["source"] == "cfpb"
    assert calls[0][1]["chunksize"] == 25_000


def test_conversion_job_name_includes_year_and_source(tmp_path):
    manifest = tmp_path / "jobs.json"
    manifest.write_text(json.dumps([{"year": 2004, "source": "nara"}]))

    assert hmda_conversion.conversion_job_name(manifest, 0) == "hmda-2004-nara"


def test_submit_slurm_calls_sbatch_with_generated_script(tmp_path, monkeypatch):
    script = tmp_path / "jobs.slurm"
    script.write_text("#!/bin/bash\n")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return CompletedProcess(command, 0, "Submitted batch job 12345\n", "")

    monkeypatch.setattr(hmda_conversion.subprocess, "run", fake_run)

    assert hmda_conversion.submit_slurm(script) == "Submitted batch job 12345"
    assert calls == [
        (
            ["sbatch", str(script.resolve())],
            {"check": True, "capture_output": True, "text": True},
        )
    ]
