import json

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


def test_write_conversion_slurm_writes_manifest_and_capped_array(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    jobs = hmda_conversion.conversion_jobs(years=[2004, 2005], sources=["nara"])
    manifest, script = hmda_conversion.write_conversion_slurm(
        jobs,
        destination="output/slurm/hmda",
        repo_dir="$LABDIR/repo",
        data_dir="$PY_TOOLS_DATA_DIR/hmda",
        activate="/cluster/venv/bin/activate",
        max_concurrent=2,
    )
    assert json.loads(manifest.read_text()) == jobs
    contents = script.read_text()
    assert "#SBATCH --array=0-1%2" in contents
    assert '--job-index "${SLURM_ARRAY_TASK_ID}"' in contents
    assert '--data-dir "$PY_TOOLS_DATA_DIR/hmda"' in contents
    assert "sbatch " not in contents


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
