import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from hmda_seconds import historical_model_family
from hmda_seconds import historical_model_family_cluster as historical_cluster


class FakeRatioModel:
    def __init__(self, model_id, scale=1.0, transformer=None):
        self.model_id = model_id
        self.scale = scale
        self.transformer = transformer

    def log_ratio(self, frame):
        centered = frame["log_lti"].to_numpy() - frame["log_lti"].mean()
        return centered * self.scale


def fake_models():
    transformer = SimpleNamespace(
        raw_location_={"log_lti": 0.0, "log_county_value_to_loan": 1.0},
        raw_scale_={"log_lti": 1.0, "log_county_value_to_loan": 2.0},
        knots_={"log_lti": np.array([-1.0, -0.2, 0.3, 1.0])},
    )
    return {
        ("unrestricted", "logistic"): FakeRatioModel(
            "unrestricted_logistic", 1.0, transformer
        ),
        ("unrestricted", "boosting"): FakeRatioModel(
            "unrestricted_boosting", 1.2
        ),
        ("hmda_only", "logistic"): FakeRatioModel("hmda_logistic", 0.8),
        ("hmda_only", "boosting"): FakeRatioModel("hmda_boosting", 1.1),
    }


def frame(year=2004):
    n = 20
    return pd.DataFrame(
        {
            "year": [year] * n,
            "log_lti": np.linspace(-2, 2, n),
            "log_county_value_to_loan": np.linspace(-1, 3, n),
            "purchaser_type": pd.Categorical(
                np.arange(n) % 10, categories=range(10)
            ),
            "loan_type": pd.Categorical(
                np.arange(n) % 4 + 1, categories=range(1, 5)
            ),
            "lien_status": np.where(np.arange(n) % 5 == 0, 2, 1),
        }
    )


def test_evaluate_year_retains_only_aggregate_model_and_support_records():
    result = historical_model_family.evaluate_year(frame(), fake_models())

    assert result["year"] == 2004
    assert len(result["annual"]) == 4
    assert len(result["continuous_support"]) == 2
    assert len(result["categorical_support"]) == 14
    assert {row["actual_second_share"] for row in result["annual"]} == {0.2}
    assert all(row["n_model_sample"] == 20 for row in result["annual"])
    assert "probability_q99" in result["annual"][0]
    assert "log_ratio_q01" in result["annual"][0]
    assert "fraction_abs_z_gt_3" in result["continuous_support"][0]


def synthetic_shard(year):
    actual = None if year < 2004 else 0.2
    annual = []
    for predictor_set, family in historical_model_family.MODEL_KEYS:
        offset = 0.01 if family == "boosting" else 0.0
        share = 0.1 + (year - 1990) / 1000 + offset
        annual.append(
            {
                "year": year,
                "predictor_set": predictor_set,
                "model_family": family,
                "model_id": f"{predictor_set}_{family}",
                "n_model_sample": 100,
                "actual_second_share": actual,
                "mixture_share": share,
                "mean_probability": share,
                "hard_share_050": share / 2,
                "optimizer_converged": True,
                "optimizer_iterations": 10,
                "mixture_at_boundary": False,
                "mean_log_likelihood": -0.2,
                "em_converged": True,
                "em_iterations": 1,
                "mixture_em_difference": 1e-10,
                "probability_below_001": 0.0,
                "probability_above_099": 0.0,
                "share_error": None if actual is None else share - actual,
                "absolute_share_error": (
                    None if actual is None else abs(share - actual)
                ),
            }
        )
    continuous = []
    for variable in ("log_lti", "log_county_value_to_loan"):
        continuous.append(
            {
                "year": year,
                "variable": variable,
                "n": 100,
                "standardized_mean": (year - 2004) / 10,
                "standardized_sd": 1.0,
                "fraction_abs_z_gt_2": 0.05,
                "fraction_abs_z_gt_3": 0.01,
                "standardized_q01": -2.0,
                "standardized_q05": -1.5,
                "standardized_q50": 0.0,
                "standardized_q95": 1.5,
                "standardized_q99": 2.0,
                "fraction_below_outer_knot": 0.05 if variable == "log_lti" else None,
                "fraction_above_outer_knot": 0.05 if variable == "log_lti" else None,
            }
        )
    categorical = []
    for variable, levels in (("purchaser_type", range(10)), ("loan_type", range(1, 5))):
        levels = tuple(levels)
        for level in levels:
            categorical.append(
                {
                    "year": year,
                    "variable": variable,
                    "level": level,
                    "count": 100 // len(levels),
                    "share": 1 / len(levels),
                }
            )
    return {
        "schema_version": 1,
        "year": year,
        "annual": annual,
        "continuous_support": continuous,
        "categorical_support": categorical,
    }


def test_aggregate_historical_requires_and_combines_all_frozen_years(tmp_path):
    result_root = tmp_path / "results"
    manifest = historical_model_family.write_historical_manifest(
        tmp_path / "jobs.json",
        comparison_manifest=tmp_path / "comparison.json",
        output_root=result_root,
    )
    for year in range(1990, 2017):
        path = historical_model_family.historical_shard_path(result_root, year)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(synthetic_shard(year)))

    destinations = historical_model_family.aggregate_historical(
        manifest,
        output_dir=tmp_path / "tables",
        figure_dir=tmp_path / "figures",
        result_root=result_root,
    )

    annual = pd.read_csv(tmp_path / "tables" / "historical_model_family_annual.csv")
    paired = pd.read_csv(tmp_path / "tables" / "historical_model_family_paired.csv")
    boundary = pd.read_csv(tmp_path / "tables" / "historical_model_family_boundary.csv")
    envelope = pd.read_csv(
        tmp_path / "tables" / "historical_model_family_support_envelope.csv"
    )
    assert len(destinations) == 8
    assert len(annual) == 108
    assert len(paired) == 54
    assert len(boundary) == 4
    assert set(envelope["envelope_position"]) >= {"below"}
    assert np.allclose(
        paired["boosting_minus_logistic_mixture_share"], 0.01
    )


def test_cluster_workflow_uses_canonical_directory(tmp_path):
    orchestration = tmp_path / "slurm" / "historical_model_family"
    prepared = historical_cluster.prepare_workflow(
        repository_root=tmp_path,
        orchestration_dir=orchestration,
        comparison_manifest=tmp_path / "comparison.json",
        selection_data_dir=tmp_path / "selection",
        hmda_data_dir=tmp_path / "hmda",
        output_root=tmp_path / "results",
        table_dir=tmp_path / "tables",
        figure_dir=tmp_path / "figures",
        activate=None,
    )

    assert prepared.orchestration_dir == orchestration
    assert prepared.manifest == orchestration / "historical_model_family_jobs.json"
    assert prepared.array_script.is_file()
    assert prepared.aggregate_script.is_file()


def test_cluster_submission_chains_aggregation_after_year_array(
    tmp_path, monkeypatch
):
    prepared = historical_cluster.PreparedHistoricalWorkflow(
        orchestration_dir=tmp_path,
        manifest=tmp_path / "manifest.json",
        array_script=tmp_path / "array.slurm",
        aggregate_script=tmp_path / "aggregate.slurm",
    )
    monkeypatch.setattr(
        historical_cluster.cluster_tools,
        "submit_slurm",
        lambda path: SimpleNamespace(job_id="301"),
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="302\n")

    monkeypatch.setattr(historical_cluster.subprocess, "run", fake_run)

    payload = historical_cluster.submit_workflow(prepared)

    assert payload["year_array"] == {"job_id": "301", "dependency": None}
    assert payload["aggregate"]["dependency"] == "afterok:301"
    assert "--dependency=afterok:301" in calls[0]
    assert "--kill-on-invalid-dep=yes" in calls[0]
    assert (tmp_path / "submission.json").is_file()
