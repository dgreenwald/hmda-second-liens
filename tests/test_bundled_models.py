import json
from importlib.resources import files

import numpy as np
import pandas as pd
import pytest

from hmda_seconds import load_benchmark
from hmda_seconds.benchmark_release import bundle_logistic_models, snapshot_payloads
from hmda_seconds.density_ratio import artifacts
from hmda_seconds.density_ratio.families.logistic import (
    fit_known_source_prior_model,
    known_source_prior_model_path,
)
from hmda_seconds.logistic_features import FeatureSpecification


@pytest.mark.parametrize("feature_set", ["core", "hmda_only"])
def test_bundled_predictions_are_complete_and_independent(feature_set):
    first = load_benchmark(feature_set)
    second = load_benchmark(feature_set)
    assert set(first.intercepts) == set(range(1990, 2017))
    inputs = {name: np.zeros(27) for name in first.required_columns}
    inputs["loan_type"][:] = 1
    years = np.arange(1990, 2017)
    result = first.predict_proba_second_lien(inputs, year=years)
    assert np.isfinite(result).all()
    assert ((result > 0) & (result < 1)).all()
    np.testing.assert_array_equal(
        first.predict(inputs, year=years), np.where(result >= 0.5, 2, 1)
    )
    for index, year in enumerate(years):
        single = {name: values[index : index + 1] for name, values in inputs.items()}
        np.testing.assert_allclose(
            first.predict_proba_second_lien(single, year=int(year)),
            result[index : index + 1],
            atol=1e-12,
            rtol=0,
        )
    first.coefficients[:] = 0
    first.intercepts[1990] = 100
    np.testing.assert_array_equal(
        second.predict_proba_second_lien(inputs, year=years), result
    )
    with pytest.raises(ValueError, match="saved intercept"):
        second.predict(inputs, year=1989)
    resource = files("hmda_seconds").joinpath("benchmarks", f"{feature_set}.json")
    payload = json.loads(resource.read_text())
    assert payload["provenance"]["source_files_sha256"]
    assert "Dropbox" not in resource.read_text()


def test_unknown_benchmark():
    with pytest.raises(ValueError, match="feature_set"):
        load_benchmark("rf")


@pytest.fixture
def snapshot(tmp_path):
    rng = np.random.default_rng(46)
    frame = pd.DataFrame(
        {
            "year": np.repeat(range(2004, 2008), 40),
            "lien_status": np.tile([1, 2], 80),
            "log_lti": rng.normal(size=160),
            "log_county_value_to_loan": rng.normal(size=160),
            "purchaser_type": np.tile(range(10), 16),
            "loan_type": np.tile(range(1, 5), 40),
        }
    )
    rows = []
    for feature_set, interaction, c in (
        ("core", "purchaser_type", 0.1),
        ("hmda_only", "none", 1.0),
    ):
        spec = FeatureSpecification("spline_lti", interaction, feature_set=feature_set)
        path = known_source_prior_model_path(
            range(2004, 2008),
            spec,
            c,
            tmp_path / "model_family_comparison/models/logistic" / spec.name,
        )
        model = fit_known_source_prior_model(frame, spec, c, model_file=path)
        metadata = artifacts.load_metadata(path)
        shard = {
            "fold": {"direction": "forward", "train_years": list(range(2004, 2008))},
            "models": [
                {
                    "artifact_path": str(path),
                    "model_id": model.model_id,
                    "configuration": metadata.configuration.to_dict(),
                }
            ],
        }
        directory = tmp_path / "model_family_comparison/shards"
        directory.mkdir(parents=True, exist_ok=True)
        (
            directory
            / f"model_family_comparison_forward__logistic__{spec.name}__train_2004_2007__test.json"
        ).write_text(json.dumps(shard))
        rows.extend(
            {
                "year": year,
                "predictor_set": "unrestricted"
                if feature_set == "core"
                else "hmda_only",
                "model_family": "logistic",
                "model_id": model.model_id,
                "mixture_share": 0.2,
                "optimizer_converged": True,
                "mixture_at_boundary": False,
                "n_model_sample": 160,
            }
            for year in range(1990, 2017)
        )
    (tmp_path / "tables").mkdir()
    pd.DataFrame(rows).to_csv(
        tmp_path / "tables/historical_model_family_annual.csv", index=False
    )
    directory = tmp_path / "historical_model_family/shards"
    directory.mkdir(parents=True)
    for year in range(1990, 2017):
        (directory / f"year_{year}.json").write_text(
            json.dumps(
                {"year": year, "annual": [row for row in rows if row["year"] == year]}
            )
        )
    return tmp_path


def test_snapshot_release_is_deterministic_and_preserves_source(snapshot, tmp_path):
    source = snapshot / "tables/historical_model_family_annual.csv"
    before = source.read_bytes()
    paths = bundle_logistic_models(
        snapshot, staging_dir=tmp_path / "stage", resource_dir=tmp_path / "resources"
    )
    assert len(paths) == 4
    assert paths[0].read_bytes() == paths[2].read_bytes()
    assert paths[1].read_bytes() == paths[3].read_bytes()
    assert source.read_bytes() == before
    assert (
        "not_asserted"
        in json.loads(paths[0].read_text())["provenance"]["annual_artifact_binding"]
    )


@pytest.mark.parametrize(
    "failure", ["identity", "duplicate", "missing", "share", "artifact"]
)
def test_snapshot_rejects_inconsistent_sources(snapshot, failure):
    csv = snapshot / "tables/historical_model_family_annual.csv"
    annual = pd.read_csv(csv)
    if failure == "identity":
        annual.loc[0, "model_id"] = "wrong"
    elif failure == "duplicate":
        annual = pd.concat([annual, annual.iloc[:1]])
    elif failure == "missing":
        annual = annual.iloc[1:]
    elif failure == "share":
        annual.loc[0, "mixture_share"] = 0.3
    else:
        path = next(
            (snapshot / "model_family_comparison/models/logistic").glob("*/*.pkl")
        )
        path.write_bytes(path.read_bytes() + b"corrupt")
    annual.to_csv(csv, index=False)
    with pytest.raises(ValueError):
        snapshot_payloads(snapshot)
