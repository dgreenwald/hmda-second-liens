import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hmda_seconds import mixture, plausibility, portable_export, portable_predict
from hmda_seconds.density_ratio import artifacts
from hmda_seconds.logistic_features import FeatureSpecification


@pytest.fixture(scope="module")
def fitted_export(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("portable_source")
    rng = np.random.default_rng(67)
    n = 800
    second = rng.random(n) < 0.25
    frame = pd.DataFrame(
        {
            "year": np.repeat(np.arange(2004, 2008), n // 4),
            "lien_status": np.where(second, 2, 1),
            "log_lti": rng.normal(second * 1.5, 1),
            "log_county_value_to_loan": rng.normal(second * 0.7, 1),
            "purchaser_type": np.tile(np.arange(10), n // 10),
            "loan_type": np.tile(np.repeat(np.arange(1, 5), 10), n // 40),
        }
    )
    model_file = tmp_path / "known.pkl"
    model = mixture.fit_known_source_prior_model(
        frame,
        FeatureSpecification("spline_lti", "purchaser_type"),
        0.1,
        model_file=model_file,
    )
    metadata = artifacts.load_metadata(model_file)
    annual_file = tmp_path / "annual.csv"
    rows = []
    for year, target in ((2000, frame.iloc[:300]), (2001, frame.iloc[300:])):
        estimate = mixture.estimate_mixture_share(model.log_ratio(target))
        rows.append(
            {
                "year": year,
                "mixture_share": estimate.share,
                "mixture_optimizer_converged": estimate.optimizer_converged,
                "mixture_at_boundary": estimate.at_boundary,
                "mixture_model_id": model.model_id,
                "mixture_model_sha256": metadata.payload_sha256,
            }
        )
    pd.DataFrame(rows).to_csv(annual_file, index=False)
    output = tmp_path / "portable"
    path = portable_export.export_portable_logistic(
        model_file,
        annual_file,
        output,
        years=(2000, 2001),
    )
    return model, frame, path, model_file, annual_file


@pytest.fixture
def exported(fitted_export, tmp_path):
    model, frame, path, model_file, annual_file = fitted_export
    shutil.copytree(path.parent, tmp_path / "portable")
    for source in (model_file, artifacts.metadata_path(model_file), annual_file):
        shutil.copy2(source, tmp_path / source.name)
    return (
        model,
        frame,
        tmp_path / "portable" / "model.json",
        tmp_path / model_file.name,
        tmp_path / annual_file.name,
    )


@pytest.fixture
def restricted_export(exported, tmp_path):
    frame = exported[1].drop(columns="log_county_value_to_loan")
    model_file = tmp_path / "restricted.pkl"
    model = mixture.fit_known_source_prior_model(
        frame,
        FeatureSpecification("spline_lti", "none", feature_set="hmda_only"),
        1.0,
        model_file=model_file,
    )
    digest = artifacts.load_metadata(model_file).payload_sha256
    rows = []
    for year, target in ((2000, frame.iloc[:300]), (2001, frame.iloc[300:])):
        estimate = mixture.estimate_mixture_share(model.log_ratio(target))
        rows.append(
            {
                "year": year,
                "predictor_set": "hmda_only",
                "model_family": "logistic",
                "model_id": model.model_id,
                "model_sha256": digest,
                "mixture_share": estimate.share,
                "optimizer_converged": estimate.optimizer_converged,
                "mixture_at_boundary": estimate.at_boundary,
            }
        )
    # Same years for other families must neither collide nor supply intercepts.
    rows.extend(
        [
            {**row, "model_family": "boosting", "mixture_share": 0.9}
            for row in rows.copy()
        ]
    )
    annual_file = tmp_path / "historical.csv"
    pd.DataFrame(rows).to_csv(annual_file, index=False)
    path = portable_export.export_portable_logistic(
        model_file,
        annual_file,
        tmp_path / "restricted",
        years=(2000, 2001),
        feature_set="hmda_only",
    )
    return model, frame, path, model_file, annual_file


def test_restricted_predictions_need_only_three_columns(restricted_export):
    model, frame, path, _, _ = restricted_export
    portable = portable_predict.load_model(path)
    assert portable.required_columns == ("log_lti", "purchaser_type", "loan_type")
    assert portable.coefficients.shape == (15,)
    target = frame[list(portable.required_columns)].copy()
    knots = model.transformer.knots_["log_lti"]
    target.loc[:5, "log_lti"] = [knots[0] - 10, *knots, knots[-1] + 10]
    payload = json.loads(path.read_text())
    years = np.resize([2000, 2001], len(target))
    expected = np.empty(len(target))
    for year in (2000, 2001):
        share = payload["annual"][str(year)]["mixture_share"]
        np.testing.assert_allclose(
            portable.predict_proba_second_lien(target, year=year),
            mixture.adjusted_probability(model.log_ratio(target), share),
            rtol=0,
            atol=1e-12,
        )
        mask = years == year
        expected[mask] = mixture.adjusted_probability(
            model.log_ratio(target.loc[mask]), share
        )
    actual = portable.predict_proba_second_lien(target.to_dict("list"), year=years)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(
        portable.predict(target, year=years), np.where(expected >= 0.5, 2, 1)
    )
    np.testing.assert_allclose(
        portable.predict_proba_second_lien(target.iloc[:5], year=years[:5]),
        actual[:5],
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        portable.predict_proba_second_lien(
            target.assign(log_county_value_to_loan=np.nan), year=years
        ),
        actual,
    )
    for column in portable.required_columns:
        with pytest.raises(ValueError):
            portable.predict(target.drop(columns=column), year=2000)
    malformed = copy.deepcopy(payload)
    malformed["specification"] = portable_predict.SPECIFICATION
    with pytest.raises(ValueError, match="feature ordering"):
        portable_predict.PortableLogisticModel(malformed)


def test_restricted_export_validates_feature_set_and_provenance(restricted_export):
    _, _, path, model_file, annual_file = restricted_export
    with pytest.raises(ValueError, match="core model"):
        portable_export.export_portable_logistic(
            model_file, annual_file, path.parent, years=(2000, 2001)
        )
    annual = pd.read_csv(annual_file)
    for invalid in (
        annual.drop(columns="model_sha256"),
        annual.assign(model_sha256="wrong"),
        annual.assign(predictor_set="unrestricted"),
    ):
        invalid.to_csv(annual_file, index=False)
        with pytest.raises(ValueError):
            portable_export.export_portable_logistic(
                model_file,
                annual_file,
                path.parent,
                years=(2000, 2001),
                feature_set="hmda_only",
            )


def test_restricted_defaults_do_not_overwrite_core():
    assert "hmda_only__spline_lti__none__c_1" in str(
        portable_export.default_model_path("hmda_only")
    )
    with pytest.raises(ValueError):
        portable_export.default_model_path("unknown")


def test_probabilities_match_original_and_do_not_depend_on_batch(exported):
    model, frame, path, _, _ = exported
    portable = portable_predict.load_model(path)
    payload = json.loads(path.read_text())
    target = frame.copy()
    knots = model.transformer.knots_["log_lti"]
    target.loc[:5, "log_lti"] = [knots[0] - 10, *knots, knots[-1] + 10]
    years = np.where(np.arange(len(frame)) % 2, 2000, 2001)
    expected = np.empty(len(frame))
    for year in (2000, 2001):
        mask = years == year
        expected[mask] = mixture.adjusted_probability(
            model.log_ratio(target.loc[mask]),
            payload["annual"][str(year)]["mixture_share"],
        )
    actual = portable.predict_proba_second_lien(target, year=years)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(
        portable.predict(target, year=years), np.where(expected >= 0.5, 2, 1)
    )
    np.testing.assert_allclose(
        portable.predict_proba_second_lien(target.iloc[:50], year=years[:50]),
        actual[:50],
        rtol=0,
        atol=1e-12,
    )
    chunked = np.concatenate(
        [
            portable.predict_proba_second_lien(
                target.iloc[i : i + 100], year=years[i : i + 100]
            )
            for i in range(0, len(target), 100)
        ]
    )
    np.testing.assert_allclose(actual, chunked, rtol=0, atol=1e-12)
    mapping = {
        column: target[column].to_numpy()
        for column in portable_predict.CONTINUOUS + tuple(portable_predict.CATEGORIES)
    }
    np.testing.assert_array_equal(
        actual, portable.predict_proba_second_lien(mapping, year=years)
    )


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p.update(format_version=99),
        lambda p: p.update(specification="linear__none"),
        lambda p: p["feature_names"].reverse(),
        lambda p: p["coefficients"].pop(),
        lambda p: p["knots"].reverse(),
        lambda p: p["raw_scale"].__setitem__(0, 0),
        lambda p: p["basis_scale"].__setitem__(0, float("nan")),
        lambda p: p["category_levels"]["loan_type"].reverse(),
        lambda p: p["annual"]["2000"].update(intercept=100),
        lambda p: p["annual"]["2000"].update(mixture_share=0),
        lambda p: p.pop("coefficients"),
    ],
)
def test_rejects_malformed_model(exported, change):
    payload = json.loads(exported[2].read_text())
    change(payload)
    with pytest.raises(ValueError):
        portable_predict.PortableLogisticModel(payload)


@pytest.mark.parametrize(
    "column,value",
    [
        ("log_lti", np.nan),
        ("log_county_value_to_loan", np.inf),
        ("purchaser_type", np.nan),
        ("purchaser_type", 10),
        ("loan_type", 1.5),
    ],
)
def test_rejects_bad_inputs(exported, column, value):
    portable = portable_predict.load_model(exported[2])
    frame = exported[1].copy()
    frame[column] = value
    with pytest.raises(ValueError):
        portable.predict(frame, year=2000)


def test_years_shapes_empty_and_missing_columns(exported):
    portable = portable_predict.load_model(exported[2])
    frame = exported[1]
    for year in (1999, 2000.5, np.nan, [2000], [[2000]]):
        with pytest.raises(ValueError):
            portable.predict(frame, year=year)
    with pytest.raises(ValueError, match="Missing input"):
        portable.predict(frame.drop(columns="log_lti"), year=2000)
    inputs = {
        name: frame[name].to_numpy()
        for name in (*portable_predict.CONTINUOUS, *portable_predict.CATEGORIES)
    }
    inputs["log_lti"] = inputs["log_lti"][:2]
    with pytest.raises(ValueError, match="aligned"):
        portable.predict(inputs, year=2000)
    assert portable.predict(frame.iloc[:0], year=2000).shape == (0,)


def test_stable_sigmoid_and_threshold(exported):
    payload = json.loads(exported[2].read_text())
    payload["coefficients"] = [0.0] * len(payload["coefficients"])
    payload["annual"] = {"2000": {"mixture_share": 0.5, "intercept": 0.0}}
    for offset, probability, label in (
        (0.0, 0.5, 2),
        (1000.0, 1.0, 2),
        (-1000.0, 0.0, 1),
    ):
        payload["density_ratio_offset"] = offset
        payload["annual"]["2000"]["intercept"] = offset
        portable = portable_predict.PortableLogisticModel(copy.deepcopy(payload))
        with np.errstate(over="raise"):
            assert np.all(
                portable.predict_proba_second_lien(exported[1], year=2000)
                == probability
            )
            assert np.all(portable.predict(exported[1], year=2000) == label)


@pytest.mark.parametrize(
    "change",
    [
        lambda df: df.assign(mixture_model_sha256="wrong"),
        lambda df: df.assign(mixture_model_id="wrong"),
        lambda df: df.drop(columns="mixture_model_sha256"),
        lambda df: df.assign(mixture_optimizer_converged=False),
        lambda df: df.assign(mixture_share=1),
        lambda df: df.iloc[:1],
        lambda df: pd.concat([df, df]),
    ],
)
def test_export_rejects_unusable_annual_estimates(exported, change):
    _, _, path, model_file, annual_file = exported
    change(pd.read_csv(annual_file)).to_csv(annual_file, index=False)
    original = path.read_bytes()
    with pytest.raises(ValueError):
        portable_export.export_portable_logistic(
            model_file,
            annual_file,
            path.parent,
            years=(2000, 2001),
        )
    assert path.read_bytes() == original


def test_provenance_controls_checkpoint_reuse():
    provenance = {
        "mixture_model_id": "model",
        "mixture_model_sha256": "digest",
        "raw_model_sha256": "raw",
    }
    annual = pd.DataFrame([{"year": 2000, **provenance}])
    assert plausibility._year_complete(annual, 2000, provenance)
    assert not plausibility._year_complete(annual, 2001, provenance)
    assert not plausibility._year_complete(
        annual, 2000, {**provenance, "mixture_model_sha256": "changed"}
    )
    assert not plausibility._year_complete(
        annual.drop(columns="mixture_model_id"), 2000, provenance
    )


@pytest.mark.parametrize("packaged", [False, True])
@pytest.mark.parametrize("fixture", ["exported", "restricted_export"])
def test_copied_module_has_no_training_dependencies(request, fixture, packaged):
    path = request.getfixturevalue(fixture)[2]
    code = """
import importlib.abc
import sys
class BlockTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = {"pandas", "scipy", "sklearn", "py_tools", "dotenv", "matplotlib", "pyarrow", "joblib", "threadpoolctl"}
        if sys.argv[3] == "False":
            blocked.add("hmda_seconds")
        if fullname.split(".")[0] in blocked:
            raise AssertionError("Forbidden dependency: " + fullname)
sys.meta_path.insert(0, BlockTrainingImports())
sys.path.insert(0, sys.argv[1])
if sys.argv[3] == "True":
    from hmda_seconds.portable_predict import load_model
else:
    from predict import load_model
model = load_model(sys.argv[2])
result = model.predict_proba_second_lien({
    "log_lti": [0.], "log_county_value_to_loan": [0.],
    "purchaser_type": [0], "loan_type": [1],
}, year=2000)
assert result.shape == (1,)
assert 0 <= result[0] <= 1
"""
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            code,
            str(
                Path(__file__).resolve().parents[1] / "src" if packaged else path.parent
            ),
            str(path),
            str(packaged),
        ],
        check=True,
        cwd=path.parent,
        capture_output=True,
        text=True,
    )
