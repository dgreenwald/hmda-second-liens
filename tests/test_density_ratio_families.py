from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from hmda_seconds import (
    config,
    gradient_boosting,
    logistic_features,
    mixture,
    mixture_logistic_selection,
    random_forest_mixture,
)
from hmda_seconds.density_ratio import (
    DensityRatioFamily,
    FittedDensityRatioModel,
    ModelConfiguration,
)
from hmda_seconds.density_ratio.artifacts import metadata_path
from hmda_seconds.density_ratio.families import (
    GradientBoostingFamily,
    LogisticFamily,
    RandomForestFamily,
)


def synthetic_frame(n=500, seed=17):
    rng = np.random.default_rng(seed)
    year = rng.choice([2005, 2006], size=n)
    log_lti = rng.normal(size=n)
    loan_type = rng.integers(1, 5, size=n)
    probability = 1 / (1 + np.exp(-(-1.3 + log_lti + 0.15 * (loan_type == 2))))
    second = rng.random(n) < probability
    return pd.DataFrame(
        {
            "year": year,
            "lien_status": np.where(second, 2, 1),
            "log_lti": log_lti,
            "log_county_value_to_loan": rng.normal(size=n),
            "purchaser_type": rng.integers(0, 10, size=n),
            "loan_type": loan_type,
        }
    )


def test_logistic_family_reuses_path_and_matches_existing_fit(tmp_path, monkeypatch):
    training = synthetic_frame()
    specification = logistic_features.FeatureSpecification("linear", "none")
    configurations = [
        ModelConfiguration.from_mapping(
            "logistic", specification.name, {"C": regularization_c}
        )
        for regularization_c in (0.1, 1.0)
    ]
    calls = []
    original = mixture_logistic_selection.fit_candidate_path

    def recording_fit(training, specification, c_values):
        calls.append(tuple(c_values))
        return original(training, specification, c_values)

    monkeypatch.setattr(
        mixture_logistic_selection, "fit_candidate_path", recording_fit
    )
    family = LogisticFamily(tmp_path)
    fitted = family.fit_many(training, configurations, train_years=(2005, 2006))

    assert isinstance(family, DensityRatioFamily)
    assert calls == [(0.1, 1.0)]
    assert len(fitted) == 2
    assert all(isinstance(model, FittedDensityRatioModel) for model in fitted.values())
    direct, _ = original(training, specification, [0.1, 1.0])
    for model in fitted.values():
        c_value = model.fitted.regularization_c
        assert model.log_ratio(training) == pytest.approx(
            direct[c_value].log_ratio(training), abs=1e-12
        )
        path = mixture.known_source_prior_model_path(
            (2005, 2006), specification, c_value, tmp_path
        )
        assert path.exists()
        assert metadata_path(path).exists()


def test_gradient_boosting_family_matches_existing_fit(tmp_path):
    training = synthetic_frame()
    parameters = gradient_boosting.BoostingParameters(
        max_leaf_nodes=3,
        learning_rate=0.1,
        max_iter=8,
        l2_regularization=1.0,
        min_samples_leaf=10,
    )
    configuration = ModelConfiguration.from_mapping(
        "hist_gradient_boosting",
        "primitive_continuous_and_native_categories",
        asdict(parameters),
        random_seed=config.BOOSTING_RANDOM_STATE,
    )
    family = GradientBoostingFamily(tmp_path)
    fitted = next(
        iter(
            family.fit_many(
                training, [configuration], train_years=(2005, 2006)
            ).values()
        )
    )
    direct, _ = gradient_boosting.fit_boosting_ratio_model(training, parameters)

    assert isinstance(family, DensityRatioFamily)
    assert fitted.log_ratio(training) == pytest.approx(
        direct.log_ratio(training), abs=1e-12
    )
    path = gradient_boosting.boosting_model_path(
        (2005, 2006), parameters, tmp_path
    )
    assert path.exists() and metadata_path(path).exists()


def test_random_forest_family_matches_existing_fit(tmp_path):
    training = synthetic_frame(n=250)
    configuration = ModelConfiguration.from_mapping(
        "random_forest",
        "raw_continuous_and_full_one_hot_categories",
        {
            "max_depth": config.RF_KWARGS["max_depth"],
            "n_estimators": config.RF_KWARGS["n_estimators"],
        },
        random_seed=config.RF_KWARGS["random_state"],
    )
    family = RandomForestFamily(tmp_path)
    fitted = next(
        iter(
            family.fit_many(
                training, [configuration], train_years=(2005, 2006)
            ).values()
        )
    )
    direct, _ = random_forest_mixture.fit_forest_ratio_model(training)

    assert isinstance(family, DensityRatioFamily)
    assert fitted.log_ratio(training) == pytest.approx(
        direct.log_ratio(training), abs=1e-12
    )
    path = random_forest_mixture.forest_model_path((2005, 2006), tmp_path)
    assert path.exists() and metadata_path(path).exists()


def test_family_rejects_training_year_mismatch_before_fitting(tmp_path):
    configuration = ModelConfiguration.from_mapping(
        "logistic", "linear__none", {"C": 0.1}
    )
    with pytest.raises(ValueError, match="do not match"):
        LogisticFamily(tmp_path).fit_many(
            synthetic_frame(), [configuration], train_years=(2005,)
        )


def test_family_reuses_matching_saved_fit(tmp_path, monkeypatch):
    training = synthetic_frame()
    configuration = ModelConfiguration.from_mapping(
        "logistic", "linear__none", {"C": 0.1}
    )
    family = LogisticFamily(tmp_path)
    first = family.fit_many(
        training, [configuration], train_years=(2005, 2006)
    )

    def unexpected_fit(*args, **kwargs):
        raise AssertionError("saved fit should have been reused")

    monkeypatch.setattr(
        mixture_logistic_selection, "fit_candidate_path", unexpected_fit
    )
    second = family.fit_many(
        training, [configuration], train_years=(2005, 2006)
    )

    assert next(iter(second.values())).log_ratio(training) == pytest.approx(
        next(iter(first.values())).log_ratio(training)
    )
