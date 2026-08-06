import numpy as np
import pandas as pd
import pytest

from hmda_seconds import logistic_features, mixture_logistic_selection


def synthetic_frame(n=1_000, year=2005, seed=17):
    rng = np.random.default_rng(seed)
    log_lti = rng.normal(size=n)
    second = rng.random(n) < 1 / (1 + np.exp(-(-1.5 + log_lti)))
    return pd.DataFrame(
        {
            "year": year,
            "lien_status": np.where(second, 2, 1),
            "log_lti": log_lti,
            "log_county_value_to_loan": rng.normal(size=n),
            "purchaser_type": rng.integers(0, 10, size=n),
            "loan_type": rng.integers(1, 5, size=n),
        }
    )


def test_candidate_path_retains_every_penalty():
    training = pd.concat(
        [synthetic_frame(), synthetic_frame(year=2006, seed=29)],
        ignore_index=True,
    )
    specification = logistic_features.FeatureSpecification("linear", "none")

    models, diagnostics = mixture_logistic_selection.fit_candidate_path(
        training, specification, [0.1, 1.0]
    )

    assert set(models) == {0.1, 1.0}
    assert set(diagnostics) == set(models)
    assert all(model.train_years == (2005, 2006) for model in models.values())
    assert all(np.isfinite(model.log_ratio(training)).all() for model in models.values())


def test_shared_grid_translation_matches_existing_cell_evaluation(tmp_path):
    training = pd.concat(
        [synthetic_frame(), synthetic_frame(year=2006, seed=29)],
        ignore_index=True,
    )
    target = synthetic_frame(year=2004, seed=41)
    data = {2004: target, 2005: training.loc[training["year"] == 2005], 2006: training.loc[training["year"] == 2006]}
    specification = logistic_features.FeatureSpecification("linear", "none")
    fold = mixture_logistic_selection.temporal_folds.temporal_fold(
        (2005, 2006), (2004,), direction="reverse"
    )
    direct_models, diagnostics = mixture_logistic_selection.fit_candidate_path(
        training, specification, [0.1]
    )
    direct = mixture_logistic_selection.evaluate_target(
        direct_models[0.1], target, fold, diagnostics[0.1]
    ).iloc[0]

    translated = mixture_logistic_selection.evaluate_grid(
        data,
        [fold],
        {specification: [0.1]},
        pd.DataFrame(),
        tmp_path / "cells.csv",
        tmp_path / "models",
    ).iloc[0]

    assert translated["mixture_share"] == pytest.approx(direct["mixture_share"])
    assert translated["brier_score"] == pytest.approx(direct["brier_score"])
    assert translated["log_loss"] == pytest.approx(direct["log_loss"])


def test_screen_survivors_force_incumbent():
    specifications = mixture_logistic_selection.candidate_specifications()
    incumbent = logistic_features.FeatureSpecification(
        "spline_lti", "purchaser_type"
    )
    rows = []
    for index, specification in enumerate(specifications):
        rows.append(
            {
                "specification": specification.name,
                "continuous_form": specification.continuous_form,
                "interactions": specification.interactions,
                "regularization_c": 0.1,
                "selection_brier": index / 100,
            }
        )
    summary = pd.DataFrame(rows)

    survivors = mixture_logistic_selection.select_screen_survivors(
        summary, incumbent, n_survivors=4
    )

    assert len(survivors) == 4
    assert survivors[0] == incumbent
    assert len(set(survivors)) == 4


def test_candidate_aggregation_weights_horizons_equally():
    common = {
        "specification": "linear__none",
        "continuous_form": "linear",
        "interactions": "none",
        "regularization_c": 1.0,
    }
    cells = pd.DataFrame(
        [
            {**common, "horizon": 1, "brier_score": 0.1, "log_loss": 0.2, "mixture_share_error": 0.0, "validation_year": 2004},
            {**common, "horizon": 1, "brier_score": 0.3, "log_loss": 0.4, "mixture_share_error": 0.0, "validation_year": 2005},
            {**common, "horizon": 2, "brier_score": 0.8, "log_loss": 0.9, "mixture_share_error": 0.2, "validation_year": 2004},
        ]
    )

    horizons, summary = mixture_logistic_selection.aggregate_candidates(cells)

    assert horizons.set_index("horizon").loc[1, "mean_brier"] == pytest.approx(0.2)
    assert summary.iloc[0]["selection_brier"] == pytest.approx(0.5)
