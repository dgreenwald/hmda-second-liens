import numpy as np
import pandas as pd
import pytest

from hmda_seconds import gradient_boosting, model_selection


def synthetic_frame(n=2_000, year=2005, seed=17):
    rng = np.random.default_rng(seed)
    log_lti = rng.normal(size=n)
    purchaser = rng.integers(0, 4, size=n)
    second = rng.random(n) < 1 / (1 + np.exp(-(-1.5 + log_lti)))
    return pd.DataFrame(
        {
            "year": year,
            "lien_status": np.where(second, 2, 1),
            "log_lti": log_lti,
            "log_county_value_to_loan": rng.normal(size=n),
            "purchaser_type": purchaser,
            "loan_type": rng.integers(1, 5, size=n),
        }
    )


def test_boosting_model_round_trip_and_finite_log_ratio(tmp_path):
    training = pd.concat(
        [synthetic_frame(year=2005), synthetic_frame(year=2006, seed=29)],
        ignore_index=True,
    )
    parameters = gradient_boosting.BoostingParameters(
        max_leaf_nodes=7,
        learning_rate=0.1,
        max_iter=20,
        min_samples_leaf=20,
    )
    fitted, diagnostics = gradient_boosting.fit_boosting_ratio_model(
        training, parameters
    )
    path = gradient_boosting.boosting_model_path(
        (2005, 2006), parameters, tmp_path
    )
    gradient_boosting.save_boosting_model(fitted, path)
    restored = gradient_boosting.load_boosting_model(path)

    assert diagnostics["n_iter_fitted"] == 20
    assert restored.train_years == (2005, 2006)
    assert np.isfinite(restored.log_ratio(training)).all()
    assert restored.log_ratio(training) == pytest.approx(
        fitted.log_ratio(training)
    )


def test_target_evaluation_applies_mixture_adjustment():
    training = pd.concat(
        [synthetic_frame(year=2005), synthetic_frame(year=2006, seed=29)],
        ignore_index=True,
    )
    target = synthetic_frame(year=2004, seed=41)
    parameters = gradient_boosting.BoostingParameters(
        max_leaf_nodes=7,
        learning_rate=0.1,
        max_iter=20,
        min_samples_leaf=20,
    )
    fitted, diagnostics = gradient_boosting.fit_boosting_ratio_model(
        training, parameters
    )
    fold = model_selection.ReverseFold((2005, 2006), (2004,))

    result = gradient_boosting.evaluate_target_year(
        fitted, target, fold, diagnostics
    ).iloc[0]

    assert result["optimizer_converged"]
    assert result["mean_adjusted_probability"] == pytest.approx(
        result["mixture_share"], abs=1e-7
    )
    assert 0 < result["adjusted_brier"] < 1


def test_boosting_aggregation_weights_horizons_equally():
    parameter = gradient_boosting.BoostingParameters(7, 0.1)
    common = {
        "parameter_id": parameter.identifier,
        "max_leaf_nodes": parameter.max_leaf_nodes,
        "learning_rate": parameter.learning_rate,
        "max_iter": parameter.max_iter,
        "l2_regularization": parameter.l2_regularization,
        "min_samples_leaf": parameter.min_samples_leaf,
    }
    cells = pd.DataFrame(
        [
            {**common, "horizon": 1, "adjusted_brier": 0.1, "mixture_share_error": 0.0},
            {**common, "horizon": 1, "adjusted_brier": 0.3, "mixture_share_error": 0.0},
            {**common, "horizon": 2, "adjusted_brier": 0.8, "mixture_share_error": 0.2},
        ]
    )

    horizons, summary = gradient_boosting.aggregate_brier(cells)

    assert horizons.set_index("horizon").loc[1, "mean_brier"] == pytest.approx(0.2)
    assert summary.iloc[0]["selection_brier"] == pytest.approx(0.5)
