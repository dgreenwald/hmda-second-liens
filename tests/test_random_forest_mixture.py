import numpy as np
import pandas as pd
import pytest

from hmda_seconds import random_forest_mixture


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


def test_forest_features_have_stable_full_indicator_schema():
    frame = synthetic_frame()
    features, names = random_forest_mixture.forest_features(frame)
    subset_features, subset_names = random_forest_mixture.forest_features(
        frame.loc[frame["purchaser_type"] == 0]
    )

    assert features.shape[1] == 16
    assert names == subset_names
    assert subset_features.shape[1] == features.shape[1]


def test_forest_density_ratio_round_trip(tmp_path):
    training = pd.concat(
        [synthetic_frame(), synthetic_frame(year=2006, seed=29)],
        ignore_index=True,
    )
    model, diagnostics = random_forest_mixture.fit_forest_ratio_model(training)
    path = random_forest_mixture.forest_model_path((2005, 2006), tmp_path)
    random_forest_mixture.save_forest_model(model, path)
    restored = random_forest_mixture.load_forest_model(path)

    assert diagnostics["fit_seconds"] > 0
    assert restored.train_years == (2005, 2006)
    assert np.isfinite(restored.log_ratio(training)).all()
    assert restored.log_ratio(training) == pytest.approx(model.log_ratio(training))
