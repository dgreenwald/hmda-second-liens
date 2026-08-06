import numpy as np
import pandas as pd
import pytest

from hmda_seconds import model_selection, random_forest_mixture
from hmda_seconds.density_ratio import artifacts


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
    metadata = artifacts.load_metadata(path, allow_legacy=False)

    assert diagnostics["fit_seconds"] > 0
    assert metadata.n_training == len(training)
    assert metadata.feature_names == model.feature_names
    assert restored.train_years == (2005, 2006)
    assert np.isfinite(restored.log_ratio(training)).all()
    assert restored.log_ratio(training) == pytest.approx(model.log_ratio(training))


def test_reverse_runner_translation_matches_shared_evaluation(tmp_path):
    training = pd.concat(
        [synthetic_frame(), synthetic_frame(year=2006, seed=29)],
        ignore_index=True,
    )
    target = synthetic_frame(year=2004, seed=41)
    data = {
        2004: target,
        2005: training.loc[training["year"] == 2005],
        2006: training.loc[training["year"] == 2006],
    }
    fold = model_selection.ReverseFold((2005, 2006), (2004,))

    translated = random_forest_mixture._reverse_metrics_from_runner(
        data, [fold], tmp_path / "models"
    ).iloc[0]
    fitted = random_forest_mixture.load_forest_model(
        random_forest_mixture.forest_model_path(
            fold.train_years, tmp_path / "models"
        )
    )
    direct = random_forest_mixture.evaluation.evaluate_target(
        random_forest_mixture.adapters.adapt_random_forest_model(fitted),
        target,
        fold,
        label_var="lien_status",
        second_lien_class=2,
    ).result

    assert translated["mixture_share"] == pytest.approx(direct.mixture_share)
    assert translated["brier_score"] == pytest.approx(direct.brier_score)
    assert translated["log_loss"] == pytest.approx(direct.log_loss)
