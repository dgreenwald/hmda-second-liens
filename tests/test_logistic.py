import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from hmda_seconds import config, logistic
from hmda_seconds.density_ratio import artifacts


def test_fit_and_predict_learn_the_separating_signal(training_frame):
    model = logistic.fit(training_frame, random_state=0)

    pred = logistic.predict(model, training_frame)
    prob = logistic.predict_proba_second_lien(model, training_frame)

    assert set(np.unique(pred)) <= {1, 2}
    assert ((prob >= 0.0) & (prob <= 1.0)).all()
    accuracy = (pred == training_frame[config.LABEL_VAR].to_numpy()).mean()
    assert accuracy > 0.8


def test_fit_uses_configured_features_by_default(training_frame):
    model = logistic.fit(training_frame)

    expected_features = len(config.CONTINUOUS_VARS) + sum(
        len(config.CATEGORY_LEVELS[var]) for var in config.CATEGORY_VARS
    )
    assert model.n_features_in_ == expected_features


def test_predict_preserves_absent_category_columns(training_frame):
    model = logistic.fit(training_frame)
    one_category = training_frame.loc[
        (training_frame["purchaser_type"] == 0)
        & (training_frame["loan_type"] == 1)
    ]

    pred = logistic.predict(model, one_category)

    assert len(pred) == len(one_category)


def test_fit_allows_feature_ablation(training_frame):
    model = logistic.fit(training_frame, category_vars=["purchaser_type"])
    pred = logistic.predict(
        model, training_frame, category_vars=["purchaser_type"]
    )

    assert len(pred) == len(training_frame)


def test_save_load_round_trip(training_frame, tmp_path):
    fitted = logistic.fit(training_frame)
    outfile = tmp_path / "logistic.pkl"
    logistic.save(fitted, outfile)
    loaded = logistic.load(outfile)
    metadata = artifacts.load_metadata(outfile, allow_legacy=False)

    assert isinstance(loaded, LogisticRegression)
    assert metadata.n_training == len(training_frame)
    assert np.array_equal(
        logistic.predict(fitted, training_frame),
        logistic.predict(loaded, training_frame),
    )


def test_load_rejects_wrong_model_type(tmp_path):
    outfile = tmp_path / "not_logistic.pkl"
    import pickle

    with outfile.open("wb") as file:
        pickle.dump("not a model", file)

    with pytest.raises(TypeError, match="Expected LogisticRegression"):
        logistic.load(outfile)
