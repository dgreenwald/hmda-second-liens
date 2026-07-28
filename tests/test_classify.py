from hmda_seconds import classify, config, train


def _fit(training_frame, **rf_kwargs):
    return train.fit(training_frame, train_size=0.5, test_size=0.5, **rf_kwargs)


def test_classify_frame_adds_prediction_and_probability_columns(training_frame):
    rfw = _fit(training_frame, n_estimators=10, max_depth=3, random_state=0)

    out = classify.classify_frame(training_frame, rfw)

    assert config.PREDICTED_LABEL_VAR in out.columns
    assert config.PROB_SECOND_LIEN_VAR in out.columns
    assert set(out[config.PREDICTED_LABEL_VAR].unique()) <= {1, 2}
    assert out[config.PROB_SECOND_LIEN_VAR].between(0.0, 1.0).all()


def test_classify_frame_does_not_mutate_input(training_frame):
    rfw = _fit(training_frame, n_estimators=5, max_depth=2, random_state=0)
    original_columns = set(training_frame.columns)

    classify.classify_frame(training_frame, rfw)

    assert set(training_frame.columns) == original_columns
