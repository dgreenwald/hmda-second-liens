from hmda_seconds import train


def test_fit_returns_evaluated_wrapper(training_frame):
    rfw = train.fit(
        training_frame,
        train_size=0.5,
        test_size=0.5,
        n_estimators=5,
        max_depth=2,
        random_state=0,
    )

    assert hasattr(rfw.rf, "feature_importances_")
    assert 0.0 <= rfw.err_rate <= 1.0
    assert len(rfw.predictions) == len(rfw.test_labels)


def test_fit_learns_the_separating_signal(training_frame):
    rfw = train.fit(
        training_frame,
        train_size=0.5,
        test_size=0.5,
        n_estimators=20,
        max_depth=4,
        random_state=0,
    )
    # log_lti is constructed to cleanly separate the two classes, so a
    # reasonably-sized forest should do much better than chance (0.5).
    assert rfw.err_rate < 0.2
