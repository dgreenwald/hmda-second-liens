from py_tools.econometrics.machine_learning import RandomForestWrapper

from hmda_seconds import train
from hmda_seconds.density_ratio import artifacts


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


def test_fit_full_uses_every_training_row(training_frame):
    rfw = train.fit_full(
        training_frame,
        n_estimators=5,
        max_depth=2,
        random_state=0,
    )

    assert len(rfw.train_labels) == len(training_frame)
    assert rfw.test_labels is None


def test_fit_full_writes_compatible_atomic_artifact(training_frame, tmp_path):
    path = tmp_path / "forest.pkl"

    train.fit_full(
        training_frame,
        outfile=str(path),
        n_estimators=5,
        max_depth=2,
        random_state=0,
    )
    restored = RandomForestWrapper(infile=str(path))
    metadata = artifacts.load_metadata(path, allow_legacy=False)

    assert restored.rf.n_estimators == 5
    assert metadata.n_training == len(training_frame)
    assert metadata.configuration.family == "legacy_random_forest"
