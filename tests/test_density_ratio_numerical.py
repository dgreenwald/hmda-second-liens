import numpy as np
import pytest

from hmda_seconds.density_ratio import numerical


class FakeClassifier:
    def __init__(self, classes, probability):
        self.classes_ = np.asarray(classes)
        self.probability = np.asarray(probability)

    def predict_proba(self, features):
        assert len(features) == len(self.probability)
        return self.probability


def test_predict_class_probability_respects_classifier_class_order():
    classifier = FakeClassifier([2, 1], [[0.8, 0.2], [0.3, 0.7]])

    result = numerical.predict_class_probability(classifier, np.zeros((2, 1)), 2)

    np.testing.assert_array_equal(result, [0.8, 0.3])


def test_predict_class_probability_rejects_missing_or_misaligned_class():
    classifier = FakeClassifier([1, 2], [[0.2], [0.7]])

    with pytest.raises(ValueError, match="does not contain unique class"):
        numerical.predict_class_probability(classifier, np.zeros((2, 1)), 3)
    with pytest.raises(ValueError, match="does not align"):
        numerical.predict_class_probability(classifier, np.zeros((2, 1)), 2)


def test_finite_vector_validates_shape_emptiness_and_finiteness():
    np.testing.assert_array_equal(
        numerical.finite_vector([1, 2], "sample"), np.array([1.0, 2.0])
    )
    with pytest.raises(ValueError, match="nonempty one-dimensional"):
        numerical.finite_vector([], "sample")
    with pytest.raises(ValueError, match="nonempty one-dimensional"):
        numerical.finite_vector([[1.0]], "sample")
    with pytest.raises(ValueError, match="non-finite"):
        numerical.finite_vector([np.inf], "sample")


def test_log_mean_exp_is_stable_for_large_values():
    values = np.array([1_000.0, 1_001.0])
    expected = 1_000.0 + np.log(np.mean(np.exp([0.0, 1.0])))

    assert numerical.log_mean_exp(values) == pytest.approx(expected)
