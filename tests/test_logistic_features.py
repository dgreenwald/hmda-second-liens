import numpy as np
import pytest

from hmda_seconds import logistic_features


def test_core_grid_crosses_all_predeclared_forms():
    specifications = logistic_features.core_specifications()

    assert len(specifications) == 12
    assert len({specification.name for specification in specifications}) == 12


def test_hmda_only_grid_excludes_county_value_feature(training_frame):
    specifications = logistic_features.hmda_only_specifications()

    assert len(specifications) == 8
    assert len({specification.name for specification in specifications}) == 8
    for specification in specifications:
        transformer = logistic_features.LogisticFeatureTransformer(specification)
        transformer.fit(training_frame.drop(columns=["log_county_value_to_loan"]))
        assert all("county_value" not in name for name in transformer.feature_names_)
        assert specification.name.startswith("hmda_only__")


def test_reference_coding_and_interaction_dimensions(training_frame):
    specification = logistic_features.FeatureSpecification("linear", "both")
    transformer = logistic_features.LogisticFeatureTransformer(specification)

    features = transformer.fit_transform(training_frame)

    # 2 continuous + 9 purchaser + 3 loan indicators, plus
    # 2 continuous x (9 purchaser + 3 loan) interactions.
    assert features.shape == (len(training_frame), 38)
    assert "purchaser_type_0" not in transformer.feature_names_
    assert "loan_type_1" not in transformer.feature_names_


def test_fold_scaling_is_not_refitted_on_validation(training_frame):
    specification = logistic_features.FeatureSpecification("linear", "none")
    transformer = logistic_features.LogisticFeatureTransformer(specification)
    transformer.fit(training_frame)
    shifted = training_frame.copy()
    shifted["log_lti"] += 10.0

    transformed = transformer.transform(shifted)

    assert transformed[:, 0].mean() > 5.0


def test_spline_basis_has_linear_right_tail():
    knots = np.array([-2.0, -0.5, 0.5, 2.0])
    values = np.array([3.0, 4.0, 5.0])

    basis = logistic_features.restricted_cubic_basis(values, knots)

    assert np.diff(basis, n=2, axis=0) == pytest.approx(0.0, abs=1e-12)


def test_spline_knots_come_from_training_frame(training_frame):
    specification = logistic_features.FeatureSpecification("spline_both", "none")
    transformer = logistic_features.LogisticFeatureTransformer(specification)
    transformer.fit(training_frame)

    expected = np.quantile(
        training_frame["log_lti"], logistic_features.SPLINE_QUANTILES
    )
    assert transformer.knots_["log_lti"] == pytest.approx(expected)


def test_purchaser_spline_interacts_every_lti_basis_column(training_frame):
    specification = logistic_features.FeatureSpecification(
        "spline_lti", "purchaser_type_spline_lti"
    )
    transformer = logistic_features.LogisticFeatureTransformer(specification)

    features = transformer.fit_transform(training_frame)

    # 3 LTI spline terms + 1 county-value term + 12 category indicators,
    # plus 3 LTI spline terms x 9 purchaser indicators and one linear
    # county-value term x 9 purchaser indicators.
    assert features.shape == (len(training_frame), 52)
    assert "log_lti_x_purchaser_type_1" not in transformer.feature_names_
    assert "log_lti_rcs_1_x_purchaser_type_1" in transformer.feature_names_
    assert "log_lti_rcs_3_x_purchaser_type_9" in transformer.feature_names_
    assert "log_county_value_to_loan_x_purchaser_type_1" in (transformer.feature_names_)


def test_purchaser_spline_interaction_requires_spline_main_effect():
    with pytest.raises(ValueError, match="require an LTI spline main effect"):
        logistic_features.FeatureSpecification("linear", "purchaser_type_spline_lti")


def test_geographic_challengers_are_reference_coded(training_frame):
    frame = training_frame.copy()
    frame["state_code"] = np.where(np.arange(len(frame)) % 2, 6, 36)
    specification = logistic_features.FeatureSpecification(
        "linear", "none", geography="region"
    )
    transformer = logistic_features.LogisticFeatureTransformer(specification)

    features = transformer.fit_transform(frame)

    assert features.shape[1] == 17
    assert "region_south" not in transformer.feature_names_
    assert "region_northeast" in transformer.feature_names_
    assert "region_west" in transformer.feature_names_


def test_unknown_category_is_rejected(training_frame):
    frame = training_frame.copy()
    frame.loc[0, "loan_type"] = 99
    specification = logistic_features.FeatureSpecification("linear", "none")

    with pytest.raises(ValueError, match="Unknown loan_type levels"):
        logistic_features.LogisticFeatureTransformer(specification).fit(frame)
