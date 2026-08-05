import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from scipy.stats import norm

from hmda_seconds import logistic_features, mixture


def test_mixture_likelihood_recovers_known_share_and_em_agrees():
    rng = np.random.default_rng(17)
    n = 40_000
    true_share = 0.18
    is_second = rng.random(n) < true_share
    values = rng.normal(0.0, 1.0, n)
    values[is_second] = rng.normal(2.0, 1.0, is_second.sum())
    log_ratio = norm.logpdf(values, 2.0, 1.0) - norm.logpdf(
        values, 0.0, 1.0
    )

    estimate = mixture.estimate_mixture_share(log_ratio)

    assert estimate.optimizer_converged
    assert estimate.em_converged
    assert estimate.share == pytest.approx(true_share, abs=0.01)
    assert estimate.em_share == pytest.approx(estimate.share, abs=1e-7)
    assert not estimate.at_boundary


def test_adjusted_probabilities_are_self_consistent_at_mle():
    rng = np.random.default_rng(29)
    values = np.r_[rng.normal(0, 1, 8_000), rng.normal(2, 1, 2_000)]
    log_ratio = 2 * values - 2

    estimate = mixture.estimate_mixture_share(log_ratio)
    adjusted = mixture.adjusted_probability(log_ratio, estimate.share)

    assert adjusted.mean() == pytest.approx(estimate.share, abs=1e-7)
    assert np.corrcoef(log_ratio, adjusted)[0, 1] > 0.8


def test_year_fixed_effect_fit_absorbs_different_source_priors(tmp_path):
    rng = np.random.default_rng(41)
    rows = []
    for year, share in ((2005, 0.1), (2006, 0.3)):
        n = 4_000
        second = rng.random(n) < share
        rows.append(
            pd.DataFrame(
                {
                    "year": year,
                    "lien_status": np.where(second, 2, 1),
                    "log_lti": rng.normal(second * 1.5, 1.0),
                    "log_county_value_to_loan": rng.normal(
                        second * 0.5, 1.0
                    ),
                    "purchaser_type": rng.integers(0, 10, n),
                    "loan_type": rng.integers(1, 5, n),
                }
            )
        )
    training = pd.concat(rows, ignore_index=True)
    specification = logistic_features.FeatureSpecification("linear", "none")

    fitted = mixture.fit_density_ratio_models(training, specification, 1.0)

    diagnostics = fitted.source_year_diagnostics
    assert len(diagnostics) == 2
    assert fitted.year_fixed_effect.mean_ratio_first == pytest.approx(1.0)
    assert np.isfinite(fitted.known_source_prior.log_ratio_offset)
    assert diagnostics["normalization_gap"].max() - diagnostics[
        "normalization_gap"
    ].min() < 0.25
    path = mixture.density_ratio_models_path(
        (2005, 2006), specification, 1.0, tmp_path
    )
    mixture.save_density_ratio_models(fitted, path)
    restored = mixture.load_density_ratio_models(path)
    assert restored.year_fixed_effect.log_ratio_offset == pytest.approx(
        fitted.year_fixed_effect.log_ratio_offset
    )


def test_prior_adjustment_is_an_intercept_shift():
    source_probability = np.array([0.1, 0.5, 0.9])
    source_share = 0.2
    target_share = 0.05
    log_ratio = np.log(source_probability / (1 - source_probability))
    log_ratio -= np.log(source_share / (1 - source_share))

    adjusted = mixture.adjusted_probability(log_ratio, target_share)

    expected = expit(
        np.log(source_probability / (1 - source_probability))
        + np.log(target_share / (1 - target_share))
        - np.log(source_share / (1 - source_share))
    )
    assert adjusted == pytest.approx(expected)
    assert np.all(np.diff(adjusted) > 0)


def test_known_source_prior_fold_model_round_trip(tmp_path):
    rng = np.random.default_rng(53)
    n = 1_000
    second = rng.random(n) < 0.2
    training = pd.DataFrame(
        {
            "year": np.where(np.arange(n) % 2, 2005, 2006),
            "lien_status": np.where(second, 2, 1),
            "log_lti": rng.normal(second, 1.0),
            "log_county_value_to_loan": rng.normal(0, 1.0, n),
            "purchaser_type": rng.integers(0, 10, n),
            "loan_type": rng.integers(1, 5, n),
        }
    )
    specification = logistic_features.FeatureSpecification("linear", "none")
    path = mixture.known_source_prior_model_path(
        (2005, 2006), specification, 0.1, tmp_path
    )

    fitted = mixture.fit_known_source_prior_model(
        training, specification, 0.1, model_file=path
    )
    restored = mixture.load_known_source_prior_model(path)

    assert path.exists()
    assert restored.train_years == (2005, 2006)
    assert restored.regularization_c == 0.1
    assert restored.log_ratio(training) == pytest.approx(
        fitted.log_ratio(training)
    )


def test_share_error_aggregation_weights_horizons_equally():
    cells = pd.DataFrame(
        {
            "train_start": [2005, 2006, 2006],
            "validation_year": [2004, 2004, 2005],
            "horizon": [1, 2, 1],
            "actual_second_share": [0.1, 0.1, 0.1],
            "raw_mean_probability": [0.1, 0.5, 0.1],
            "raw_hard_share": [0.1, 0.5, 0.1],
            "mixture_share_pooled": [0.1, 0.5, 0.1],
            "mixture_share_year_fixed_effect": [0.1, 0.5, 0.1],
            "mixture_share_known_source_prior": [0.1, 0.5, 0.1],
            "adjusted_hard_share_pooled": [0.1, 0.5, 0.1],
            "adjusted_hard_share_year_fixed_effect": [0.1, 0.5, 0.1],
            "adjusted_hard_share_known_source_prior": [0.1, 0.5, 0.1],
        }
    )

    horizons, summary = mixture.aggregate_share_errors(cells)

    raw_horizons = horizons.loc[horizons["estimator"] == "raw_mean_probability"]
    assert raw_horizons.set_index("horizon").loc[1, "mean_absolute_error"] == 0
    raw_summary = summary.loc[
        summary["estimator"] == "raw_mean_probability"
    ].iloc[0]
    assert raw_summary["selection_absolute_error"] == pytest.approx(0.2)


def test_invalid_share_is_rejected():
    with pytest.raises(ValueError, match="strictly between"):
        mixture.adjusted_probability(np.array([0.0]), 0.0)
