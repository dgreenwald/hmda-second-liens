# Logistic Model-Selection Findings

## Selected core

The frozen 12-specification grid selects `spline_lti__purchaser_type` with ridge penalty
`C=0.1`. The model has a common four-knot restricted-cubic-spline function of `log_lti`, a
linear `log_county_value_to_loan` effect, reference-coded purchaser and loan-type indicators,
and linear interactions between both continuous variables and purchaser type. Its raw Brier
score, averaged equally across backward horizons 1--9, is **0.065387**.

## Purchaser-specific LTI spline challenger

A focused post-selection challenger replaces the linear `log_lti` by purchaser adjustments
with interactions between every fold-fitted `log_lti` spline-basis term and purchaser type.
It retains the linear `log_county_value_to_loan` by purchaser interactions and uses the same
reverse folds, raw-Brier objective, and staged ridge search.

The challenger also selects `C=0.1`, but its equal-horizon Brier score is **0.065609**, which
is 0.000223 higher (worse) than the core. It improves 16 of 45 individual validation cells.
Its mean Brier is slightly better at horizon 1 by 0.000016, but worse at every horizon from 2
through 9. It improves no individual cells at horizons 6--9. The richer interaction is
therefore rejected, and the formal core estimator is unchanged.

Detailed results are in `output/tables/logistic_selection_spline_purchaser_*.csv`. These are
regenerable diagnostic outputs and do not contain loan-level HMDA records.

## Guarded geography results

Adding Census-region indicators to the selected core lowers equal-horizon Brier to 0.064427
and improves 44 of 45 cells. Adding state indicators selects `C=1000` and yields 0.065478,
slightly worse than the non-geographic core under equal-horizon weighting. As predeclared,
neither challenger automatically replaces the core: region's broad predictive gain must be
weighed against the risk that geography proxies for period-specific market structure before
the backward application to 1990--2003.
