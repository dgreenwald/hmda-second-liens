# Step 9: Gradient-Boosting Challenger Findings

## Frozen design

The challenger is scikit-learn's `HistGradientBoostingClassifier`, trained with equal first-
and second-lien prior mass within each source year. It uses only the two continuous variables
and the native purchaser- and loan-type categories. Balanced-prior log odds are treated as the
log density ratio, and every target year receives its own mixture-share adjustment before
scoring. See `GRADIENT_BOOSTING_PROTOCOL.md` for the staged grid fixed before the run.

Every fitted fold/candidate model is persisted. The selected challenger is also refit on
2004--2007 and saved even though the results below do not automatically make it the primary
estimator.

## Selected boosted specification

The staged grid selects:

- 7 maximum leaf nodes;
- learning rate 0.05;
- 200 boosting iterations;
- L2 leaf regularization 10; and
- 1,000 minimum observations per leaf.

The choice of L2 is not substantively important. Holding the other settings fixed, selection
Brier is 0.026773 at L2 10, 0.026780 at L2 0, and 0.026787 at L2 1. Larger trees, a learning
rate of 0.1, and 400 iterations perform somewhat worse; 100 iterations clearly underfit.

## Reverse-time comparison

Metrics are averaged within horizon and then equally across horizons 1--9.

| Metric | Mixture logistic | Mixture boosting | Difference |
|---|---:|---:|---:|
| Brier score | 0.031779 | 0.026773 | -0.005006 |
| Log loss | 0.105932 | 0.092328 | -0.013604 |
| Mean probability minus observed share | -0.024513 | -0.015447 | +0.009066 |
| Calibration intercept | 0.740 | 0.533 | -0.208 |
| Calibration slope | 1.066 | 1.031 | -0.036 |

Boosting lowers Brier by 15.8% and log loss by 12.8%. It has lower Brier in all 45 individual
cells and at every backward horizon. The gain grows with backward distance: mean Brier falls by
0.00114 at horizon one and by 0.00723 at horizon nine. All target-share optimizers converge,
none reaches a boundary, and adjusted probability means equal fitted mixture shares to
numerical tolerance.

This is unusually consistent evidence that flexible nonlinearities improve the density-ratio
shape in the labeled backward-validation design.

## Forward-regime warning

The separate forward check fits on 2004--2007 and scores 2008--2016.

| Metric | Mixture logistic | Mixture boosting | Difference |
|---|---:|---:|---:|
| Brier score | 0.007307 | 0.011994 | +0.004687 |
| Log loss | 0.026763 | 0.039529 | +0.012766 |
| Mean probability minus observed share | -0.002665 | -0.011881 | -0.009216 |
| Calibration intercept | 0.133 | 1.734 | +1.601 |
| Calibration slope | 0.788 | 0.983 | +0.195 |

Boosting has worse Brier and log loss in every forward year. Its fitted annual shares average
1.19 percentage points below observed prevalence, compared with a 0.27-point shortfall for
logistic. The near-one calibration slope does not rescue the large level error: conditional
probability spread is reasonable after accounting for a substantial intercept miss.

The backward experiment is the primary design because the release is applied backward, and
its evidence strongly favors boosting. Nevertheless, the forward failure is relevant to
transportability. Histogram trees extrapolate by carrying terminal-leaf values beyond observed
split ranges, whereas the selected logistic spline has linear tails. The forward result is
therefore consistent with a boosted ratio that works well within the tested backward sequence
but adapts poorly to a sharply different covariate regime.

## Decision

Retain gradient boosting as a serious finalist and robustness estimator. It passes the stated
reverse-fold objective much more convincingly than a marginal challenger would: improvement is
universal across cells and becomes larger at long backward horizons. Do not automatically
replace logistic yet. The forward-regime failure and weaker extrapolation behavior warrant a
direct comparison of the resulting 1990--2003 series and support diagnostics before the final
Step 10 estimator choice.

## Outputs

- `output/tables/boosting_challenger_decision.csv`
- `output/tables/boosting_challenger_cells.csv`
- `output/tables/boosting_challenger_horizons.csv`
- `output/tables/boosting_challenger_comparison.csv`
- `output/tables/boosting_reverse_summary.csv`
- `output/tables/boosting_forward_summary.csv`
- `output/figures/boosting_calibration_reverse_horizons.pdf`
- `output/figures/boosting_calibration_forward_years.pdf`
- `output/model/boosting_challenger.pkl`
- `output/model/boosting_folds/*.pkl`
