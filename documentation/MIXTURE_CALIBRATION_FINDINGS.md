# Known-Source-Prior Mixture Calibration Findings

## Frozen estimator and validation

These diagnostics use the frozen `spline_lti__purchaser_type` specification
with ridge `C=0.1`. No feature, regularization, threshold, or calibration
parameter was selected using these results. Each source year is reweighted to a
50/50 lien-status prior; the complete fitted log odds are used as the density-
ratio score, and the target-year intercept is set by the unlabeled mixture-share
likelihood.

The primary design contains all 45 reverse-validation cells and horizons 1-9.
The separate forward check trains on 2004-2007 and evaluates 2008-2016. Every
source-window model is saved under `output/model/mixture_folds/`.

## Reverse-validation result

Metrics are averaged within horizon and then equally across horizons.

| Metric | Raw logistic | Known-prior mixture |
| --- | ---: | ---: |
| Brier score | 0.0654 | 0.0318 |
| Log loss | 0.1938 | 0.1059 |
| Mean probability minus observed share | -0.0747 | -0.0245 |
| Calibration intercept | 2.462 | 0.740 |
| Calibration slope | 1.163 | 1.066 |

The mixture adjustment reduces Brier score by about 51 percent and log loss by
about 45 percent. It improves mean Brier at every horizon, and improves Brier in
32 of 45 individual cells. Log loss improves in 31 of 45 cells. The annual
share shortfall falls by about 67 percent but is not eliminated.

Calibration is strongest at short horizons. At horizon one, the adjusted mean
share error is -0.0013, the calibration intercept is -0.091, and the slope is
0.952. At horizons six through nine, share underprediction is 3.4-4.0 percentage
points and calibration intercepts are about 1.16. Thus performance degrades
gradually with backward distance rather than failing abruptly.

The reliability bins show that at long horizons the highest-probability decile
is underpredicted, while several middle bins are mildly overpredicted. The
mixture shift therefore repairs much of the annual level error but does not
remove all shape error.

## Forward-regime result

Across 2008-2016, with years equally weighted:

| Metric | Raw logistic | Known-prior mixture |
| --- | ---: | ---: |
| Brier score | 0.0123 | 0.0073 |
| Log loss | 0.0411 | 0.0268 |
| Mean probability minus observed share | 0.0163 | -0.0027 |

Both proper scores improve in every one of the nine forward years. The raw
model's systematic overprediction becomes modest underprediction. The adjusted
forward calibration intercept averages 0.133 and its slope averages 0.788, so
the level adjustment is effective but the probability spread remains imperfect.

## Tail and structural diagnostics

The target log-ratio distribution has a median near -4.47, an average 99th
percentile near 6.29, and an average 99.9th percentile near 10.78. On average,
0.17 percent of target observations have a log ratio above 10. With millions of
loans per cell, these are not isolated numerical accidents and they help
explain why exponential density-ratio moments are unstable.

The known-prior estimator matches weighted logistic score equations, not the
two exact density-ratio normalization moments. Its strong predictive
calibration therefore supports using it as a pseudo-likelihood probability and
annual-share estimator, but does not by itself validate a literal structural
interpretation as `f_1(x) / f_0(x)`.

## Decision

The Step 6 diagnostics support advancing the frozen known-source-prior mixture
estimator to threshold and subgroup diagnostics. Retain the raw probabilities
as a benchmark, report the remaining long-horizon underprediction, and avoid
further tuning on these 45 cells.
