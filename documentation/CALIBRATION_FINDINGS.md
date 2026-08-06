# Logistic Calibration Findings

## Main conclusion

The selected logistic model discriminates liens well enough to retain, but its raw
probabilities do **not** have a temporally stable level. They should not presently be released
as transportable expected second-lien shares without an explicit calibration caveat or an
independent, defensible prevalence anchor.

This conclusion does not reverse model selection. Raw Brier remains the declared selection
criterion, and the selected core continues to outperform the tested feature alternatives.
Step 6 instead clarifies what the selected probabilities can support when transported across
different mortgage-market regimes.

## Reverse-validation results

Across the 45 reverse-validation cells, averaging first within backward horizon and then
equally across horizons:

| Statistic | Value |
|---|---:|
| Raw Brier | 0.065387 |
| Log loss | 0.193833 |
| Observed second-lien share | 13.63% |
| Mean predicted second-lien probability | 6.16% |
| Prediction minus observed share | -7.47 pp |
| Mean calibration intercept | 2.462 |
| Mean calibration slope | 1.163 |

The level error grows with backward distance. At horizon 1, the mean observed share is 8.88%
and the mean prediction is 7.58%, a 1.30-point shortfall. The shortfall reaches 11.80 points at
horizon 7 and remains 8.18 points at horizon 9. Calibration slopes are close to one at short
horizons but rise to approximately 1.17--1.26 at horizons 4--9. Thus, prevalence/intercept
transport is the dominant failure, accompanied by some deterioration in probability shape at
longer horizons.

This pattern has a direct economic interpretation. Later, predominantly post-crisis training
windows have much lower second-lien prevalence than the earlier boom-era validation years.
The model cannot infer that aggregate regime shift solely from the loan-level covariates.

## Forward-regime robustness

The direction reverses when the final 2004--2007 boom-era model is applied to 2008--2016. It
overpredicts the second-lien share in every year:

| Year | Observed | Mean predicted | Error |
|---:|---:|---:|---:|
| 2008 | 3.44% | 5.04% | +1.60 pp |
| 2009 | 1.74% | 3.34% | +1.60 pp |
| 2010 | 1.67% | 3.47% | +1.80 pp |
| 2011 | 1.86% | 3.84% | +1.98 pp |
| 2012 | 1.78% | 3.62% | +1.85 pp |
| 2013 | 1.69% | 3.37% | +1.68 pp |
| 2014 | 1.85% | 3.39% | +1.54 pp |
| 2015 | 1.78% | 3.16% | +1.37 pp |
| 2016 | 1.80% | 3.07% | +1.27 pp |

Forward calibration intercepts range from -2.05 to -1.12 and slopes from 0.73 to 0.94. The
reversed sign of the level error is strong evidence that a model trained in one regime does
not automatically recover the aggregate prevalence of another.

## Release implication

Keep loan-level probabilities because they retain ranking and uncertainty information. Do not
describe their unadjusted mean as an estimated pre-2004 second-lien count share without a clear
transportability caveat. Step 7 should separately evaluate whether the fixed 0.5 hard labels
produce useful classification and aggregate count-share behavior. No year-specific prevalence
matching, intercept shift, Platt scaling, or isotonic regression is applied here because the
pre-2004 target prevalence is unobserved.

The complete cell, horizon, and reliability-bin tables are the
`output/tables/logistic_calibration_*.csv` files. Standard and log-scale reliability diagrams
are the `output/figures/logistic_calibration_*.pdf` files. All outputs are aggregated and
contain no loan-level HMDA records.
