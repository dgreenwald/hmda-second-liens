# Step 10 historical model-family findings

## Scope and integrity

This document reports the frozen Step 10 application of the four 2004--2007
equal-source-prior finalists to the common 1990--2016 historical sample. It combines the
unrestricted and HMDA-only predictor sets with ridge logistic and histogram gradient boosting.
Each model estimates its own annual target mixture share. The pre-2004 outcomes remain
unobserved, so the historical comparison is evidence about agreement, extrapolation, and
plausibility rather than direct validation.

All 108 planned `(year, predictor set, family)` cells are present. Every direct and EM mixture
optimization converges, none reaches a boundary, and the largest absolute optimizer-versus-EM
difference is `5.35e-9`. Mean adjusted probabilities agree with fitted mixture shares to the
same tolerance. All four models use identical observations within each year.

Three years contain a small number of otherwise-clean records with missing purchaser type.
The common finite-feature restriction excludes 700 of 1,664,933 clean rows in 1990 (0.042%),
195 of 1,662,241 in 1991 (0.012%), and three of 4,189,285 in 1999 (less than 0.001%). The
restriction is applied before all four models and is too small to drive the comparisons below.

## Historical aggregate paths

| Predictor set | Family | Mean, 1990--2003 | Minimum | Maximum | 1990 | 2003 |
|---|---|---:|---:|---:|---:|---:|
| Unrestricted | Logistic | 5.87% | 2.68% (1993) | 9.28% (2000) | 4.07% | 9.14% |
| Unrestricted | Boosting | 5.78% | 2.72% (1993) | 9.16% (2003) | 2.94% | 9.16% |
| HMDA-only | Logistic | 7.09% | 3.60% (1993) | 10.21% (2000) | 7.10% | 9.47% |
| HMDA-only | Boosting | 6.76% | 3.41% (1993) | 9.81% (2000) | 5.71% | 9.41% |

The unrestricted logistic and boosting paths are highly correlated (`0.991`). Their mean
absolute annual difference is 0.20 percentage point over 1990--2003 and 0.13 point after 1991.
The largest difference is in 1990, when boosting is 1.13 points lower. From 1992 onward their
largest annual difference is 0.29 point. Boosting is above logistic in eight of 14 unrestricted
years and below it in six, so there is no persistent post-1991 family ordering.

The HMDA-only paths are also highly correlated (`0.988`), but boosting is below logistic in all
14 pre-2004 years. Their mean absolute difference is 0.33 point and the 1990 maximum is 1.39
points. Family choice therefore matters somewhat more when the county-value predictor is
removed.

Predictor-set choice matters more than family choice. The unrestricted estimate is below the
corresponding HMDA-only estimate in every pre-2004 year. The mean gap is 0.99 point for boosting
and 1.22 points for logistic; in 1990 it reaches 2.77 and 3.03 points, respectively. This is
consistent with the common reverse-validation evidence that the county-value predictor carries
important information for backward transport. The HMDA-only paths should consequently be read
as portability checks, not alternative primary series.

## Reporting-boundary and labeled-period evidence

| Predictor set | Family | 2003 | 2004 | 2004 minus 2003 | 2004 minus actual |
|---|---|---:|---:|---:|---:|
| Unrestricted | Logistic | 9.14% | 13.23% | +4.09 pp | -0.31 pp |
| Unrestricted | Boosting | 9.16% | 13.49% | +4.33 pp | -0.05 pp |
| HMDA-only | Logistic | 9.47% | 13.00% | +3.53 pp | -0.54 pp |
| HMDA-only | Boosting | 9.41% | 13.36% | +3.95 pp | -0.18 pp |

All four methods show a large 2003--2004 increase, while all are close to the observed 2004
share of 13.54%. The boundary jump is therefore not unique to one model family or to inclusion
of the county-value predictor. It remains evidence of a feature-distribution or sample-regime
change, not an estimable correction to the pre-2004 series.

Boosting has slightly lower mean absolute annual share error in the 2004--2007 source period:
0.35 versus 0.37 point for unrestricted models and 0.33 versus 0.44 point for HMDA-only models.
The later labeled period reverses that result decisively. Over 2008--2016, unrestricted
logistic has mean absolute share error of 0.27 point versus 1.21 points for boosting; HMDA-only
logistic has 0.16 point versus 1.01 points. Logistic has the lower share error in every one of
the nine forward years for both predictor sets. This matches the previously reported Brier and
log-loss comparison rather than introducing a metric-specific disagreement.

## Support and extrapolation diagnostics

The historical sample is not outside source support in a simple all-or-nothing sense, but its
distribution differs materially from 2004--2007.

- The standardized mean of `log_county_value_to_loan` is below the source-year envelope in all
  14 pre-2004 years, and its standardized dispersion is also below that envelope in all 14.
  Its 95th percentile is below the source envelope in every year. This indicates a lower and
  more compressed historical county-value-to-loan distribution on the frozen source scale.
- The `log_lti` standardized mean is within the source envelope in 11 of 14 years, but its
  dispersion is below the envelope in 13. Its upper-knot exceedance share is below the source
  envelope in 13 years. Tail behavior is most unusual in 1990: 2.24% of observations lie beyond
  three source standard deviations versus 0.76--0.96% in the four source years.
- Across all reported continuous summary-statistic cells, 90 of 126 county-value cells and 115
  of 154 `log_lti` cells fall outside the 2004--2007 annual envelope. These counts involve
  correlated summaries and should not be interpreted as hypothesis tests; they show that
  historical transport is consequential rather than nearly in-support replication.
- Categorical composition shifts are at least as prominent. Across the 14 pre-2004 years, 48
  of 56 loan-type level-year shares and 101 of 140 purchaser-type level-year shares lie outside
  their corresponding 2004--2007 envelopes. In 1990, loan type 1 is 71.95% versus a source
  range of 89.32--92.73%, loan type 2 is 22.78% versus 5.06--7.98%, and purchaser type 0 is
  42.30% versus 22.70--27.70%. Even in 2003, ten of the 14 categorical shares remain outside
  their source envelopes.

Adjusted probabilities are highly concentrated near zero, as expected for low fitted annual
shares: on average, 82.46% of unrestricted logistic probabilities and 85.97% of unrestricted
boosting probabilities are below 0.01 in 1990--2003. The corresponding HMDA-only fractions are
78.75% and 81.82%. Boosting produces no probabilities above 0.99, whereas unrestricted
logistic places an average 0.54% above 0.99 and reaches 1.31% in 2000. The bounded boosting
tail and longer logistic tail are visible model-family extrapolation differences, but neither
causes mixture instability or boundary estimates.

## Final Step 10 decision

Retain the unrestricted ridge-logistic density-ratio model as the primary estimator for annual
shares and mixture-adjusted probabilities. Retain unrestricted boosting as the principal
model-family robustness series, and retain both HMDA-only variants as portability checks.

The decision does not dismiss the reverse-time evidence: boosting improves Brier in all 45
reverse cells and is a serious challenger. Step 10 shows, however, that boosting and logistic
produce nearly the same unrestricted historical aggregate path after the first two years. The
unlabeled path therefore provides no direct accuracy evidence that resolves the family choice
in boosting's favor. Against that limited historical distinction, logistic has lower Brier,
log loss, and annual share error in every 2008--2016 forward year, while its linear spline tails
provide a clearer extrapolation rule under the documented support shifts than boosting's
terminal-leaf continuation.

The boosting series should be reported alongside the primary historical estimates so the 1990
and 1991 divergence remains visible. No model should be selected because its pre-2004 series is
smoother, and these results do not justify retuning, changing thresholds, or treating pre-2004
accuracy as observed. The final release must state that identification still relies on
transporting the 2004--2007 class-conditional feature relationship backward.

## Generated outputs

- `output/tables/historical_model_family_annual.csv`
- `output/tables/historical_model_family_paired.csv`
- `output/tables/historical_model_family_boundary.csv`
- `output/tables/historical_model_family_continuous_support.csv`
- `output/tables/historical_model_family_categorical_support.csv`
- `output/tables/historical_model_family_support_envelope.csv`
- `output/figures/historical_model_family_shares.pdf`
- `output/figures/historical_model_family_differences.pdf`
- `output/historical_model_family/shards/`

Only fitted artifacts from the common comparison and aggregate diagnostics are retained; no
loan-level historical probabilities or target characteristics are persisted.
