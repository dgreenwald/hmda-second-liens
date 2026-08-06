# Random-Forest Mixture Robustness Findings

## Design

The fixed Random Forest uses the established 50-tree, depth-10 specification with the two raw
continuous variables and full purchaser- and loan-type indicator sets. Each source year is
reweighted to equal first- and second-lien prior mass. Balanced-prior forest odds provide the
density-ratio score, and every target year receives the same mixture-share adjustment used for
logistic and gradient boosting. No forest hyperparameter was selected using these results.

All nine reverse source-window fits and the final 2004--2007 fit are saved. Every target-share
optimizer converged, none reached a boundary, and mean adjusted probabilities equal fitted
mixture shares to numerical tolerance.

## Reverse-time comparison

Metrics are averaged within horizon and then equally across horizons 1--9.

| Metric | Logistic mixture | Random Forest mixture | Boosting mixture |
|---|---:|---:|---:|
| Brier score | 0.031779 | 0.030394 | 0.026773 |
| Log loss | 0.105932 | 0.099972 | 0.092328 |
| Mean probability minus observed share | -0.024513 | -0.023430 | -0.015447 |
| Calibration intercept | 0.740 | 0.876 | 0.533 |
| Calibration slope | 1.066 | 1.102 | 1.031 |

The annual mixture adjustment materially changes the interpretation of the earlier forest
comparison. Relative to mixture logistic, the adjusted forest lowers Brier by 4.4% and log loss
by 5.6%. It has lower Brier in 35 of 45 cells and at all nine mean horizons. Thus, the original
forest's raw source-period intercept was an important part of its apparent disadvantage.

The forest nevertheless does not catch gradient boosting. It has higher Brier in 42 of 45
cells and at every horizon. Its equal-horizon Brier is 0.003621 higher, a 13.5% disadvantage.
The gap generally becomes larger at long horizons. Forest probabilities improve on logistic
discrimination enough to lower proper scores, but their mixture shares and calibration shape
remain notably weaker than boosting.

## Forward-regime comparison

| Metric | Logistic mixture | Random Forest mixture | Boosting mixture |
|---|---:|---:|---:|
| Brier score | 0.007307 | 0.015284 | 0.011994 |
| Log loss | 0.026763 | 0.078949 | 0.039529 |
| Mean probability minus observed share | -0.002665 | -0.015574 | -0.011881 |
| Calibration intercept | 0.133 | 4.319 | 1.734 |
| Calibration slope | 0.788 | 1.045 | 0.983 |

The forest is worse than both alternatives in every 2008--2016 forward year. Its annual fitted
share averages 1.56 percentage points below observed prevalence, and the very large calibration
intercept indicates a severe level failure despite a slope near one.

## Decision

Retain the mixture-adjusted forest as the appropriate Random Forest robustness specification.
It demonstrates that annual mixture adjustment improves the original estimator and that the
logistic-versus-forest conclusion should not rest on raw source-period probabilities.

Do not reopen RF tuning. The fixed forest is consistently dominated by boosting in the primary
reverse design, its disadvantage grows at the longer horizons most relevant to historical
application, and it performs worst in every forward year. Earlier tests also found no material
benefit from increasing tree count. Plausible depth or leaf-size refinements are unlikely to
overturn all three pieces of evidence sufficiently to justify the computational cost.

## Outputs

- `output/tables/rf_mixture_reverse_metrics.csv`
- `output/tables/rf_mixture_reverse_summary.csv`
- `output/tables/rf_mixture_forward_metrics.csv`
- `output/tables/rf_mixture_forward_summary.csv`
- `output/tables/rf_mixture_comparison.csv`
- `output/figures/rf_mixture_reverse_horizons.pdf`
- `output/figures/rf_mixture_forward_years.pdf`
- `output/model/rf_mixture_challenger.pkl`
- `output/model/rf_mixture_folds/*.pkl`
