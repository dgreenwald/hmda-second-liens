# Density-Ratio Mixture Validation Findings

## Status

The reverse-time validation completed for all 45 training-window/target-year
cells. Each four-year source window was used to estimate three candidate
density ratios: a pooled logistic score with one-sided normalization, a score
whose logistic source fit absorbed source-year fixed effects before the same
normalization, and a known-source-prior score. The last specification reweights
each source year to a 50/50 lien-status prior, so its complete fitted log odds,
including the intercept, estimates the log density ratio. Target-year mixture
shares were then estimated without using the target labels.

## Main result

Using the known source shares greatly improves aggregate count-share prediction.
With absolute share errors first averaged within forecast horizon and then
equally across the nine horizons, the results are:

| Estimator | Mean absolute count-share error |
| --- | ---: |
| Adjusted hard classification, known source prior | 0.0239 |
| Mixture, known source prior | 0.0249 |
| Raw logistic mean probability | 0.0750 |
| Raw logistic hard classification | 0.0892 |
| Mixture, source-year fixed effects | 0.1267 |
| Mixture, pooled source model | 0.1269 |
| Adjusted hard classification, source-year fixed effects | 0.1319 |
| Adjusted hard classification, pooled source model | 0.1320 |

Relative to the previous fixed-effect normalization, the known-prior mixture
reduces mean absolute share error by about 80 percent. Relative to the raw
logistic mean, it reduces error by about 67 percent. It has lower absolute error
than both alternatives at every horizon. Its mean absolute error rises from
0.0040 at horizon one to 0.0397 at horizon seven, then remains below 0.039 at
horizons eight and nine.

## Numerical diagnostics

The new result is numerically stable:

- all nine known-prior source logistic fits converged;
- the bounded optimizer and EM fixed-point check agree within `5.2e-9`;
- none of the 45 known-prior estimates reaches a share boundary.

An important structural warning remains. For a
valid ratio `r(x) = f_1(x) / f_0(x)`, both
`E_0[r(X)] = 1` and `E_1[1 / r(X)] = 1` should hold. The known-prior estimator
matches the weighted conditional-log-likelihood score rather than imposing
these nonlinear moments. In the source folds, `E_0[r(X)]` ranges from about 44
to 32 million and `E_1[1 / r(X)]` ranges from about 1.9 to 10.5. The most severe
discrepancies occur in the earliest folds. Thus the score is very effective for
share prediction but does not yet behave as a coherent literal density ratio.

## Interpretation

The full reverse-time backtest supports the known-source-prior mixture as a
promising main estimator of annual count shares. The previous negative result
was largely caused by its one-sided normalization. For now, the known-prior
procedure should be described as a predictive pseudo-likelihood estimator, not
as a structurally validated estimate of `f_1(x) / f_0(x)`. Any structural claim
would require a constrained density-ratio estimator or further evidence that
the failed moments are caused by a small and manageable set of tail
observations.

## Reproducibility

Run `make estimate-mixture-shares`. The cell-level estimates, horizon summaries,
source-year intercept diagnostics, and ratio-fit diagnostics are written under
`output/tables/`. The focused full-validation results are also retained under
`output/tables/mixture_known_prior_validation/`. The automated suite passes 95
tests.
