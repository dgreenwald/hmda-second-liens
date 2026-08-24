# Common model-family comparison findings

## Scope

This document reports the synchronized post-selection comparison of the four frozen logistic
and histogram-gradient-boosting finalists. Unlike the family-specific selection tables, every
model here is fitted with equal first- and second-lien mass within each source year, interpreted
as a density ratio, adjusted using the same target-year mixture-share estimator, and scored on
the same observations. The comparison therefore supports paired logistic-versus-boosting
statements that cannot be made from the original raw-logistic and adjusted-boosting selection
objectives.

The four finalists are:

1. unrestricted logistic: `spline_lti__purchaser_type`, ridge `C=0.1`;
2. HMDA-only logistic: `hmda_only__spline_lti__none`, ridge `C=1`;
3. unrestricted boosting: 7 leaves, learning rate 0.05, 200 iterations, L2 1, and minimum leaf
   size 1,000; and
4. HMDA-only boosting: the same boosting hyperparameters without the county-value predictor.

Reverse evaluation uses the frozen 45-cell triangle and first averages within backward horizon,
then equally across horizons 1--9. Forward evaluation fits on 2004--2007 and gives equal weight
to validation years 2008--2016.

## Validation integrity

The synchronized comparison contains all 216 planned model cells: 45 reverse and nine forward
cells for each finalist. All four models use identical observation counts and observed shares
within every evaluation cell. All 216 mixture optimizations converge, none reaches a boundary,
and the largest absolute EM-versus-direct-optimization share difference is below
`4.9e-9`. Mean adjusted probability equals the fitted mixture share within the required
numerical tolerance. Every fitted artifact and metadata digest passes validation.

## Aggregate results under the common convention

| Design | Predictor set | Family | Brier | Log loss | Share error | Absolute share error | Calibration intercept | Calibration slope |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Reverse | Unrestricted | Logistic | 0.031765 | 0.105881 | -0.024519 | 0.024946 | 0.741 | 1.067 |
| Reverse | Unrestricted | Boosting | **0.026648** | **0.091985** | -0.015275 | **0.016159** | 0.530 | 1.032 |
| Reverse | HMDA-only | Logistic | 0.048864 | 0.152511 | -0.044160 | 0.044535 | 1.135 | 1.000 |
| Reverse | HMDA-only | Boosting | **0.039918** | **0.128940** | -0.032580 | **0.032743** | 0.915 | 0.979 |
| Forward | Unrestricted | Logistic | **0.007306** | **0.026759** | -0.002660 | **0.002660** | 0.132 | 0.788 |
| Forward | Unrestricted | Boosting | 0.012197 | 0.040129 | -0.012131 | 0.012131 | 1.788 | 0.986 |
| Forward | HMDA-only | Logistic | **0.007472** | **0.027213** | 0.000305 | **0.001643** | -0.041 | 1.013 |
| Forward | HMDA-only | Boosting | 0.012103 | 0.038209 | -0.010087 | 0.010087 | 1.578 | 1.191 |

Share error is mean adjusted probability minus the observed second-lien share. Reverse values
use the declared equal-horizon weighting, while forward values are equal-year means.

## Matched logistic-versus-boosting evidence

| Design | Predictor set | Boosting minus logistic Brier | Relative Brier change | Boosting minus logistic log loss | Cells with lower boosting Brier |
|---|---|---:|---:|---:|---:|
| Reverse | Unrestricted | -0.005117 | -16.1% | -0.013896 | 45 of 45 |
| Reverse | HMDA-only | -0.008945 | -18.3% | -0.023571 | 45 of 45 |
| Forward | Unrestricted | +0.004891 | +66.9% | +0.013370 | 0 of 9 |
| Forward | HMDA-only | +0.004632 | +62.0% | +0.010996 | 0 of 9 |

The result is not a marginal or metric-specific difference. Boosting has lower Brier and log
loss in every reverse cell for both predictor sets. Logistic has lower Brier and log loss in
every forward year for both predictor sets. The same split appears in prevalence accuracy:
boosting has lower absolute share error in 30 of 45 unrestricted reverse cells and 34 of 45
HMDA-only reverse cells, but in none of the 18 forward cells.

## Reverse-time pattern

The reverse advantage from boosting is present at all nine horizons and generally grows with
backward distance. For the unrestricted models, the Brier gain increases from 0.001139 at
horizon 1 to 0.007395 at horizon 9 and peaks at 0.007518 at horizon 7. For the HMDA-only models,
it increases from 0.002422 at horizon 1 to 0.010098 at horizon 9 and peaks at 0.013391 at
horizon 7. Log-loss differences follow the same pattern.

Both families still underpredict the second-lien share on average in reverse validation, but
boosting reduces the absolute share error by 35.2% for the unrestricted predictor set and
26.5% for HMDA-only predictors. Its calibration slopes are also close to one. These results
provide broad evidence that the boosted density-ratio shape adapts better when transported to
earlier labeled regimes.

## Forward-regime warning

The 2004--2007 logistic fits outperform boosting in every year from 2008 through 2016. The
unrestricted boosting model underpredicts the annual share by 1.21 percentage points on
average, compared with 0.27 point for logistic. The HMDA-only boosting model underpredicts by
1.01 points, while HMDA-only logistic has a signed mean error of only +0.03 point and an
absolute mean error of 0.16 point.

Although the unrestricted boosted calibration slope is closer to one than the logistic slope,
its calibration intercept and prevalence error are much worse. Conditional probability spread
therefore does not compensate for the level miss. This is consistent with the previously
documented transportability concern: histogram trees carry terminal-leaf values beyond their
observed split support, whereas the selected logistic spline has linear tails.

## Predictor-set comparison

The county-value predictor remains important in reverse validation under the common mixture
convention. Relative to unrestricted models, HMDA-only Brier is higher by 0.017099 for logistic
and 0.013271 for boosting; log loss is higher by 0.046630 and 0.036955, respectively. This is
large compared with the within-family differences and supports treating the HMDA-only models
as portability checks rather than primary estimators.

That restriction penalty does not carry into the forward regime. Unrestricted and HMDA-only
logistic differ in Brier by only 0.000166, and HMDA-only boosting is slightly better than
unrestricted boosting by 0.000093. HMDA-only models also have smaller forward share errors.
The county-value variable is therefore strongly useful for backward transport in the labeled
design, but its incremental forward value is regime-dependent.

## Interpretation and Step 10 implication

The refreshed common-score results reproduce, rather than overturn, the earlier unrestricted
finding. The reverse evidence for boosting is temporally universal, material, and somewhat
stronger at long horizons. The forward warning is equally universal and is slightly worse for
the refreshed unrestricted boosting winner than in the stale diagnostic run.

These results satisfy the challenger's reverse-time breadth criterion but do not by themselves
resolve the final estimator choice. The intended application is backward, so the reverse
design is the primary validation exercise; however, the complete forward loss and known tree
extrapolation behavior are directly relevant to transportability. Following the frozen
protocol, the completed Step 10 comparison retains unrestricted logistic as the primary
estimator and unrestricted boosting as the principal model-family robustness series. Their
unrestricted historical aggregate paths are close after 1991, while explicit support shifts
and logistic's universal forward advantage favor its clearer linear-tail extrapolation rule.
Do not retune either family. Retain both HMDA-only variants as portability robustness models.
See `HISTORICAL_MODEL_FAMILY_FINDINGS.md` for the complete historical evidence and final
decision.

## Generated outputs

- `output/tables/model_family_comparison_cells.csv`
- `output/tables/model_family_comparison_reverse_horizons.csv`
- `output/tables/model_family_comparison_summary.csv`
- `output/tables/model_family_comparison_paired_cells.csv`
- `output/tables/model_family_comparison_paired_summary.csv`
- `output/figures/model_family_comparison_reverse.pdf`
- `output/figures/model_family_comparison_forward.pdf`
- `output/model_family_comparison/models/`
- `output/model_family_comparison/shards/`

Only fitted model artifacts and aggregate diagnostics are retained; no loan-level diagnostic
probabilities are persisted.
