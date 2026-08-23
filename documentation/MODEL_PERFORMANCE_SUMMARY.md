# Logistic and boosting model-performance summary

## Scope

This document summarizes the synchronized cluster results for the four selected loan-level
models:

1. unrestricted ridge logistic;
2. HMDA-only ridge logistic;
3. unrestricted histogram gradient boosting; and
4. HMDA-only histogram gradient boosting.

The results use the frozen reverse-time design: models are fitted on four-year windows from
2005--2008 through 2013--2016 and evaluated on every earlier labeled year available to each
window. This produces 45 validation cells at backward horizons 1--9. Reported selection
statistics first average cells within horizon and then give all nine horizons equal weight.
The HMDA-only searches use the same estimation sample as the unrestricted searches, so their
within-family comparisons show the best selected performance with and without the county-value
predictor on that common sample.

These are reverse-validation selection results, not current forward-validation diagnostics.
The synchronized outputs do not contain a refreshed, common-score forward comparison for all
four selected models.

## Selected models and aggregate performance

| Family | Predictor set | Selected specification | Selection score | Mean share error | Other score |
|---|---|---|---:|---:|---:|
| Logistic | Unrestricted | Spline `log_lti`; linear county-value-to-loan ratio; purchaser and loan-type indicators; purchaser interactions; ridge `C=0.1` | Raw Brier: **0.065321** | -0.074651 | -- |
| Logistic | HMDA-only | Spline `log_lti`; purchaser and loan-type indicators; no interactions; ridge `C=1` | Raw Brier: **0.077116** | -0.083124 | -- |
| Boosting | Unrestricted | 7 leaves, learning rate 0.05, 200 iterations, L2 1, minimum leaf size 1,000 | Mixture-adjusted Brier: **0.026648** | -0.015275 | Log loss: **0.091985** |
| Boosting | HMDA-only | 7 leaves, learning rate 0.05, 200 iterations, L2 1, minimum leaf size 1,000 | Mixture-adjusted Brier: **0.039918** | -0.032580 | Log loss: **0.128940** |

Mean share error is mean predicted second-lien probability minus observed second-lien share,
using the same equal-horizon aggregation. Thus all four models underpredict second-lien shares
on average in the backward exercise. The mean absolute share errors are 0.074951 and 0.083420
for unrestricted and HMDA-only logistic, respectively, and 0.016159 and 0.032743 for
unrestricted and HMDA-only boosting.

The logistic and boosting selection scores must not be compared directly. Logistic selection
uses probabilities from a model fitted to the observed source distribution and a raw Brier
objective. Boosting fits with equal first- and second-lien mass within each source year and
adjusts probabilities using a target-year mixture share before computing Brier and log loss.
Consequently, the smaller boosting numbers combine density-ratio shape and mixture adjustment;
they do not by themselves establish that boosting outperforms logistic under a common scoring
convention.

## Unrestricted versus HMDA-only performance

### Logistic

Removing `log_county_value_to_loan` increases equal-horizon raw Brier by **0.011795**, from
0.065321 to 0.077116. This is an 18.1% deterioration relative to the unrestricted score. The
unrestricted logistic model has lower Brier in **all 45 of 45** matched validation cells.

The unrestricted search selects purchaser-type interactions and fairly strong ridge
regularization (`C=0.1`). The HMDA-only search selects the simpler no-interaction spline model
with `C=1`. Regularization is nearly irrelevant around the restricted winner: changing its
`C` from 1 to 10 changes equal-horizon Brier by less than 0.000001. The best restricted model
with loan-type interactions scores 0.077140, also very close to the restricted winner but
still well behind the unrestricted model.

### Gradient boosting

Removing the county-value-to-loan predictor increases equal-horizon mixture-adjusted Brier by
**0.013271**, from 0.026648 to 0.039918, a 49.8% deterioration. Adjusted log loss rises by
**0.036955**, from 0.091985 to 0.128940. The unrestricted boosting model has lower Brier and
lower log loss in **all 45 of 45** matched validation cells.

Both boosting searches independently select the same structure and hyperparameters: 7 leaves,
learning rate 0.05, 200 iterations, L2 regularization 1, and minimum leaf size 1,000. This
makes the unrestricted/HMDA-only gap especially interpretable: it is driven by the predictor
set rather than different selected tree complexity. All 45 target-share optimizations converge
for both selected models, and none reaches a mixture boundary.

## Performance by backward horizon

The restriction penalty is positive at every horizon for both families and generally grows as
the validation year moves farther behind the training window.

| Horizon | Logistic unrestricted raw Brier | Logistic HMDA-only raw Brier | Restriction penalty | Boosting unrestricted adjusted Brier | Boosting HMDA-only adjusted Brier | Restriction penalty |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.019520 | 0.025038 | 0.005519 | 0.014713 | 0.018927 | 0.004214 |
| 2 | 0.031940 | 0.040220 | 0.008280 | 0.017089 | 0.022919 | 0.005831 |
| 3 | 0.046992 | 0.057585 | 0.010593 | 0.020073 | 0.028313 | 0.008241 |
| 4 | 0.062560 | 0.074544 | 0.011984 | 0.023933 | 0.035223 | 0.011290 |
| 5 | 0.075902 | 0.089399 | 0.013497 | 0.028526 | 0.043537 | 0.015011 |
| 6 | 0.091510 | 0.107444 | 0.015933 | 0.033503 | 0.052692 | 0.019190 |
| 7 | 0.098931 | 0.115540 | 0.016609 | 0.034904 | 0.056416 | 0.021512 |
| 8 | 0.089769 | 0.103910 | 0.014141 | 0.035546 | 0.055073 | 0.019527 |
| 9 | 0.070767 | 0.080366 | 0.009599 | 0.031544 | 0.046165 | 0.014621 |

Absolute Brier is not monotone in horizon because different horizons contain different target
years and have declining support. The informative comparison is the matched restriction
penalty within each horizon. It peaks at horizon 7 for both families, at 0.016609 for logistic
and 0.021512 for boosting.

## Interpretation

The synchronized evidence supports three conclusions.

First, `log_county_value_to_loan` carries substantial and temporally broad predictive
information. Omitting it worsens every reverse-validation cell in both model families, not
just the aggregate average. The HMDA-only variants therefore work as portability and
robustness models but do not match unrestricted performance on the common selection sample.

Second, the preferred functional form changes when the county-value predictor is removed.
Unrestricted logistic benefits from purchaser-specific continuous slopes, while the
HMDA-only winner uses a spline in `log_lti` without continuous-by-category interactions. In
contrast, both boosting searches choose exactly the same compact tree configuration.

Third, these outputs do not settle logistic versus boosting. A fair family comparison requires
both selected models to be evaluated with the same probability convention and target-share
adjustment in the same cells. Earlier unrestricted diagnostics did such a comparison for an
older boosting fit, but the project documentation marks those forward and reverse values as
stale after the refreshed cluster selection changed the boosting winner. They should not be
used as the current four-model comparison. In particular, no current forward results for the
HMDA-only finalists are present in the synchronized selection outputs.

## Common diagnostic implementation

The repository now provides a post-selection comparison that holds the evaluation convention
fixed without reopening either model search. It translates each of the four frozen winners
into an equal-source-year-prior density-ratio fit, estimates a separate mixture share in every
target year, and scores all probabilities through the shared evaluator. The design contains
45 reverse and nine forward cells per finalist, or 216 model-cell records in total. Logistic
versus boosting differences are matched within the unrestricted and HMDA-only predictor sets.

The preferred cluster entry point prepares the 40-task array and a dependent aggregation job.
With `--submit`, one invocation submits both jobs and records their IDs under a run-specific
directory; the aggregation job uses `afterok` and therefore runs only if every array task
succeeds:

```bash
make model-family-comparison-cluster-submit
```

Use `make model-family-comparison-cluster` to prepare and inspect the scripts without
submission. The following lower-level targets remain available for debugging or manual
recovery:

```bash
make generate-model-family-comparison
make submit-model-family-comparison
make aggregate-model-family-comparison
```

The generator does not submit unless passed `--submit`. Aggregation requires every immutable
shard, verifies common sample counts and observed shares, and writes
`model_family_comparison_{cells,reverse_horizons,summary,paired_cells,paired_summary}.csv` plus
reverse and forward figures. It never persists loan-level probabilities. Numerical findings
should be added here only after the refreshed winner artifacts have completed this workflow.

## Source artifacts

The aggregate values above come from the synchronized compatibility tables:

- `output/tables/logistic_selection_decision.csv`
- `output/tables/logistic_selection_core_{cells,horizons,summary}.csv`
- `output/tables/logistic_selection_hmda_only_decision.csv`
- `output/tables/logistic_selection_hmda_only_{cells,horizons,summary}.csv`
- `output/tables/boosting_challenger_{decision,cells,horizons,summary}.csv`
- `output/tables/hmda_only_boosting_{decision,cells,horizons,summary}.csv`

The selected 2004--2007 refits and their metadata sidecars are:

- `output/model/logistic_selected.pkl`
- `output/model/logistic_hmda_only_selected.pkl`
- `output/model/boosting_challenger.pkl`
- `output/model/boosting_hmda_only_challenger.pkl`

All values in this document were read from the selected rows and their 45 matched cell records;
no loan-level predictions were persisted or added.
