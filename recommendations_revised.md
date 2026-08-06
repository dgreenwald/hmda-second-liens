# Revised Recommendations and Implementation Agenda

## Objective

The project should select and document the most credible estimator for imputing
first- versus second-lien status in 1990--2003 HMDA records. The central problem
is not maximizing same-era classification accuracy. It is transporting a
relationship estimated after 2004 into an earlier period where true lien status
is unavailable and the second-lien prevalence is unknown.

The current evidence makes logistic regression a serious candidate for the main
estimator:

- On 22.75 million observations from 2008--2016, logistic regression has higher
  accuracy and second-lien F1 than the Random Forest in every year.
- Logistic regression and the Random Forest have nearly identical ROC-AUC.
- Loan-type indicators add modest but consistent out-of-time value to logit and
  should be retained.
- The existing RF grid found no meaningful gain from increasing the number of
  trees from 50 to 200, while `max_depth=10` was the best tested depth.

Two qualifications matter before choosing the final model. First, logit is fit
on all 2004--2007 observations, whereas the deployed RF is fit on a 25% random
subsample. The comparison should be repeated using equal training information.
Second, 2008--2016 has already informed several modeling decisions, so it should
no longer be described as a pristine test set. Subsequent work should use a
predeclared temporal evaluation protocol and describe the results as rolling or
out-of-era validation.

## Guiding Principles

1. **Define the deliverable before optimizing it.** Hard classifications,
   individual probabilities, and aggregate predicted shares require different
   evaluation criteria.
2. **Separate discrimination from calibration.** High ROC-AUC does not establish
   that predicted probabilities are accurate.
3. **Do not treat calibration as identification.** Recalibration on 2004 or later
   cannot reveal the unknown pre-2004 second-lien prevalence without an external
   anchor or an additional assumption.
4. **Use temporal, not random, model selection.** The application is backward
   transport across market regimes.
5. **Compare estimators on equal footing.** Models should use the same eligible
   observations and information whenever computationally feasible.
6. **Prefer parsimonious, transportable features.** A feature that raises
   same-era accuracy but depends on time-specific institutions or coding may
   weaken the backward imputation.
7. **Preserve public reproducibility.** Proprietary data may provide a useful
   supplemental plausibility check, but should not become a required pipeline
   input.

## Step-by-Step Implementation Agenda

### Step 1: Define the target output and loss

Before further model selection, write down the intended public outputs and how
errors affect their use.

Decide whether the release will contain:

- A hard first-/second-lien classification.
- An estimated probability of second-lien status.
- Aggregate second-lien shares by year or subgroup.
- All three, with clear guidance about their distinct interpretations.

For hard classifications, specify whether false positives and false negatives
have equal cost. If not, state an economically motivated relative cost or a
required recall/precision target. Do not select a threshold merely because it
maximizes an arbitrary metric.

For aggregate shares, decide whether the estimand is the share of hard labels or
the mean predicted probability. The latter is preferable only if probability
calibration is credible.

**Deliverable:** a short `MODEL_TARGET.md` section, or equivalent prose in the
paper, defining the estimand, outputs, and error costs before tuning begins.

### Step 2: Audit sample construction and feature definitions

Add a data-flow audit that counts observations after each restriction:

1. Raw yearly HMDA records.
2. Originated purchase loans for owner occupants.
3. Valid state, county, and loan-type restrictions.
4. Positive, nonmissing income and loan amount.
5. Successful FHFA merge.
6. Final model sample.

Report counts and retention rates by year. For labeled years, also report
retention separately for first and second liens. This will show whether the
model is trained and applied to a selected sample whose composition changes over
time.

Pay particular attention to the balanced FHFA panel. The current construction
drops a county from every year if it is missing HPI in any application year.
Compare this with a year-specific merge and document how many loans and counties
the balanced-panel requirement removes.

Audit the meaning of `log_ltv = log(hpi / loan_amt)`. County HPI is an index, not
a property value, so this variable is only an LTV proxy. Check whether HPI index
levels are comparable across counties and base periods. If level comparability
is weak, consider replacing or supplementing the variable with HPI growth rates
that have a clearer interpretation.

Missing-value imputation should not be introduced mechanically. Income and loan
amount restrictions define the current analysis sample, while the FHFA merge is
a separate coverage problem. Any expansion of the estimand should be reported
explicitly rather than described as a purely predictive improvement.

**Deliverables:**

- `sample_attrition_by_year.csv`.
- `sample_attrition_by_lien_status.csv` for labeled years.
- A short comparison of balanced-panel and year-specific FHFA coverage.
- A documented decision about retaining or revising `log_ltv`.

**Implemented decision:** retire the native-index `log_ltv` feature. Annual Zillow county
ZHVI is used to estimate a county-specific dollar-per-FHFA-index-point scale, with the
geometric mean Zillow/FHFA ratio over all available overlap years as the primary estimator.
The resulting feature is `log_county_value_to_loan`. FHFA matching is year-specific rather
than balanced over 1990--2016. See `COUNTY_VALUE_SCALING.md` for support, fit, robustness,
and loan-level coverage diagnostics.

### Step 3: Establish fair RF and logistic baselines

Refit the Random Forest on the full 2004--2007 training extract, using the
currently supported hyperparameters, and compare it with logit trained on the
same observations. This determines whether the existing logit advantage partly
reflects its four-times-larger estimation sample.

Retain the existing 25%-sample RF as a historical specification, but do not use
it as the sole basis for choosing between estimator families.

Compare the models using:

- Accuracy.
- Second-lien precision, recall, and F1.
- ROC-AUC and average precision.
- Log loss and Brier score.
- Calibration intercept and slope.
- Paired disagreement counts and McNemar tests for hard classifications.
- Fit time, prediction time, and artifact size.

Statistical significance should always be accompanied by effect sizes. With
millions of observations, substantively trivial differences will be highly
significant.

**Decision gate:** if full-sample RF does not materially improve on logit, make
logistic regression the provisional primary estimator and retain RF as a
robustness check.

**Implemented decision:** the fair full-sample comparison makes logistic regression the
provisional primary estimator and retains RF as a substantive robustness model. Logit performs
better on accuracy, second-lien F1, average precision, and Brier score in every 2008--2016
validation year, while RF performs better on ROC-AUC and log loss and has calibration slopes
closer to one. See `BENCHMARK_FINDINGS.md` and the `benchmark_*.csv` tables for the complete
results and computational comparison.

### Step 4: Predeclare a temporal model-selection protocol

Do not use random cross-validation to choose features or hyperparameters. Use
expanding or rolling temporal folds over the labeled period. A practical design
is to train on earlier labeled years and score the next year, repeating the
exercise as the training window expands.

Because all 2008--2016 results have already been inspected, no subset of those
years is genuinely untouched. The paper should say so transparently. From this
point forward:

- Freeze the candidate specifications before rerunning comparisons.
- Use identical temporal folds for all candidates.
- Report the complete sequence of yearly results, not only the best years.
- Avoid choosing a model separately for each validation year.

Training on 2008 or later should be treated as an experiment, not an automatic
improvement. A broader training window exposes the model to more base rates, but
post-crisis relationships need not resemble the 1990s. Compare alternative
training windows under the same rolling protocol.

**Deliverable:** a machine-readable table listing each fold's training years,
validation year, model specification, and metrics.

**Frozen protocol:** use four-year later training windows to predict every available earlier
labeled year, producing backward horizons 1--9. Raw out-of-sample Brier is the sole selection
metric, averaged within horizon and then equally across horizons. See
`MODEL_SELECTION_PROTOCOL.md` for the complete predeclared design.

### Step 5: Optimize the logistic specification

Treat logistic regression as a model in its own right rather than inheriting
feature decisions made for the RF.

Test a small, predeclared set of specifications:

1. Current model: `log_lti`, `log_ltv`, purchaser type, and loan type.
2. Add `has_edit_status` and `loan_below_10k` individually and jointly.
3. Add flexible terms for `log_lti` and `log_ltv`, preferably restricted cubic
   splines or a small number of prespecified polynomial terms.
4. Add the `log_lti` by `log_ltv` interaction.
5. If the HPI audit supports it, add trailing one- and three-year county HPI
   growth using only information available at origination.
6. Test state or broad-region effects as a separate specification.

Retain loan type unless a new temporal ablation reverses the existing result.

Tune regularization strength `C` on the temporal folds. Elastic net may be
tested, but it is unlikely to be central with a small feature set and millions
of observations. Standardize continuous terms within the estimation pipeline
if regularization comparisons depend on coefficient scale.

For interpretable coefficient tables, use explicit reference categories rather
than all category indicators plus an intercept. Report odds-ratio contrasts or
average marginal effects, with the feature transformation stated clearly.

Avoid year fixed effects: they cannot be transported meaningfully to
1990--2003. Treat MSA, tract, and lender features cautiously because geographic
definitions and lender identifiers change over time. Lender identity should not
be used unless a defensible rule exists for previously unseen lenders.

**Decision gate:** choose the simplest specification whose temporal gains are
consistent and substantively meaningful, not merely statistically significant.

**Frozen candidate set:** cross three continuous functional forms with four loan-type and
purchaser-type interaction structures, and tune ridge strength using the reverse folds.
Exclude HPI growth, edit-status, small-loan, year, lender, MSA, and tract features. State and
region indicators are guarded post-selection challengers, not automatic additions. The exact
grid and exclusions are recorded in `MODEL_SELECTION_PROTOCOL.md`.

**Focused post-selection challenger:** after the original grid selected a common `log_lti`
spline with linear purchaser-specific slope adjustments, evaluate a richer specification that
interacts every `log_lti` spline-basis term with purchaser type. Keep the linear
`log_county_value_to_loan` by purchaser interactions, use the same reverse folds and staged
ridge grid, and compare raw Brier at both the equal-horizon and individual-cell levels. Do not
replace the selected core automatically; first assess whether any improvement is material and
consistent across backward horizons.

**Result:** the spline-by-purchaser challenger selects `C=0.1` and has equal-horizon Brier
0.065609, compared with 0.065387 for the simpler core. It improves only 16 of 45 cells and is
worse at every horizon from 2 through 9, so it is rejected. See
`LOGISTIC_SELECTION_FINDINGS.md` for the complete interpretation.

### Step 6: Diagnose probability calibration

For each temporal validation year, produce:

- A reliability diagram using bins with adequate support.
- Brier score and log loss.
- Calibration-in-the-large, measured by the mean prediction minus the observed
  prevalence or an equivalent intercept estimate.
- Calibration slope.
- Observed and mean-predicted second-lien shares.

These results will distinguish ranking performance from prevalence error. A
stable intercept error is different from a changing calibration slope or
subgroup-specific failure.

Do not automatically apply isotonic regression or Platt scaling. Isotonic
calibration can overfit regime-specific patterns, while Platt scaling on a
boom-era random holdout cannot correct an unknown target-period base rate. Any
post-hoc calibration method must be trained using a temporally defensible rule
and evaluated in later years not used to fit the calibrator.

**Implemented protocol:** diagnose the fixed selected logistic specification in every cell of
the 45-cell reverse-validation triangle. Report raw Brier, log loss, observed and predicted
shares, mean calibration error, and calibration intercept/slope; average metrics using the
same equal-horizon convention as model selection. Construct ten approximate equal-count
reliability bins within each cell and pool their counts by backward horizon. Separately apply
the final 2004--2007 refit to 2008--2016 as a forward-regime robustness check. No recalibrator
is fitted. See `CALIBRATION_PROTOCOL.md`.

**Result:** calibration levels are not temporally stable. In reverse validation, the
equal-horizon observed share is 13.63% while the mean predicted probability is 6.16%, a
7.47-point shortfall that becomes much larger at long horizons. In the separate forward check,
the 2004--2007 model overpredicts every 2008--2016 observed share by 1.27--1.98 points. Retain
probabilities as loan-level uncertainty/ranking outputs, but do not treat their unadjusted mean
as a transportable pre-2004 count-share estimate. See `CALIBRATION_FINDINGS.md`.

**Known-source-prior extension:** freeze the selected specification and `C=0.1`, reweight each
labeled source year to equal first- and second-lien prior mass, and estimate each target year's
intercept from its unlabeled feature distribution. Across the same 45 reverse cells, this lowers
equal-horizon Brier from 0.0654 to 0.0318, log loss from 0.1938 to 0.1059, and the aggregate-share
shortfall from 7.47 to 2.45 percentage points. Mean Brier improves at all nine horizons. In the
2008-2016 forward check, Brier and log loss improve in every year. Proceed to Step 7 without
retuning on these validation outcomes. See `MIXTURE_CALIBRATION_FINDINGS.md`.

If an external, definitionally comparable estimate of pre-2004 prevalence is
available, an intercept or prior-probability adjustment may be considered. The
paper must then identify the external estimate and the maintained assumption
that allows it to anchor HMDA predictions.

**Decision gate:** use mean probabilities for aggregate shares only if their
calibration is sufficiently stable to support that interpretation. Otherwise,
release probabilities with an explicit calibration caveat.

### Step 7: Evaluate thresholds and subgroup performance

Keep 0.5 as the transparent baseline threshold. Class imbalance alone does not
make 0.5 theoretically incorrect: with calibrated probabilities and equal
misclassification costs, it is the Bayes threshold.

Also report precision-recall curves and performance at thresholds tied to the
loss or operating target defined in Step 1. Do not tune a separate threshold to
match each validation year's observed share; that uses information unavailable
in the pre-2004 application.

Report metrics by:

- Loan type.
- Purchaser type, combining sparse categories where necessary.
- State or broad region.
- Relevant bins of `log_lti` and `log_ltv`.

Include observation counts and positive-class prevalence for every subgroup so
small cells are not overinterpreted.

Do not prioritize additional class weighting unless the declared loss function
requires it. With very large bootstrap samples, RF
`class_weight="balanced_subsample"` will behave nearly like `"balanced"` and is
unlikely to solve the observed false-positive problem.

**Deliverables:** precision-recall figures, an operating-threshold table, and
subgroup metric tables.

**Implemented protocol:** compare the frozen raw and known-source-prior probabilities at the
canonical 0.5 threshold in all reverse and forward cells. Construct fixed-grid precision-recall
curves without selecting a new threshold. Report loan type, purchaser type, Census region, and
target-year deciles of both continuous variables, retaining counts and prevalence. Persist every
fitted source-window model and only aggregated diagnostic outputs. See
`THRESHOLD_SUBGROUP_PROTOCOL.md`.

**Result:** in reverse validation, the known-prior adjustment raises second-lien recall from
0.374 to 0.724 and F1 from 0.490 to 0.788 while precision remains high at 0.872. F1 improves in
35 of 38 subgroups and is stable across regions. In the forward check, precision rises from
0.530 to 0.838 and F1 from 0.669 to 0.699. Retain 0.5 and proceed without threshold or subgroup
tuning. See `THRESHOLD_SUBGROUP_FINDINGS.md`.

### Step 8: Perform external and internal plausibility checks

Plot predicted second-lien shares for 1990--2016 on one figure, overlaying actual
shares for every labeled year. Mark the 2004 reporting boundary clearly. Show
both hard-label shares and mean predicted probabilities if both are candidate
outputs.

The continuity check is informative but not dispositive: actual pre-2004 lien
status is unavailable, so smoothness at 2004 cannot prove accuracy. Conversely,
a large unexplained discontinuity would be a warning sign.

Search for external series that measure a genuinely comparable concept:
second-lien originations among owner-occupied home-purchase loans. Broader
measures such as outstanding second-mortgage balances or all home-equity credit
should not be treated as direct validation without a reconciliation of
definitions, coverage, and timing.

Prefer public sources. If a proprietary source is used, make it a supplemental
check rather than a required input, and document its coverage and limitations.

**Deliverables:** the predicted/actual share figure and a source-comparison table
with explicit definition mappings.

**Implemented protocol:** fit no new specification. Apply the persisted final 2004--2007 raw
logistic model and its equal-source-prior density-ratio counterpart to every year from 1990
through 2016. Report the mixture mean probability (which equals the fitted annual mixture
share), its hard 0.5 count share, the raw-logistic mean benchmark, and actual labeled shares.
Checkpoint annual aggregates only. Evaluate the 2003--2004 reporting boundary explicitly and
map each potential public comparison series to the project estimand before interpreting it.

**Result:** the mixture mean rises from 9.14% in 2003 to 13.23% in 2004, a 4.09 percentage-point
jump; the observed 2004 share is 13.54%, so the 2004 mixture error is -0.31 point. The raw mean
also jumps by 3.52 points, indicating a substantial feature-distribution or sample-composition
change at the boundary rather than a discontinuity unique to the mixture adjustment. Over all
labeled years 2004--2016, the mixture mean has mean absolute annual share error 0.30 point,
compared with 1.22 points for the raw mean. The historical mixture series rises from roughly
3--5% in the early 1990s to about 9% in 2000--2003, then peaks at 22.32% in 2006. These patterns
are plausible and labeled-period fit is strong, but neither continuity nor fit after 2004
identifies pre-2004 accuracy. The reviewed FHFA NMDB, Federal Reserve Financial Accounts, New
York Fed Consumer Credit Panel, and American Housing Survey products measure first-lien
mortgages, balances, HELOCs, borrowers, or the housing stock; none directly measures annual
second-lien originations among owner-occupied home-purchase loans. See
`STEP8_PLAUSIBILITY_FINDINGS.md` and `data/public/step8_external_source_comparison.csv`.

### Step 9: Test challenger models only after the core design is stable

`HistGradientBoostingClassifier` is a reasonable optional challenger once the
sample, features, temporal folds, and evaluation metrics are fixed. It should
not be presumed to outperform logit, and its probabilities should undergo the
same calibration diagnostics.

Native missing-value handling does not recover observations removed before
estimation by the cleaning and FHFA-merge stages. Changing those stages is a
sample-definition decision under Step 2, not a free benefit of the estimator.

Avoid adding LightGBM unless sklearn's implementation reveals a clear modeling
limitation. A new dependency should earn its reproducibility and maintenance
cost through a material, stable temporal improvement.

**Decision gate:** retain a challenger only if it improves the declared primary
objective consistently across temporal folds without materially worsening
calibration or transportability.

**Implemented challenger:** use `HistGradientBoostingClassifier` on the four primitive core
features, with purchaser and loan type treated as native categories. Train with equal source-
year class priors, interpret balanced-prior log odds as the density ratio, and fit the same
target-year mixture adjustment used by logistic. A frozen staged grid screens tree structure
on the latest source window, carries two candidates into all 45 reverse cells, and refines
iterations and L2 regularization around the winner. Every fitted model is saved. See
`GRADIENT_BOOSTING_PROTOCOL.md`.

**Result:** the selected 7-leaf, learning-rate 0.05, 200-iteration challenger lowers
equal-horizon reverse Brier from 0.031779 to 0.026773 and log loss from 0.105932 to 0.092328.
It improves Brier in all 45 cells and at all nine backward horizons; its gain increases at long
horizons. However, the final 2004--2007 boosted model is worse than mixture logistic in every
2008--2016 forward year: average Brier is 0.011994 rather than 0.007307, driven chiefly by a
1.19-point prevalence shortfall. Retain boosting as a serious finalist, but do not replace
logistic automatically. Step 10 should compare the historical series and extrapolation/support
behavior before resolving the strong backward evidence against the forward-regime warning.
See `GRADIENT_BOOSTING_FINDINGS.md`.

**Random Forest mixture revisit:** re-evaluate the established 50-tree, depth-10 full-sample
forest using equal source-year class priors and the same annual target mixture adjustment. Its
reverse Brier is 0.030394, improving on mixture logistic's 0.031779 in 35 of 45 cells, which
confirms that the missing annual intercept contributed to the forest's earlier disadvantage.
However, boosting reaches 0.026773 and beats the forest in 42 of 45 cells and at every mean
horizon. The forest is also worst in every 2008--2016 forward year, with Brier 0.015284 versus
0.011994 for boosting and 0.007307 for logistic. Retain the adjusted forest as the correct RF
robustness model, but do not reopen RF tuning. See `RANDOM_FOREST_MIXTURE_FINDINGS.md`.

### Step 10: Select, document, and release the final estimator

Select the final estimator using the predeclared criteria and report the full
comparison, including specifications that were rejected. The final release
should contain:

- The fitted model artifact and exact feature schema.
- Hard predicted labels and second-lien probabilities, if both are supported.
- Join identifiers and HMDA vintage information.
- Training years, software versions, and hyperparameters.
- Calibration and temporal-validation tables.
- A clear statement that pre-2004 accuracy cannot be directly observed.
- Guidance on whether probabilities may be aggregated as expected shares.

The paper should distinguish three claims:

1. The model discriminates first and second liens well in labeled out-of-era
   HMDA data.
2. Its probability calibration is or is not stable across labeled regimes.
3. Applying it before 2004 requires a transportability assumption that cannot
   be fully tested from HMDA alone.

## Recommended Priority Summary

| Priority | Workstream | Expected value |
|---|---|---|
| 1 | Define estimand and loss | Prevents optimizing the wrong output |
| 2 | Sample and HPI audit | Addresses selection and feature-validity risks |
| 3 | Equal-sample RF/logit comparison | Establishes a fair estimator comparison |
| 4 | Temporal protocol and logistic optimization | Improves the leading estimator credibly |
| 5 | Calibration, PR, subgroup, and share diagnostics | Measures the main remaining weaknesses |
| 6 | External plausibility checks | Provides evidence about backward transport |
| 7 | Gradient boosting challenger | Optional test after the research design is fixed |

The project should resist accumulating models and features before completing
the sample audit and defining the estimand. Given the current results, the most
promising path is a carefully specified logistic model, evaluated under a fair
and temporally explicit comparison, with the Random Forest retained as a
robustness benchmark.
