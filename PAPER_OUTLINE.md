# Proposed paper outline

## Working premise and paper-level claim

The paper should be a concise methods-and-measurement letter about recovering annual
second-lien origination shares in HMDA before reliable lien-status reporting begins in 2004.
The primary estimator is the unrestricted ridge-logistic density-ratio model with an annual
known-source-prior mixture adjustment. Gradient boosting is the principal model-family
robustness check; HMDA-only logistic and boosting are portability checks; the fixed Random
Forest is an appendix robustness benchmark.

The paper should distinguish three claims throughout:

1. lien status can be predicted and calibrated in labeled HMDA years;
2. reverse-time validation provides evidence about backward transport, but forward validation
   exposes a meaningful model-family risk; and
3. pre-2004 estimates remain conditional on an untestable class-conditional transportability
   assumption.

The main empirical contribution is therefore a transparent historical annual-share series with
explicit validation and support diagnostics, not a claim that pre-2004 loan-level labels are
observed or perfectly recovered.

## Proposed main-text structure

### Title

Use a title that leads with the measurement problem rather than the algorithm. A working title
is:

> **Recovering Second-Lien Originations in Historical HMDA Data**

Possible subtitle:

> Reverse-Time Validation and Density-Ratio Mixture Adjustment

### Abstract

One paragraph covering:

- the missing/reliably unusable pre-2004 HMDA lien-status field;
- the target estimand: annual second-lien origination-count shares among eligible
  owner-occupied home-purchase loans;
- the formal ridge-logistic density-ratio estimator, trained on 2004--2007 and combined with
  annual target mixture-share estimation;
- the 45-cell reverse-time and nine-year forward validation designs;
- the central comparison: boosting wins every reverse cell, while logistic wins every forward
  year and remains the primary estimator because its extrapolation rule is clearer under
  substantial historical support shifts;
- the resulting 1990--2003 series and the explicit limitation that its accuracy cannot be
  directly observed.

No table or figure appears in the abstract.

### 1. Introduction

#### Purpose

Motivate why historical subordinate-lien originations matter and why the HMDA reporting break
prevents direct measurement before 2004. State the estimand and explain why a simple classifier
probability average is inadequate when the target-year class share changes sharply.

#### Core contributions

Keep the contribution list short:

1. a formal, reproducible ridge-logistic model replacing the exploratory Random Forest;
2. a reverse-time validation design tailored to backward application;
3. a known-source-prior density-ratio mixture estimator for annual shares and adjusted
   probabilities;
4. a fair four-finalist comparison using identical samples and probability conventions; and
5. a historical 1990--2003 series accompanied by explicit support and sensitivity evidence.

#### Main findings preview

Preview the model-family tension rather than hiding it. Boosting has lower Brier score in all
45 reverse cells, but logistic has lower Brier score, log loss, and annual share error in every
2008--2016 forward year. The unrestricted historical logistic and boosting series differ by
only 0.20 percentage point per year on average, with the material divergence concentrated in
1990--1991. This evidence supports logistic as the baseline and boosting as a required
robustness series.

#### Assets

- **Figure 1 (new): Data and validation timeline.** A compact timeline showing unlabeled
  application years 1990--2003, source years 2004--2007, forward validation years 2008--2016,
  and the rolling four-year reverse-validation windows. This should explain the research design
  more efficiently than prose alone.

### 2. Data, target population, and estimand

#### 2.1 HMDA sample

Define the annual loan-level source, vintages, and the owner-occupied home-purchase origination
sample. List the principal restrictions: originated loans, purchase purpose, owner occupancy,
valid state/county and loan type, positive income and loan amount, county-value match, and the
common finite-feature restriction. Explain that lien status is used only where reliably
reported.

Report the three negligible historical purchaser-type exclusions transparently: 700 rows in
1990, 195 in 1991, and three in 1999.

#### 2.2 County value and predictors

Explain the Zillow-scaled FHFA county-value construction and the balanced county panel. Define
the four primitive predictors:

- log loan-to-income ratio;
- log county-value-to-loan ratio;
- purchaser type; and
- loan type.

State that the HMDA-only variants omit county value but retain the identical target sample.

#### 2.3 Estimands and release objects

Define separately:

- the annual second-lien origination-count share;
- the loan-level mixture-adjusted probability used to aggregate expected counts; and
- the frozen 0.5 hard classification, which is secondary and not tuned by year.

Clarify that dollar shares, outstanding balances, HELOC stocks, and all-purpose home-equity
borrowing are different objects.

#### Assets

- **Table 1 (new paper table): Sample construction and variable definitions.** Panel A should
  report representative sample attrition for 1990, 2003, 2004, and 2016 plus totals for
  1990--2003 and 2004--2016. Panel B should define the outcome, four primitive predictors,
  transformations, and data sources. Build from `sample_attrition_by_year.csv`,
  `county_value_coverage_by_year.csv`, and the frozen configuration.
- **Appendix Table A1 (existing pipeline inputs): Annual sample attrition.** Full year-by-year
  version of `output/tables/sample_attrition_by_year.csv`.
- **Appendix Table A2 (existing pipeline inputs): County-value coverage.** Use
  `output/tables/county_value_coverage_by_year.csv` and
  `output/tables/county_value_coverage_by_lien_status.csv`.
- **Appendix Figure A1 (new): County-value coverage over time.** A simple loan-weighted match
  rate series; include only if the coverage variation is substantively useful.

### 3. Estimation and validation design

#### 3.1 Ridge-logistic density ratio

Present the selected unrestricted logistic specification: restricted-cubic-spline `log_lti`,
linear `log_county_value_to_loan`, purchaser- and loan-type reference indicators, and linear
interactions of both continuous predictors with purchaser type, with ridge `C=0.1`. Explain
equal source-year class-prior weighting and why the fitted log odds estimate a density ratio
rather than a posterior tied to the source prevalence.

Keep basis construction and coefficient bookkeeping in an appendix. The main text needs only
the model equation and the interpretation of the fitted log density ratio.

#### 3.2 Annual mixture-share adjustment

State the target density as a mixture of the two source class-conditional densities. Define the
bounded likelihood for the annual share and the adjusted probability. Emphasize that target
labels are not used to estimate the annual share and that, at the optimum, the mean adjusted
probability equals the fitted mixture share.

Contrast this estimator briefly with averaging raw classifier probabilities, whose implicit
source prior is inappropriate when prevalence changes.

#### 3.3 Temporal validation

Describe:

- the 45-cell reverse-time triangle formed by four-year later training windows and earlier
  labeled target years;
- equal weighting within horizon followed by equal weighting across horizons 1--9;
- the final 2004--2007 fit evaluated forward in each year from 2008 through 2016; and
- the common metrics: Brier score, log loss, annual share error, calibration intercept and
  slope, and the frozen hard-classification threshold.

Explain why reverse validation is application-aligned while forward validation remains an
important stress test rather than an interchangeable fold.

#### 3.4 Frozen challengers

Introduce, without reproducing their tuning grids:

- unrestricted gradient boosting with seven leaves;
- HMDA-only logistic and boosting; and
- the fixed Random Forest robustness model.

State that every comparison in the main results uses the same target observations, equal source
priors, and annual mixture adjustment.

#### Assets

- **Table 2 (new paper table): Estimator and validation protocol.** One compact row per finalist
  reporting predictor set, functional form, fixed hyperparameters, source years, weighting,
  target adjustment, and role in the paper. This is assembled from the persisted metadata and
  frozen protocols.
- **Figure 1:** Reuse the data/validation timeline introduced in Section 1.
- **Appendix Table B1 (new): Full logistic specification.** Feature names, reference levels,
  spline knots/rule, scaling convention, and coefficient schema.
- **Appendix Table B2 (existing source tables): Logistic selection grid summary.** Condense
  `logistic_selection_core_coarse_summary.csv`,
  `logistic_selection_core_refinement_summary.csv`, and
  `logistic_selection_decision.csv`.
- **Appendix Table B3 (existing source tables): Boosting selection grid summary.** Condense the
  screen, survivor, refinement, and decision tables. Do not reproduce every candidate in the
  main text.

### 4. Labeled-period validation and model choice

#### 4.1 Primary logistic performance

Report aggregate reverse and forward performance for the unrestricted logistic model. Discuss
annual-share accuracy and calibration alongside Brier and log loss; do not present accuracy or
ROC measures as the primary criteria.

#### 4.2 Logistic versus boosting

Make the conflicting temporal evidence the center of the section:

- boosting improves Brier in all 45 reverse cells and at all nine backward horizons;
- logistic wins all nine forward years on Brier and log loss for both predictor sets; and
- unrestricted forward annual-share mean absolute error is 0.27 percentage point for logistic
  and 1.21 points for boosting.

Explain the extrapolation distinction: the logistic spline has declared linear tails, whereas
histogram boosting carries terminal-leaf values outside its split support.

#### 4.3 Final choice

State the decision directly: unrestricted logistic is the baseline and release estimator;
unrestricted boosting is the principal model-family robustness series. HMDA-only variants test
dependence on the external county-value input. The fixed Random Forest establishes that the
conclusion is not an artifact of comparing only two families, but it does not motivate renewed
tuning.

#### Assets

- **Table 3 (existing data, paper-formatted): Common labeled-period model comparison.** Use the
  eight-row reverse/forward summary from
  `output/tables/model_family_comparison_summary.csv`, reporting Brier, log loss, signed and
  absolute share error, calibration intercept, and calibration slope. Visually distinguish the
  unrestricted primary comparison from HMDA-only robustness rows.
- **Figure 2 (existing figures, preferably combined): Performance by temporal distance.** Place
  `output/figures/model_family_comparison_reverse.pdf` and
  `output/figures/model_family_comparison_forward.pdf` as panels A and B with a common legend
  and paper styling. If space is tight, keep reverse horizons in the main text and move the
  forward-year panel to the appendix while retaining the forward summary in Table 3.
- **Appendix Table C1 (existing): Complete 216-cell comparison.** Use
  `output/tables/model_family_comparison_cells.csv` or a formatted long table derived from it.
- **Appendix Table C2 (existing): Paired model-family differences.** Use
  `model_family_comparison_paired_cells.csv` and
  `model_family_comparison_paired_summary.csv`.
- **Appendix Figure C1 (pipeline asset): Calibration diagnostics.** Reverse-horizon and
  forward-year calibration plots from the logistic and mixture-calibration workflows.
- **Appendix Table C3 (pipeline asset): Threshold and subgroup diagnostics.** Summarize the
  frozen 0.5 threshold and the declared loan type, purchaser type, region, LTI-decile, and
  county-value-to-loan-decile groups. Keep precision-recall curves diagnostic only.

### 5. Historical second-lien shares, 1990--2003

#### 5.1 Primary series

Present the unrestricted logistic mixture-share series as the main historical estimate. Report
the broad pattern rather than narrating every year: a low early-1990s level, a minimum around
1993, and a rise toward roughly 9% by 2000--2003. Overlay observed shares from 2004 onward to
make the reporting boundary visible.

The text should call the values "estimated shares," not imputed truth.

#### 5.2 Family and predictor-set sensitivity

Report that unrestricted logistic and boosting differ by 0.20 percentage point per year on
average over 1990--2003 and by 0.13 point after 1991. Preserve the 1990 difference of 1.13
points rather than smoothing it away. Explain that omitting county value has a larger effect:
the HMDA-only series exceeds the unrestricted series in every pre-2004 year, by 1.22 points on
average for logistic.

#### 5.3 The 2003--2004 boundary

Report the unrestricted logistic increase from 9.14% in 2003 to 13.23% in 2004 and its
-0.31-point 2004 error. All four methods show a 3.53--4.33-point increase, so the break is not
specific to the logistic family or the annual mixture adjustment. Treat it as a visible regime
warning.

#### Assets

- **Figure 3 (existing; central paper figure): Historical model-family shares.** Use
  `output/figures/historical_model_family_shares.pdf`. In the paper-facing version, visually
  emphasize unrestricted logistic, use a contrasting line for unrestricted boosting, and
  de-emphasize the two HMDA-only paths. Keep actual 2004--2016 shares and the reporting boundary.
- **Table 4 (new paper table): Historical levels and sensitivity.** Panel A should report 1990,
  1993, 2000, 2003, and 2004 shares for all four finalists. Panel B should report mean and
  maximum family differences, unrestricted-minus-HMDA-only differences, the 2003--2004 jump,
  and 2004 error. Build from `historical_model_family_annual.csv`,
  `historical_model_family_paired.csv`, and `historical_model_family_boundary.csv`.
- **Appendix Figure D1 (existing): Annual boosting-minus-logistic differences.** Use
  `output/figures/historical_model_family_differences.pdf`.
- **Appendix Table D1 (existing): Full annual series.** Format
  `output/tables/historical_model_family_annual.csv`, retaining sample counts and convergence
  flags but omitting internal model IDs from the printed table.

### 6. Support, robustness, and limitations

#### 6.1 Historical support

Summarize the two most important findings:

- the historical county-value-to-loan distribution is lower and more compressed than the
  2004--2007 source distribution; and
- loan-type and purchaser-type composition changes substantially, with many pre-2004
  level-year shares outside the source-year envelope.

Frame these as transportability diagnostics, not formal rejection tests or reasons to trim the
historical sample after seeing results.

#### 6.2 Robustness hierarchy

Organize robustness checks by the assumption they probe:

1. **Functional form:** unrestricted boosting versus unrestricted logistic.
2. **External-data dependence:** HMDA-only variants versus unrestricted models.
3. **Estimator family:** fixed mixture-adjusted Random Forest.
4. **Probability interpretation:** raw versus mixture-adjusted logistic.
5. **Threshold and subgroup behavior:** frozen-threshold and calibration diagnostics.

Avoid presenting the robustness models as an open model search. Their specifications are
frozen and their role is sensitivity analysis.

#### 6.3 Identification limits

State plainly that no reviewed public series directly measures annual subordinate-lien
origination-count shares in the paper's owner-occupied home-purchase HMDA sample. External
HELOC, junior-lien balance, and housing-stock series provide qualitative plausibility context
only. Neither historical smoothness nor the close 2004 fit identifies pre-2004 accuracy.

Discuss sample selection from the balanced FHFA merge and missing inputs. Note that the
historical finite-feature exclusions are negligible, while the county-panel restriction is the
more important sample-definition choice.

#### Assets

- **Figure 4 (new paper figure): Historical support diagnostics.** Two or three panels should
  show the annual standardized mean and dispersion of `log_lti` and
  `log_county_value_to_loan`, with the 2004--2007 source envelope shaded. A compact third panel
  may show selected loan-type and purchaser-type shares. Build from
  `historical_model_family_continuous_support.csv` and
  `historical_model_family_categorical_support.csv`.
- **Table 5 (new paper table): Robustness summary by assumption.** Report the principal result
  and interpretation for unrestricted boosting, HMDA-only logistic, HMDA-only boosting, fixed
  Random Forest, and raw logistic. Use the common comparison, RF mixture, and calibration
  findings; do not combine metrics that use different probability conventions without an
  explicit label.
- **Appendix Table E1 (existing): Continuous support envelope.** Use
  `historical_model_family_support_envelope.csv`.
- **Appendix Table E2 (existing): Categorical support.** Use
  `historical_model_family_categorical_support.csv`.
- **Appendix Table E3 (pipeline asset): Random Forest robustness.** Report reverse and forward
  results from `rf_mixture_*` tables.
- **Appendix Table E4 (existing public input): External-series comparability.** Use
  `step8_external_source_comparison.csv` to document why each public series differs from the
  target estimand.

### 7. Conclusion

Reiterate four points without introducing new estimates:

1. the known-source-prior mixture adjustment is necessary for annual share estimation under
   large prevalence shifts;
2. the unrestricted ridge logistic is the baseline because it combines strong labeled-period
   performance with a transparent extrapolation rule;
3. boosting's reverse-time gains and early historical divergence remain important robustness
   evidence; and
4. pre-2004 estimates are useful but conditional on backward transportability that HMDA cannot
   directly verify.

No new table or figure is needed in the conclusion.

## Proposed appendix structure

### Appendix A. Data construction and county-value scaling

- complete annual sample attrition;
- balanced versus year-specific county coverage;
- Zillow/FHFA scale construction and support;
- source-vintage harmonization; and
- missing-feature exclusions.

### Appendix B. Estimator details and model selection

- logistic feature matrix and spline construction;
- equal-source-prior weighting and mixture likelihood derivation;
- coarse and refinement grids;
- selected hyperparameters; and
- artifact metadata and reproducibility contract.

### Appendix C. Complete labeled-period diagnostics

- all reverse cells and horizon summaries;
- all forward years;
- calibration intercepts and slopes;
- reliability bins and tail diagnostics; and
- threshold and subgroup results.

### Appendix D. Historical estimates and support

- full 1990--2016 annual table;
- model-family difference figure;
- continuous and categorical support tables;
- 2003--2004 boundary table; and
- external-source comparability matrix.

### Appendix E. Robustness estimators

- HMDA-only logistic and boosting;
- fixed mixture-adjusted Random Forest;
- raw versus mixture-adjusted probabilities; and
- a crosswalk showing which assumption each robustness exercise changes.

## Main-text asset budget

For a short letter, target four figures and five tables at most:

| Asset | Topic | Status |
|---|---|---|
| Figure 1 | Data and validation timeline | New |
| Figure 2 | Reverse-horizon and forward-year model comparison | Existing panels; combine/restyle |
| Figure 3 | Historical 1990--2016 model-family shares | Existing; restyle for paper |
| Figure 4 | Historical continuous and categorical support | New composite |
| Table 1 | Sample construction and variables | New paper table from existing pipeline outputs |
| Table 2 | Estimator and validation protocol | New metadata/protocol summary |
| Table 3 | Common labeled-period performance | Existing data; paper formatting needed |
| Table 4 | Historical levels, boundary, and model sensitivity | New composite from existing tables |
| Table 5 | Robustness summary by identifying assumption | New synthesis table |

If the journal's letter format is tighter, move Figure 4 and Table 5 to the appendix. The
minimum viable main-text package is then Figure 1, Figure 2, Figure 3, Table 1, Table 3, and
Table 4.

## Asset-production priorities

1. Create the paper-facing Table 3 and combined Figure 2 first; these lock the model-choice
   narrative.
2. Restyle Figure 3 and construct Table 4 from the completed Step 10 outputs.
3. Regenerate or verify the sample-audit outputs needed for Table 1.
4. Build the support composite in Figure 4.
5. Construct Tables 2 and 5 as reproducible summaries rather than hand-entered LaTeX values.
6. Only then populate appendix tables and figures from the detailed pipeline outputs.

All paper assets should be generated from `src/hmda_seconds/` entry points through thin scripts
and Make targets. Paper-facing tables may reformat or combine approved aggregate outputs, but
they must not independently recompute estimators or retain loan-level diagnostic data.
