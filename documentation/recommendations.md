# Recommendations for HMDA Second-Lien Classification

After reviewing the full codebase — feature construction, Random Forest and Logistic Regression classifiers, validation framework, and imputation pipeline — here are recommendations organized from highest to lowest expected impact.

## Current Setup at a Glance

| Aspect | Detail |
|---|---|
| **Features** | `log_lti` $(\ln(\text{loan\_amt}/\text{income}))$, `log_ltv` $(\ln(\text{hpi}/\text{loan\_amt}))$, `purchaser_type` (10 levels), `loan_type` (4 levels) → 16 encoded features |
| **Random Forest** | `n_estimators=50`, `max_depth=10`, `class_weight=None`, trained on a **25% random split** of 2004–2007 |
| **Logistic Regression** | `C=1.0`, `l1_ratio=0.0`, `solver=lbfgs`, trained on **100%** of 2004–2007 |
| **Validation** | Out-of-time on 2008–2016; McNemar's test shows logistic **significantly outperforms** RF in every validation year |
| **Key finding** | Overall accuracy 97.9–98.9%, AUC 0.988–0.995, but **second-lien precision drops to 53–66%** out-of-time as the base rate collapses from ~23% (2006) to ~2% post-2007 |

---

## 1. Probability Calibration

**Problem.** The core deliverable is the imputed `prob_second_lien` for 1990–2003, which likely feeds downstream aggregate second-lien share calculations. Both RF and logistic probabilities can be miscalibrated out-of-sample — RF probabilities especially tend to cluster toward 0.5 and understate tail probabilities. The sharp base-rate shift between training (23% second liens in 2006) and application (much lower in the 1990s) makes uncalibrated probabilities particularly dangerous.

**Recommendation.** Add a post-hoc calibration step (e.g., isotonic regression or Platt scaling via `sklearn.calibration.CalibratedClassifierCV`). The natural calibration set is the 75% test split already held out in RF training, or a dedicated 2007 holdout year. Report **calibration curves** (reliability diagrams) alongside AUC/F1 in the validation output.

**Why it matters.** If the paper's analysis depends on *aggregate second-lien shares* by year (which is what the LTI histogram diagnostics suggest), biased probabilities will directly distort those shares even if discrimination (AUC) is excellent.

---

## 2. Temporal Domain Shift — The Central Threat to Validity

**Problem.** The model trains on 2004–2007 (the peak of the second-lien boom) and is applied to 1990–2003, when second liens were far rarer and mortgage market structure was fundamentally different. Features that discriminate well during the boom (e.g., high `log_lti` ratios paired with certain `purchaser_type` values) may carry weaker or different signals in earlier, more conservative lending environments. The out-of-time validation already shows this: second-lien precision collapses from ~66% to ~53% as the post-crisis base rate falls.

**Recommendations.**

- **Surface the year-by-year validation prominently.** The code already computes per-year metrics — the paper should show these, especially the precision/recall degradation in 2011–2016 when the base rate is lowest (closest analog to the 1990s application environment).
- **Plausibility checks against external aggregates.** Compare imputed second-lien shares for 1990–2003 to external estimates (e.g., LPS/Black Knight, CoreLogic, NY Fed Consumer Credit Panel). Even ballpark consistency is reassuring; a large discrepancy is a red flag.
- **Consider training on a broader year range.** If 2008+ `lien_status` data is reliable, training on 2004–2012 and validating on 2013–2016 would expose the model to a wider range of market conditions and base rates, potentially improving generalization to the low-prevalence 1990s.
- **The 2004 continuity check is well-designed** — make sure the paper highlights whether predicted second-lien shares exhibit a suspicious jump at the 2003/2004 boundary.

---

## 3. Feature Engineering

The current feature set is lean (2 continuous + 2 categorical = 16 encoded). Some additions could meaningfully improve discrimination, especially for the backward imputation task:

| Candidate Feature | Rationale | Source |
|---|---|---|
| **Cumulative HPI appreciation** (1-yr or 3-yr growth) | Current `log_ltv` uses the HPI *level*, but *recent appreciation* is the economic driver of equity extraction and second-lien demand. | FHFA county HPI (already loaded) |
| **State or MSA fixed effects** | Second-lien prevalence varies sharply by geography (sand states vs. heartland). Even coarse regional dummies would help. | HMDA `state_code`, `msa_md` |
| **Interaction: `log_lti` × `log_ltv`** | High LTI in a high-LTV (low HPI) environment has a different lien-status signal than high LTI with high HPI. | Constructed |
| **`tract_to_msa_income`** | Neighborhood income relative to MSA proxies for loan characteristics correlated with piggyback lending. | HMDA (already in raw data) |
| **Lender identity or lender-size bins** | Some lenders specialized in second liens. High-cardinality, but aggregation to size/type bins is feasible. | HMDA `resp_id` + panel data |

> [!IMPORTANT]
> **Year fixed effects** are tempting but dangerous: the model must extrapolate to years outside the 2004–2007 training range, and a year indicator would become meaningless for 1990–2003.

---

## 4. Gradient Boosting as a Likely-Better Model

**Problem.** The McNemar tests already show the logistic model outperforms the RF out-of-time. This is consistent with the broader ML literature: when the feature set is small and the task benefits from well-calibrated probabilities, gradient-boosted trees tend to dominate random forests. Boosting's sequential error correction is better suited to this setting than bagging.

**Recommendation.** Add `sklearn.ensemble.HistGradientBoostingClassifier` (or LightGBM) as a third model. Key advantages:
- **Better calibration** than RF out of the box.
- **Native handling of NaN values** (`HistGradientBoostingClassifier` handles missing data internally, avoiding the need to drop rows).
- **Native categorical support** in LightGBM, avoiding the one-hot expansion.
- The validation framework is already model-agnostic, so integration is straightforward.

Since the logistic model already beats the RF, a well-tuned gradient boosting model is likely to beat both — or at minimum provide a useful diverse model for ensembling.

---

## 5. Hyperparameter Tuning

**Problem.** Both models use fixed hyperparameters with no evidence of systematic tuning. The RF uses `n_estimators=50` (quite low) and `max_depth=10` with `class_weight=None`. The logistic model uses `C=1.0` with no regularization search.

**Recommendations.**
- **RF:** Run a grid or random search over `{n_estimators, max_depth, min_samples_leaf, class_weight}` using time-series-aware CV (e.g., train on 2004–2005, validate on 2006; train on 2004–2006, validate on 2007). The existing `hyperparameter_robustness` function in [validate.py](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/validate.py) already sweeps `n_estimators` and `max_depth` — use this to select values, rather than treating it as a post-hoc diagnostic.
- **Logistic:** Tune `C` via CV. Test `penalty='elasticnet'` with `l1_ratio` search, which may help if some of the one-hot dummies are noise.
- At minimum, **increase `n_estimators`** for the RF. 50 trees is low; 200–500 typically gives a meaningful boost for little extra cost.

---

## 6. Class Imbalance Strategy

**Problem.** The current RF uses `class_weight=None`, and the migration plan notes that `class_weight='balanced'` was tried and rejected because it increased false positives out-of-time. But the base-rate collapse (23% → 2%) is the core driver of the out-of-time precision drop, and the default weighting doesn't address this.

**Recommendations.**
- **`class_weight='balanced_subsample'`** (for RF) is a less aggressive alternative to `'balanced'` that may improve recall without destroying precision — worth a test.
- **Threshold tuning** (see §8) is a more principled way to handle the precision–recall trade-off than class weighting.
- **Cost-sensitive learning**: if the downstream application has an asymmetric cost (e.g., misclassifying a first lien as a second lien is worse than the reverse), encode that directly in the loss rather than using generic balanced weighting.

---

## 7. Missing Data and Selection Bias

**Problem.** The current pipeline drops records that fail the inner join with the balanced FHFA panel (missing county HPI for any year in 1990–2016) and drops records with missing income, loan amount, or other features. If missingness correlates with lien status, this introduces selection bias in both training and imputation.

**Recommendations.**

- **Quantify the dropout rate** at each filter step, by year and by lien status (for 2004–2007 where truth is known). Report this as a table in the paper. If the balanced-panel requirement drops a large share of counties, consider relaxing it to a year-specific merge.
- **For tree-based models**, consider imputing missing features rather than dropping records. `HistGradientBoostingClassifier` handles NaN natively.
- **For the logistic model**, missing-indicator dummies + median fill is a standard approach.

---

## 8. Threshold Selection

**Problem.** Hard classifications use the default 0.5 threshold. With severe class imbalance (especially out-of-time when second liens are ~2%), the optimal threshold is far from 0.5.

**Recommendation.**
- For aggregate analysis, use the *probability* directly (weighted aggregation of `prob_second_lien`) rather than hard classification. This avoids the threshold problem entirely.
- If hard labels are needed, tune the threshold on validation data to match the observed second-lien share in each year, or to maximize a cost-weighted metric. The existing per-year validation infrastructure supports this.

---

## 9. Model Interpretability

**Problem.** The paper needs to convince readers that the imputation is credible. The feature importance plot is already saved for the RF, but more could be done.

**Recommendations.**

- **Report logistic regression coefficients** directly — these are immediately interpretable and can go in the paper as a table.
- **Partial dependence plots** for `log_lti` and `log_ltv` (the two continuous features) would show readers *how* the model uses these features and whether the relationships are sensible.
- **Permutation importance** (for the RF) is less biased than Gini importance, especially with correlated features and dummies of different cardinality.

---

## 10. Validation Enhancements

A few additions to the already-strong validation framework:

- **Calibration plots** (reliability diagrams): Bin predicted probabilities and plot against observed frequency. This directly assesses whether `prob_second_lien = 0.3` really means 30% of such loans are second liens.
- **Precision-recall curves**: More informative than ROC when the positive class (second liens) is rare, which is the case for both the late validation years and the 1990s imputation target.
- **Subgroup analysis**: Report metrics broken out by `loan_type` and `purchaser_type` (the two categoricals). The model may perform well overall but fail for specific subpopulations important to the paper's analysis.
- **Predicted-share time series**: Plot the predicted second-lien share for all 1990–2016 years on a single chart (overlaying actual shares for 2004–2016). This would immediately surface any implausible jumps or trends in the imputed period.

---

## Summary Table

| # | Recommendation | Effort | Expected Impact | Risk if Ignored |
|---|---|---|---|---|
| 1 | Probability calibration | Low | **High** | Biased aggregate second-lien shares |
| 2 | Temporal domain-shift analysis | Low–Med | **High** | Imputed 1990s shares may be unreliable |
| 3 | Feature engineering | Medium | Med–High | Leaving accuracy on the table |
| 4 | Gradient boosting model | Low–Med | Medium | Missing a model that likely beats both RF and logistic |
| 5 | Hyperparameter tuning | Low | Low–Med | Suboptimal model (but likely close) |
| 6 | Class imbalance strategy | Low | Low–Med | Precision/recall trade-off not optimized |
| 7 | Missing data handling | Medium | Medium | Selection bias in imputed sample |
| 8 | Threshold selection | Low | Low–Med | Misclassification at the margin |
| 9 | Interpretability | Low | Medium | Reviewer skepticism |
| 10 | Validation enhancements | Low | Medium | Blind spots in model evaluation |
