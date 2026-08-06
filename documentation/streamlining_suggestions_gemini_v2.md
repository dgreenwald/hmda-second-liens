# Codebase Streamlining & Redundancy Audit Suggestions (Round 2)

Following the recent major streamlining steps (steps 7 & 8), over 2,500 lines of legacy code, unused scripts, duplicate estimators, and redundant test suites were cleanly removed. Shared utilities like `density_ratio/checkpoints.py` and `density_ratio/diagnostics.py` were introduced, and core estimator fitting logic was moved into `density_ratio/families/`.

This document outlines **Round 2 streamlining suggestions** focusing on remaining redundancies, code simplification opportunities, and architectural polish across `src/hmda_seconds/`.

---

## 1. Executive Summary & Current State

The codebase is now significantly leaner, modular, and maintainable. 

**Key Refactorings Completed in Recent Steps**:
- Removed legacy modules (`benchmark.py`, `classify.py`, `diagnostics.py`, `logistic.py`, `train.py`, `validate.py`, `adapters.py`) and unused CLI scripts.
- Relocated model fitting and dataclasses into `density_ratio/families/` (`logistic.py`, `gradient_boosting.py`, `random_forest.py`).
- Centralized CSV checkpoint reading and row-matching into `density_ratio/checkpoints.py`.

**Remaining Refactoring Opportunities**:
1. **Private Checkpoint Wrapper Functions**: Multiple top-level modules retain thin 4-line private wrapper functions (`_cell_complete`, `_metric_complete`, `_cell_present`, `_diagnostic_cell_present`, `_diagnostic_cell_present`) around `checkpoints.rows_present(...)`.
2. **Duplicate Probability Indexing & Validation**: Repeated `list(classifier.classes_).index(config.SECOND_LIEN_CLASS)` lookups across 5 files, and duplicated `_log_mean_exp` / `_finite_vector` helper functions.
3. **Data Loading & Preprocessing Alignment**: `audit.py` and `plausibility.py` retain standalone parquet loading, column filtering, and FIPS calculation logic that duplicates `clean.load_and_clean_year` and `clean.load_fhfa_county_hpi`.
4. **Repeated Two-Stage Horizon Aggregation**: Five separate modules implement custom `groupby` loops for the protocol's two-stage horizon aggregation (equal-weighted within horizon, then equal-weighted across horizons).
5. **Pairwise Model Comparison Joins**: Four separate functions in challenger modules implement nearly identical baseline vs. challenger CSV merges on `(train_start, validation_year, horizon)`.
6. **3-by-3 Subplot Grid Layout Boilerplate**: Duplicate Matplotlib figure initialization, axis titration, and legend formatting between reliability and precision-recall diagrams.

---

## 2. Detailed Findings & Round 2 Suggestions

### Category 1: Model Prediction, Indexing & Utility Validation

#### 1.1 Standardize Second-Lien Class Probability Lookup
- **Affected Files**:
  - [`src/hmda_seconds/model_selection.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/model_selection.py#L52-L55) (`predict_proba_second_lien`)
  - [`src/hmda_seconds/mixture.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/mixture.py#L66-L70) (`raw_probability`)
  - [`src/hmda_seconds/threshold_diagnostics.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/threshold_diagnostics.py#L398-L399)
  - [`src/hmda_seconds/plausibility.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/plausibility.py#L138-L141)
- **Issue**: Extracting second-lien class probabilities from a fitted scikit-learn classifier via `list(classifier.classes_).index(config.SECOND_LIEN_CLASS)` and indexing `predict_proba(X)[:, col]` is repeated across 5 separate modules.
- **Streamlining Suggestion**: Add a shared helper `extract_second_lien_proba(classifier, X)` in `density_ratio/evaluation.py` and use it uniformly.

#### 1.2 Deduplicate Validation Helpers (`_log_mean_exp` and `_finite_vector`)
- **Affected Files**:
  - [`src/hmda_seconds/mixture.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/mixture.py#L648-L660) (`_log_mean_exp`, `_finite_vector`)
  - [`src/hmda_seconds/density_ratio/families/logistic.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/density_ratio/families/logistic.py#L184-L189) (`_log_mean_exp`)
  - [`src/hmda_seconds/density_ratio/evaluation.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/density_ratio/evaluation.py#L186-L192) (`_finite_vector`)
- **Issue**: Both `_log_mean_exp` and `_finite_vector` are defined identically in two different modules.
- **Streamlining Suggestion**: Promote `log_mean_exp` and `validate_finite_vector` to `density_ratio/evaluation.py` and import them directly.

---

### Category 2: Data Cleaning, Preprocessing & Raw Extracts

#### 2.1 Reuse `clean.load_fhfa_county_hpi` in `audit.py`
- **Affected Files**:
  - [`src/hmda_seconds/clean.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/clean.py#L51-L58) (`load_fhfa_county_hpi`)
  - [`src/hmda_seconds/audit.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/audit.py#L269-L271)
- **Issue**: `audit.py` directly invokes `fhfa.load("county", ...)` and manually formats `year` and `fips` columns, duplicating the logic already wrapped in `clean.load_fhfa_county_hpi`.
- **Streamlining Suggestion**: Update `audit.run_sample_audit()` to call `clean.load_fhfa_county_hpi()`.

#### 2.2 Consolidate Historical Parquet Extract Loading
- **Affected Files**:
  - [`src/hmda_seconds/clean.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/clean.py#L146-L153) (`load_and_clean_year`)
  - [`src/hmda_seconds/plausibility.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/plausibility.py#L223-L232) (`_load_and_clean_application_year`)
- **Issue**: `plausibility._load_and_clean_application_year` inspects parquet schema names with `pyarrow.parquet.read_schema(path).names`, filters columns, drops pre-2004 `lien_status`, and calls `clean.clean_frame`.
- **Streamlining Suggestion**: Enhance `clean.load_and_clean_year` to accept an optional `columns` parameter and handle pre-2004 label stripping, replacing `plausibility._load_and_clean_application_year`.

---

### Category 3: Diagnostic Executions & Cell Checkpointing

#### 3.1 Eliminate Private Checkpoint Wrapper Functions
- **Affected Files**:
  - [`src/hmda_seconds/calibration.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/calibration.py#L366-L370) (`_bin_complete`)
  - [`src/hmda_seconds/mixture_calibration.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/mixture_calibration.py#L217-L230) (`_cell_complete`)
  - [`src/hmda_seconds/gradient_boosting.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/gradient_boosting.py#L553-L560) (`_diagnostic_cell_present`)
  - [`src/hmda_seconds/mixture.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/mixture.py#L663-L675) (`_cell_present`, `_fold_present`)
  - [`src/hmda_seconds/random_forest_mixture.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/random_forest_mixture.py#L294-L300) (`_cell_present`)
- **Issue**: Six separate top-level modules define 4-line private wrapper functions that simply call `checkpoints.rows_present(frame, {"train_start": train_start, "validation_year": validation_year})`.
- **Streamlining Suggestion**: Add `checkpoints.cell_present(df, train_start, validation_year)` to `density_ratio/checkpoints.py` and replace all 6 private wrapper functions.

#### 3.2 Unify Pairwise Model Comparison Joins
- **Affected Files**:
  - [`src/hmda_seconds/gradient_boosting.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/gradient_boosting.py#L514-L540) (`compare_with_logistic`)
  - [`src/hmda_seconds/random_forest_mixture.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/random_forest_mixture.py#L204-L228) (`estimator_comparison`)
  - [`src/hmda_seconds/mixture_logistic_selection.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/mixture_logistic_selection.py#L421-L446) (`compare_estimators`)
  - [`src/hmda_seconds/model_selection.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/model_selection.py#L787-L826) (`geographic_incremental_brier`)
- **Issue**: Each function loads baseline metric CSVs, performs a 1-to-1 merge on `["train_start", "validation_year", "horizon"]`, and computes difference columns (`challenger_brier - baseline_brier`).
- **Streamlining Suggestion**: Export a single `compare_cell_metrics(base_df, challenger_df, metric_col="brier_score", join_keys=...)` helper function in `density_ratio/evaluation.py`.

---

### Category 4: Aggregation Protocols & Plotting Utilities

#### 4.1 Reusable Two-Stage Horizon Aggregation Helper
- **Affected Files**:
  - [`src/hmda_seconds/model_selection.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/model_selection.py#L334-L368) (`aggregate_brier_cells`)
  - [`src/hmda_seconds/gradient_boosting.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/gradient_boosting.py#L113-L146) (`aggregate_brier`)
  - [`src/hmda_seconds/mixture_logistic_selection.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/mixture_logistic_selection.py#L173-L206) (`aggregate_candidates`)
  - [`src/hmda_seconds/mixture.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/mixture.py#L454-L504) (`aggregate_share_errors`)
  - [`src/hmda_seconds/threshold_diagnostics.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/threshold_diagnostics.py#L220-L243) (`aggregate_threshold_metrics`)
- **Issue**: The project's evaluation protocol mandates averaging metric cells equally within horizon, then equally across horizons. Each of these 5 functions independently implements this 2-stage `groupby` sequence.
- **Streamlining Suggestion**: Generalize `calibration.aggregate_reverse_metrics` into a generic `aggregate_two_stage_horizon(df, group_cols, metric_cols)` function in `density_ratio/evaluation.py`.

#### 4.2 Reusable 3x3 Grid Figure Renderer
- **Affected Files**:
  - [`src/hmda_seconds/calibration.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/calibration.py#L300-L346) (`render_reliability_panels`)
  - [`src/hmda_seconds/threshold_diagnostics.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/threshold_diagnostics.py#L296-L327) (`render_precision_recall_panels`)
- **Issue**: Both functions create a 3x3 subplot grid (`plt.subplots(3, 3, figsize=(9.0, 9.0), squeeze=False)`), iterate over panel values, draw curves/diagonals, set axis labels/titles, hide unused axes (`axes.flat[len(values):]`), call `fig.tight_layout()`, and save to PDF.
- **Streamlining Suggestion**: Move `render_reliability_panels` and a generic `render_3x3_panel_grid` helper into `density_ratio/diagnostics.py`.

---

## 3. Summary Roadmap of Round 2 Streamlining Targets

| Priority | Target Area | Primary Action | Expected Benefit |
| :--- | :--- | :--- | :--- |
| **High** | **Cell Checkpointing Helpers** | Add `checkpoints.cell_present()` and replace 6 private wrapper functions across top-level modules. | Eliminates ~50 lines of duplicate boilerplate; standardizes cell checks. |
| **High** | **Probability & Math Validation** | Export `extract_second_lien_proba`, `log_mean_exp`, and `validate_finite_vector` in `density_ratio/evaluation.py`. | Removes duplicated class-index lookups and array validation logic across 6 modules. |
| **Medium** | **Comparison Joins** | Create `compare_cell_metrics` helper for challenger vs. baseline cell merges. | Unifies 4 separate pairwise CSV join functions. |
| **Medium** | **Horizon Aggregation** | Implement `aggregate_two_stage_horizon` in `density_ratio/evaluation.py`. | Consolidates 5 independent implementations of the equal-horizon aggregation protocol. |
| **Low** | **Data Loading & Plotting** | Reuse `clean.load_fhfa_county_hpi` in `audit.py` and move 3x3 panel plotters to `density_ratio/diagnostics.py`. | Clean separation of visualization and raw dataset loading. |
