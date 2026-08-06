# Codebase Streamlining & Redundancy Audit Suggestions

This document presents a comprehensive audit of the Python source code located in `src/hmda_seconds/` and its subdirectory `src/hmda_seconds/density_ratio/`. The goal of this audit is to identify redundant logic, duplicated abstractions, parallel code paths, and opportunities for streamlining without disrupting frozen estimator protocols or validation contracts.

---

## 1. Executive Summary

The `src/` codebase is well-structured and highly robust, providing clear separation of empirical estimation stages. However, as the repository evolved to support density-ratio cross-family protocols, staged histogram gradient boosting, known-source-prior mixture estimation, and out-of-time validation, several parallel implementations and copy-pasted utility patterns emerged.

Key audit findings:
- **Parallel Estimator Architectures**: Model families (`logistic`, `gradient_boosting`, `random_forest`) are implemented twice—once as standalone top-level modules (`mixture.py`, `gradient_boosting.py`, `random_forest_mixture.py`, `logistic.py`) and once inside `density_ratio/families/` wrapped by `adapters.py`.
- **Duplicated Data Cleaning & Panel Loading**: Data loading, sample filtering, and Zillow-scaled FHFA county value merging are re-implemented across `clean.py`, `model_selection.py`, `plausibility.py`, `classify.py`, and `benchmark.py`.
- **Repeated Temporal Fold Wrappers & Grid Loops**: Legacy `ReverseFold` wrappers and temporal fold evaluation loops are independently written across six separate selection and diagnostic modules.
- **Diagnostic Metrics & Checkpoint Utility Copy-Pasting**: Metric calculation, reliability binning, and CSV cell replacement/checkpointing logic (`_upsert_metric_checkpoint`, `_replace_cell`, `_append_checkpoint`) are duplicated across `calibration.py`, `mixture_calibration.py`, `threshold_diagnostics.py`, and `plausibility.py`.
- **Artifact Serialization Boilerplate**: Sidecar JSON metadata construction and payload digest validation are repeated in each top-level model module instead of fully delegating to `density_ratio.artifacts`.

---

## 2. Detailed Findings & Streamlining Suggestions

### Category 1: Estimator Architecture & Model Adapter Layer Redundancy

#### 1.1 Dual Implementation of Model Fitting and Persistence
- **Affected Files**:
  - [`src/hmda_seconds/mixture.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/mixture.py#L240-L380)
  - [`src/hmda_seconds/gradient_boosting.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/gradient_boosting.py#L650-L750)
  - [`src/hmda_seconds/random_forest_mixture.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/random_forest_mixture.py#L320-L400)
  - [`src/hmda_seconds/density_ratio/families/logistic.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/density_ratio/families/logistic.py#L30-L90)
  - [`src/hmda_seconds/density_ratio/families/gradient_boosting.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/density_ratio/families/gradient_boosting.py#L37-L80)
  - [`src/hmda_seconds/density_ratio/families/random_forest.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/density_ratio/families/random_forest.py#L30-L67)
- **Issue**: Each model family has two distinct entry points for fitting, checking file existence, loading existing models, creating sidecar metadata, and saving pickles. For example, `density_ratio/families/logistic.py` manually checks `path.exists()`, calls `mixture.load_known_source_prior_model`, checks `artifacts.load_metadata`, calls `mixture_logistic_selection.fit_candidate_path`, and then wraps the result using `adapters.adapt_known_source_prior_model`.
- **Streamlining Suggestion**:
  - Consolidate model fitting and persistence into the `density_ratio/families/` abstractions.
  - Make top-level functions (e.g. `mixture.fit_known_source_prior_model`) thin wrappers over `density_ratio.families.LogisticFamily` rather than duplicating the file existence, metadata sidecar, and caching logic.

#### 1.2 Adapter Wrapper Overhead (`adapters.py`)
- **Affected Files**:
  - [`src/hmda_seconds/density_ratio/adapters.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/density_ratio/adapters.py#L25-L89)
- **Issue**: `ExistingFittedModelAdapter` wraps `KnownSourcePriorModel`, `BoostingDensityRatioModel`, and `RandomForestDensityRatioModel` to adapt them to the `FittedDensityRatioModel` protocol. The underlying dataclasses already store `train_years` and implement `log_ratio(frame)`.
- **Streamlining Suggestion**:
  - Have `KnownSourcePriorModel`, `BoostingDensityRatioModel`, and `RandomForestDensityRatioModel` directly implement the `FittedDensityRatioModel` protocol (adding a `model_id` property or field directly on the dataclass).
  - Eliminate `ExistingFittedModelAdapter` and `adapters.py` entirely, reducing runtime indirection and simplifying type annotations.

---

### Category 2: Data Cleaning, Preprocessing & County Value Panels

#### 2.1 Re-reading and Cleaning Parquet Files Across Modules
- **Affected Files**:
  - [`src/hmda_seconds/clean.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/clean.py#L146-L172) (`load_and_clean_year`)
  - [`src/hmda_seconds/model_selection.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/model_selection.py#L112-L142) (`prepare_selection_data`)
  - [`src/hmda_seconds/plausibility.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/plausibility.py#L223-L233) (`_load_and_clean_application_year`)
  - [`src/hmda_seconds/benchmark.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/benchmark.py#L90-L95)
  - [`src/hmda_seconds/classify.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/classify.py#L50-L60)
- **Issue**:
  - `plausibility._load_and_clean_application_year` manually inspects schema columns with `pyarrow.parquet.read_schema(path).names` and drops `label_var` if missing before passing to `clean.clean_frame`.
  - `model_selection.prepare_selection_data` reads, cleans, and saves selection subsets.
  - `clean.build_county_value_panel(config.APPLY_YEARS)` is called independently in 4 different files, rebuilding or re-merging county value DataFrames repeatedly during script execution.
- **Streamlining Suggestion**:
  - Centralize all annual raw loading and cleaning logic into `clean.load_and_clean_year(year, ...)` with built-in schema tolerance for pre-2004 historical LAR extracts.
  - Add `@functools.lru_cache` or standard memoization to `clean.build_county_value_panel` so county value index panels are built once per process.

#### 2.2 Duplicate Categorical Level Pinning Logic
- **Affected Files**:
  - [`src/hmda_seconds/clean.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/clean.py#L140-L141)
  - [`src/hmda_seconds/logistic.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/logistic.py#L157-L173) (`_pin_category_levels`)
- **Issue**: Both `clean.py` and `logistic.py` iterate over `config.CATEGORY_LEVELS` to convert columns into `pd.Categorical` with explicit categories.
- **Streamlining Suggestion**:
  - Export a single utility `clean.pin_category_levels(df)` or ensure `clean.clean_frame` is the sole place where categorical pinning occurs. Remove `logistic._pin_category_levels`.

---

### Category 3: Temporal Folds & Selection / Diagnostic Loops

#### 3.1 Legacy `ReverseFold` Class vs `density_ratio.folds.TemporalFold`
- **Affected Files**:
  - [`src/hmda_seconds/model_selection.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/model_selection.py#L39-L97) (`ReverseFold` & `reverse_folds`)
  - [`src/hmda_seconds/density_ratio/folds.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/density_ratio/folds.py#L10-L37)
- **Issue**: `model_selection.py` defines a subclass `ReverseFold(TemporalFold)` and a wrapper function `reverse_folds()` whose sole purpose is to convert `TemporalFold` instances back and forth for legacy signatures.
- **Streamlining Suggestion**:
  - Deprecate `ReverseFold` in favor of directly using `density_ratio.folds.TemporalFold` across `model_selection.py` and downstream consumers.

#### 3.2 Repetitive Reverse-Time Validation Loop Boilerplate
- **Affected Files**:
  - [`src/hmda_seconds/calibration.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/calibration.py#L210-L310) (`_run_reverse_diagnostics`)
  - [`src/hmda_seconds/mixture_calibration.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/mixture_calibration.py#L110-L190) (`_run_design`)
  - [`src/hmda_seconds/threshold_diagnostics.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/threshold_diagnostics.py#L160-L240)
  - [`src/hmda_seconds/gradient_boosting.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/gradient_boosting.py#L380-L460)
  - [`src/hmda_seconds/random_forest_mixture.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/random_forest_mixture.py#L130-L210)
- **Issue**: Each of these 5 modules implements a outer loop over `reversed(temporal_folds.reverse_folds())` and an inner loop over `fold.validation_years`. Inside the loop, each script performs cell completion checks, model fitting/loading, target evaluation, and incremental CSV checkpointing.
- **Streamlining Suggestion**:
  - Create a generic fold execution orchestrator in `density_ratio.runner` or `density_ratio.pipeline` (e.g. `run_temporal_evaluation(folds, model_family, evaluator, checkpoint_dir)`).
  - Let diagnostic scripts pass an evaluation callback function rather than writing custom nested loops for each model family.

---

### Category 4: Evaluation Metrics, Calibration & Checkpoint Helpers

#### 4.1 Metric Pass-Through Wrappers
- **Affected Files**:
  - [`src/hmda_seconds/calibration.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/calibration.py#L31-L33) (`probability_metrics`)
  - [`src/hmda_seconds/density_ratio/evaluation.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/density_ratio/evaluation.py#L120-L156) (`probability_metrics`)
- **Issue**: `calibration.probability_metrics` is a 3-line wrapper function that simply delegates to `evaluation.probability_metrics`.
- **Streamlining Suggestion**:
  - Re-export `probability_metrics` directly in `calibration.py` via `from .density_ratio.evaluation import probability_metrics` or update caller imports.

#### 4.2 Duplicate CSV Checkpoint & Cell Replacement Logic
- **Affected Files**:
  - [`src/hmda_seconds/calibration.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/calibration.py#L379-L432) (`_append_checkpoint`, `_metric_complete`, `_upsert_metric_checkpoint`)
  - [`src/hmda_seconds/mixture_calibration.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/mixture_calibration.py#L221-L260) (`_cell_complete`, `_read`, `_upsert`, `_replace_cell`)
  - [`src/hmda_seconds/threshold_diagnostics.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/threshold_diagnostics.py#L390-L430) (`_append_checkpoint`, `_upsert_checkpoint`)
  - [`src/hmda_seconds/plausibility.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/plausibility.py#L242-L254) (`_year_complete`, `_upsert_year`)
- **Issue**: 4 separate modules implement nearly identical functions for reading existing CSV files, checking if a `(train_start, validation_year)` or `year` row exists, filtering out old rows, and appending/upserting new rows.
- **Streamlining Suggestion**:
  - Extract a single shared utility module or class `src/hmda_seconds/checkpointing.py` containing `upsert_cell_csv(path, new_df, key_columns)` and `is_cell_complete(path, key_columns, key_values)`.

---

### Category 5: Model Artifact Persistence & Metadata Sidecars

#### 5.1 Metadata Dictionary Construction Boilerplate
- **Affected Files**:
  - [`src/hmda_seconds/mixture.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/mixture.py#L320-L340) (`save_known_source_prior_model`)
  - [`src/hmda_seconds/gradient_boosting.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/gradient_boosting.py#L640-L670) (`save_boosting_model`)
  - [`src/hmda_seconds/random_forest_mixture.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/random_forest_mixture.py#L310-L335) (`save_forest_model`)
  - [`src/hmda_seconds/logistic.py`](file:///home/dan/research/hmda-second-liens/src/hmda_seconds/logistic.py#L110-L134) (`save_selected_model`)
- **Issue**: Each module manually instantiates `ModelConfiguration` and `ModelArtifactMetadata`, computes training counts, calls `density_ratio.artifacts.build_metadata`, and invokes `artifacts.save_pickle_artifact`.
- **Streamlining Suggestion**:
  - Create a helper `artifacts.save_model_artifact(model, path, configuration, training_counts)` that encapsulates sidecar generation, payload hashing, and atomic writing for all model classes.

---

## 3. Recommended Refactoring Roadmap

To preserve all frozen contracts, migration plans, and test suites, refactoring should be executed in phased stages:

| Phase | Target Area | Primary Goal | Risk / Test Verification |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **CSV Checkpointing Utility** | Create `src/hmda_seconds/checkpointing.py` and refactor `calibration.py`, `mixture_calibration.py`, `threshold_diagnostics.py`, and `plausibility.py`. | Low risk. Verified by `pytest tests/`. |
| **Phase 2** | **Data Loading & Categorical Pinning** | Standardize `clean.load_and_clean_year` and add caching to `clean.build_county_value_panel`. Remove `logistic._pin_category_levels`. | Low risk. Verified by `pytest tests/test_clean.py`. |
| **Phase 3** | **Fitted Model Protocol & Adapters** | Implement `FittedDensityRatioModel` directly on model dataclasses; streamline `adapters.py`. | Medium risk. Verified by `pytest tests/test_density_ratio*.py`. |
| **Phase 4** | **Family Fitting Consolidation** | Unify top-level fit/save functions in `mixture.py`, `gradient_boosting.py`, and `random_forest_mixture.py` with `density_ratio/families/`. | Medium risk. Verified by `make select-logistic`, `make evaluate-gradient-boosting`. |
| **Phase 5** | **Reverse Validation Loop Orchestration** | Refactor repetitive fold loops in diagnostic modules to use a shared runner from `density_ratio.runner`. | Medium risk. Verified by diagnostic script outputs. |

---

## 4. Conclusion

By implementing these streamlining suggestions, the `src/hmda_seconds` codebase can significantly reduce duplicate lines of code, simplify model serialization/loading pipelines, eliminate redundant I/O, and improve long-term maintainability while keeping all empirical results, protocols, and artifacts 100% deterministic and reproducible.
