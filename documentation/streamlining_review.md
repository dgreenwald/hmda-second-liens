# Codebase streamlining review

## Scope

This review examines the current Python codebase for dead code, duplicated helpers, overlapping
responsibilities, and compatibility layers that may now be removable. It is a read-only review:
no streamlining changes were made as part of the audit.

The repository currently contains approximately:

- 9,774 lines under `src/hmda_seconds/`;
- 1,273 lines under `scripts/`; and
- 3,010 lines under `tests/`.

Large modules are not automatically redundant. In particular, splitting a large module can
improve navigation without reducing total code. The recommendations below prioritize actual
duplication and unnecessary dependency layers.

## Baseline status

The Step 8 add/add conflicts identified during the initial audit have been resolved. Ruff,
Python compilation, and all 164 synthetic tests pass. Streamlining work should preserve this
baseline and continue to use the bounded real-data family-parity checks where estimator code is
moved.

## Findings

### 1. Make the density-ratio family modules the actual implementation owners

**Status: completed.**

The three family modules now own their fitted-model records, feature construction, fitting
primitives, deterministic paths, and artifact save/load functions. The former top-level
locations retain compatibility re-exports so existing imports and legacy pickle global lookups
continue to resolve.

Equal-source-prior weighting now lives in a shared density-ratio utility rather than in the
logistic mixture orchestration module. A common fitted-model artifact helper owns metadata
construction followed by atomic persistence, while each family still declares its feature
schema, weighting convention, source prior, and model identity explicitly.

The top-level modules remain responsible for grids, staged decisions, compatibility tables,
comparisons, and diagnostics.

### 2. Finish consolidating common calibration-diagnostic cell generation

**Status: completed.**

The shared `density_ratio.diagnostics.evaluate_cell` now constructs the canonical metric row
and reliability bins for raw logistic, known-source-prior logistic, boosting, and Random Forest
diagnostics. It accepts metrics already computed by `evaluation.evaluate_target`, avoiding
recalculation, and supports named family-specific extension tables. Mixture ratio tails use
that extension contract.

Workflows retain their own completion checks and checkpoint writes because those differ by
artifact set; the sample-level calculations and table schemas no longer do.

### 3. Consolidate CSV checkpoint utilities

**Status: completed.**

There are six identical helpers that read a CSV if it exists and otherwise return an empty
DataFrame. At least five additional functions implement nearly identical row replacement:

- identify rows by `train_start` and `validation_year`, sometimes with an estimator or candidate
  key;
- remove matching existing rows;
- concatenate new rows; and
- write the entire CSV.

Representative copies appear in `mixture.py`, `gradient_boosting.py`,
`random_forest_mixture.py`, `calibration.py`, `mixture_calibration.py`, and
`threshold_diagnostics.py`. `plausibility.py` uses the same pattern with `year` as its logical
key and should use the shared utility as well.

The shared `density_ratio/checkpoints.py` module now provides:

- `read_csv(path)`;
- `rows_present(frame, keys)`;
- `replace_rows(existing, new, key_columns)`; and
- atomic `write_csv` and `append_rows` operations.

The logistic, mixture, boosting, Random Forest, threshold, mixture-reselection, and plausibility
workflows use these primitives. Replacement batches must represent exactly one logical key, and
all checkpoint writes use same-directory temporary files followed by atomic replacement.

### 4. Consolidate identical forward-summary builders

**Status: completed.**

The `_simple_summary` implementation is identical in:

- `random_forest_mixture.py`;
- `mixture_logistic_selection.py`; and
- `mixture_calibration.py`.

The implementations now use `calibration.aggregate_forward_metrics`, including the previously
inline boosting variant. Forward aggregation remains distinct from the reverse protocol's
equal-within-horizon, then equal-across-horizon weighting.

### 5. Centralize categorical pinning and annual-load schema handling

**Status: completed.**

Canonical category levels are now reapplied through
`clean.pin_category_levels(frame, variables)` at both cleaning and encoding boundaries. The
defensive prediction-time calls remain because parquet round trips and caller-created subsets
may lose pandas categorical metadata.

`clean.load_and_clean_year` now accepts an optional column set, a missing-column policy, and an
explicit `allow`/`drop`/`require` label policy. The plausibility loader delegates to it, retains
narrow historical reads, and explicitly drops the unreliable pre-2004 label.

Do **not** add an in-process `lru_cache` around `build_county_value_panel` without stronger
evidence. Its DataFrame result is mutable, its year arguments are not consistently hashable,
and the repeated calls generally occur in separate CLI processes where memoization would not
avoid work. Persistent source-level caches already belong in the underlying data loaders.

### 6. Remove the calibration metric pass-through wrapper

**Status: completed.**

The canonical computation is now `density_ratio.evaluation.evaluate_sample`.
`calibration.probability_metrics` and `density_ratio.evaluation.probability_metrics` remain only
as direct compatibility aliases. Diagnostic workflows build the common cell schema through the
shared metric-record helper, and mixture-adjusted evaluations reuse their already-computed
metrics.

### 7. Reassess the legacy fold compatibility layer

**Status: completed.**

All production and test setup now uses `density_ratio.folds.TemporalFold` and the canonical
constructors directly. The single-user project does not require an external compatibility
period, so `model_selection.ReverseFold` and `model_selection.reverse_folds()` have been removed.

### 8. Keep command-line scripts thin

**Status: completed.**

The complete legacy validation workflow now lives in `validate.run_validation_workflow`,
including sampling, fitting decisions, conditional out-of-time execution, and persistence.
`scripts/validate_classifier.py` is limited to argument parsing and presentation.

Package-level load/process/save entry points also own the legacy training-data, Random Forest
training, logistic training, classification, histogram-cell, and estimator-benchmark stages.
Their scripts now parse arguments, make one stage call, and report the result. Audit/report CLIs
already call one domain operation and only serialize its returned tables.

### 9. Treat adapters and legacy workflows as explicit policy decisions

The fitted-model adapters add approximately 100 lines and could eventually be replaced by
`model_id` properties directly on the fitted family classes. They currently provide a clear
pickle-compatibility boundary, however, so removing them is lower priority until artifact
migration is complete.

Likewise, the legacy Random Forest training/classification/validation workflow and the raw
logistic-selection workflow account for substantial code but remain documented Makefile paths.
Deleting them would create the largest reduction, but only if the project explicitly drops
their reproducibility and compatibility commitments. They should not be treated as accidental
dead code.

## Recommended sequence

1. **Completed:** use the canonical sample-metric pathway and metric-record schema, then apply separate
   forward and reverse aggregation rules. Consolidate identical summary builders at this
   boundary.
2. **Completed:** introduce and adopt common checkpoint/table utilities, including plausibility
   outputs.
3. **Completed:** centralize categorical pinning and annual schema-tolerant loading.
4. **Completed:** move estimator primitives into the three `density_ratio/families/` modules
   and consolidate artifact boilerplate at the same boundary.
5. **Completed:** introduce and adopt a common calibration-diagnostic cell evaluator.
6. **Completed:** migrate to canonical folds and remove the legacy fold shim.
7. **Completed:** move substantive logic out of older scripts.
8. Decide separately whether documented legacy workflows should be deprecated or removed.

After each phase, run the full synthetic suite and the existing bounded family-parity checks.
Do not combine estimator relocation with changes to specifications, folds, weighting,
thresholds, mixture estimation, or model-selection objectives.
