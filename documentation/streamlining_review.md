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
Python compilation, and all 161 synthetic tests pass. Streamlining work should preserve this
baseline and continue to use the bounded real-data family-parity checks where estimator code is
moved.

## Findings

### 1. Make the density-ratio family modules the actual implementation owners

The new family layer currently wraps implementations that remain in the old top-level modules:

- `density_ratio/families/logistic.py` imports `mixture_logistic_selection` to fit candidates;
- `density_ratio/families/gradient_boosting.py` imports `gradient_boosting`; and
- `density_ratio/families/random_forest.py` imports `random_forest_mixture`.

The legacy selection and challenger modules then call back into the family layer during grid
execution. This reverse dependency is the largest structural source of code growth and creates
avoidable circular-import pressure.

Move each family's fitted-model record, feature construction, fitting primitive, artifact
save/load functions, and deterministic path construction into its module under
`density_ratio/families/`. Leave the top-level modules responsible for grids, staged decisions,
compatibility tables, comparisons, and diagnostics. Preserve old import paths with small
re-exports only where pickle or public-API compatibility requires them.

As part of this move, centralize the remaining family-specific artifact boilerplate. The common
artifact module already owns hashing, atomic writes, metadata validation, and sidecars, but each
family still repeats configuration construction and the sequence of building metadata and
saving the payload. A narrowly scoped helper can remove that repetition while leaving feature
schema, weighting convention, and model identity explicit at each call site.

### 2. Finish consolidating common calibration-diagnostic cell generation

The boosting and Random Forest modules contain nearly identical diagnostic-cell functions:

- `gradient_boosting._diagnose_cell`;
- `random_forest_mixture._evaluate_cell`; and
- a more extensive variant in `mixture_calibration._run_design`.

The canonical sample evaluator and metric-record schema now live in
`density_ratio.evaluation`; forward and reverse workflows share those computations and differ
only in their aggregation rules. The remaining implementations still repeat this sequence:

1. check whether aggregate outputs already exist;
2. call the shared evaluator and metric-record builder;
3. construct reliability bins;
4. attach the same metadata to bins and family-specific diagnostics; and
5. replace the corresponding checkpoint rows.

Add a shared diagnostic helper that accepts a fitted density-ratio model, target frame, fold,
metadata, and optional family-specific diagnostic callback. The callback can add ratio-tail or
other specialized outputs without duplicating the common evaluation and reliability logic.

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

`model_selection.ReverseFold` and `model_selection.reverse_folds()` now delegate to
`density_ratio.folds`. Their remaining call sites are primarily tests and legacy type
annotations.

If these names are not part of a supported external API, migrate remaining callers to
`density_ratio.folds.TemporalFold` and remove the shim. If external users may import them,
retain explicit deprecated aliases until a planned compatibility break.

### 8. Keep command-line scripts thin

Several older scripts still contain workflow and persistence logic rather than only argument
parsing and one package call. `scripts/validate_classifier.py` is the clearest example: it owns
sampling, fitting decisions, output persistence, and conditional stage execution.

Move this orchestration into a package-level entry point and leave the script responsible for
argument parsing and presentation. This improves adherence to repository conventions, though it
mostly relocates code and should not be counted as a large reduction in total lines.

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
4. Move estimator primitives into the three `density_ratio/families/` modules and consolidate
   artifact boilerplate at the same boundary.
5. Introduce and adopt a common calibration-diagnostic cell evaluator.
6. Remove the fold compatibility shim if it is not public API.
7. Move substantive logic out of older scripts.
8. Decide separately whether documented legacy workflows should be deprecated or removed.

After each phase, run the full synthetic suite and the existing bounded family-parity checks.
Do not combine estimator relocation with changes to specifications, folds, weighting,
thresholds, mixture estimation, or model-selection objectives.
