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

## Findings

### 1. Resolve the current merge conflicts first

The working tree contains unresolved add/add conflicts in:

- `src/hmda_seconds/density_ratio/cluster.py`;
- `scripts/run_density_ratio_job.py`; and
- `scripts/aggregate_density_ratio_shards.py`.

The files contain `<<<<<<<`, `=======`, and `>>>>>>>` markers. Git reports them as `AA`, and
Python parsing and Ruff fail. The test suite therefore cannot run in the current state. Resolve
these conflicts before beginning cleanup so subsequent test results provide meaningful safety
checks.

### 2. Make the density-ratio family modules the actual implementation owners

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

### 3. Consolidate common calibration-diagnostic cell generation

The boosting and Random Forest modules contain nearly identical diagnostic-cell functions:

- `gradient_boosting._diagnose_cell`;
- `random_forest_mixture._evaluate_cell`; and
- a more extensive variant in `mixture_calibration._run_design`.

Each implementation performs the same core sequence:

1. check whether aggregate outputs already exist;
2. call the shared evaluator;
3. construct common probability metrics;
4. construct reliability bins;
5. attach fold and estimator metadata; and
6. replace the corresponding checkpoint rows.

Add a shared diagnostic helper that accepts a fitted density-ratio model, target frame, fold,
metadata, and optional family-specific diagnostic callback. The callback can add ratio-tail or
other specialized outputs without duplicating the common evaluation and reliability logic.

### 4. Consolidate CSV checkpoint utilities

There are six identical helpers that read a CSV if it exists and otherwise return an empty
DataFrame. At least five additional functions implement nearly identical row replacement:

- identify rows by `train_start` and `validation_year`, sometimes with an estimator or candidate
  key;
- remove matching existing rows;
- concatenate new rows; and
- write the entire CSV.

Representative copies appear in `mixture.py`, `gradient_boosting.py`,
`random_forest_mixture.py`, `calibration.py`, `mixture_calibration.py`, and
`threshold_diagnostics.py`.

Add a small shared module, such as `density_ratio/checkpoints.py`, with:

- `read_table(path)`;
- `rows_present(frame, keys)`;
- `replace_rows(existing, new, key_columns)`; and
- an atomic CSV writer.

This should remove roughly 100 lines while ensuring consistent interruption and replacement
behavior across diagnostic workflows.

### 5. Consolidate identical forward-summary builders

The `_simple_summary` implementation is identical in:

- `random_forest_mixture.py`;
- `mixture_logistic_selection.py`; and
- `mixture_calibration.py`.

Move it to `calibration.py` under a descriptive public name such as
`aggregate_forward_metrics`. This is a small deletion but has very low implementation risk.

### 6. Reassess the legacy fold compatibility layer

`model_selection.ReverseFold` and `model_selection.reverse_folds()` now delegate to
`density_ratio.folds`. Their remaining call sites are primarily tests and legacy type
annotations.

If these names are not part of a supported external API, migrate remaining callers to
`density_ratio.folds.TemporalFold` and remove the shim. If external users may import them,
retain explicit deprecated aliases until a planned compatibility break.

### 7. Keep command-line scripts thin

Several older scripts still contain workflow and persistence logic rather than only argument
parsing and one package call. `scripts/validate_classifier.py` is the clearest example: it owns
sampling, fitting decisions, output persistence, and conditional stage execution.

Move this orchestration into a package-level entry point and leave the script responsible for
argument parsing and presentation. This improves adherence to repository conventions, though it
mostly relocates code and should not be counted as a large reduction in total lines.

### 8. Treat adapters and legacy workflows as explicit policy decisions

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

1. Resolve all merge conflicts and restore a passing test suite.
2. Move estimator primitives into the three `density_ratio/families/` modules.
3. Introduce and adopt common checkpoint/table utilities.
4. Introduce and adopt a common calibration-diagnostic cell evaluator.
5. Consolidate the identical summary builders.
6. Remove the fold compatibility shim if it is not public API.
7. Move substantive logic out of older scripts.
8. Decide separately whether documented legacy workflows should be deprecated or removed.

After each phase, run the full synthetic suite and the existing bounded family-parity checks.
Do not combine estimator relocation with changes to specifications, folds, weighting,
thresholds, mixture estimation, or model-selection objectives.
