# Codebase streamlining review

## Scope

This review examines the current Python codebase for dead code, duplicated helpers, overlapping
responsibilities, and compatibility layers that may now be removable. The initial findings are
retained below, with implementation status updated as the streamlining steps are completed.

The review was comprehensively refreshed in August 2026 after all nine original streamlining
findings were completed. Items 10–14 reflect a fresh audit of the codebase as it currently
stands.

The repository currently contains approximately:

- 8,587 lines under `src/hmda_seconds/`;
- 762 lines under `scripts/`; and
- 3,010 lines under `tests/`.

Large modules are not automatically redundant. In particular, splitting a large module can
improve navigation without reducing total code. The recommendations below prioritize actual
duplication and unnecessary dependency layers.

## Baseline status

All nine original streamlining findings are complete. Ruff, Python compilation, and all 139
synthetic tests pass. Streamlining work should preserve this baseline and continue to use the
bounded real-data family-parity checks where estimator code is moved.

## Findings

### 1. Make the density-ratio family modules the actual implementation owners

**Status: completed.**

The three family modules now own their fitted-model records, feature construction, fitting
primitives, deterministic paths, and artifact save/load functions. Top-level orchestration
modules import those family-owned objects directly.

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

Every remaining script is limited to argument parsing, one package-level pipeline call, and
brief presentation. The legacy training, classification, validation, histogram-figure, and
benchmark scripts were removed with their superseded workflows in Step 8.

### 9. Treat adapters and legacy workflows as explicit policy decisions

**Status: completed.**

The fitted family classes now expose `model_id` directly and satisfy the shared protocol without
adapters. The adapter module and metadata-free artifact compatibility path have been removed;
local fitted artifacts must be regenerated after this hard switch.

The legacy full-release Random Forest training, classification, validation, figure, and
full-sample benchmark workflows have also been removed. The formal frozen logistic selection
and diagnostics remain because they define the primary estimator rather than a compatibility
workflow.

---

## Remaining opportunities (fresh audit, August 2026)

The following items were identified by inspecting the live codebase after all prior steps were
completed. They are ordered from lowest to highest implementation risk.

### 10. Remove the migration re-export in `mixture_logistic_selection.py`

**Status: completed.**

The orchestration module previously re-exported the family-owned `fit_candidate_path`. Only
`tests/test_mixture_logistic_selection.py` used that migration alias; production code did not.
Those tests now import the family primitive directly, and the orchestration-module import and
`__all__` entry have been removed, leaving the family module as the single owner.

**Risk:** Very low.

### 11. Route the sample-audit FHFA load through the canonical loader

**Status: completed.**

`audit.run_sample_audit` directly calls `fhfa.load("county", ...)`, formats `year` and `fips`,
and uses the result — duplicating `clean.load_fhfa_county_hpi`, which performs the same three
steps. The only difference is that `audit.py` hard-codes `config.FHFA_DATA_DIR` while
`clean.load_fhfa_county_hpi` accepts an optional `data_dir` argument.

`audit.run_sample_audit` now calls `clean.load_fhfa_county_hpi()`. The duplicate direct dataset
load and year/FIPS normalization have been removed from `audit.py`.

**Risk:** Very low.

### 12. Centralize class-aware probability and numerical primitives

**Status: completed.**

Second-lien probability extraction still repeats the scikit-learn class lookup in selected
logistic, density-ratio logistic, boosting, Random Forest, plausibility, threshold, and mixture
paths. `mixture._finite_vector` and `density_ratio.evaluation._finite_vector` also perform the
same validation, while `mixture._log_mean_exp` and
`density_ratio.families.logistic._log_mean_exp` duplicate the same stable calculation.

The neutral `density_ratio/numerical.py` now owns finite-vector validation, stable log-mean-exp,
and class-aware probability extraction. Model-facing prediction methods remain the public
interface and delegate to these primitives. Direct tests cover reversed class order, missing
classes, probability-column alignment, vector validation, and numerical stability. Keeping the
module independent also avoids a circular dependency between `mixture` and `evaluation`.

**Risk:** Very low.

### 13. Deduplicate pairwise estimator-comparison joins

**Status: completed.**

Three functions across three modules perform the same structural operation: select and rename
Brier columns, merge one-to-one on `[train_start, validation_year, horizon]`, and compute
difference columns:

| Module | Function |
|--------|----------|
| `gradient_boosting.py` | `compare_with_logistic` |
| `random_forest_mixture.py` | `estimator_comparison` |
| `mixture_logistic_selection.py` | `compare_estimators` |

The number of comparison models and their input column names differ, but the canonical cell
merge and subtraction logic are shared.

`density_ratio.evaluation.merge_cell_metrics` now owns canonical cell selection, explicit
metric renaming, one-to-one joins, and named differences. The three comparison workflows retain
their CSV loading and output-name declarations. Geographic and candidate-specific comparisons
remain separate. Direct tests preserve column order, difference values, schema alignment, and
duplicate-cell rejection.

**Risk:** Low. Output schemas and one-to-one validation must remain unchanged.

### 14. Consolidate the two-stage reverse-horizon aggregation protocol

**Status: completed.**

Four functions independently implement equal-within-horizon then equal-across-horizons
aggregation:

| Module | Function |
|--------|----------|
| `gradient_boosting.py` | `aggregate_brier` |
| `mixture_logistic_selection.py` | `aggregate_candidates` |
| `threshold_diagnostics.py` | `aggregate_threshold_metrics` |
| `model_selection.py` | `aggregate_brier_cells` |

`calibration.aggregate_reverse_metrics` already implements the same scientific weighting rule.
`mixture.aggregate_share_errors` also uses two stages but first constructs a distinct long-form
error table, so it need not be forced through the initial migration.

The family-neutral `density_ratio.aggregation.two_stage_horizon_means` now owns within-horizon
means, equal-across-horizon means, cell counts, and candidate horizon-completeness validation.
It accepts multiple metric columns but deliberately does not own output renaming, extrema, loan
counts, sorting, or presentation. Logistic selection, mixture-logistic reselection, boosting,
and threshold diagnostics now use the primitive while retaining those family-specific pieces.
Tests cover unequal cell counts, inconsistent candidate coverage, an explicit external horizon
plan, numerical parity, and exact caller schemas.

This duplication is more than cosmetic: it allows the scientific weighting protocol to drift
between estimators.

**Risk:** Medium. Requires careful parameter design for the variable group-key columns and
metric names. Execute with parity tests.

---

## Intentionally deferred or rejected items

The following items were evaluated and are recorded here so they are not rediscovered in future
audits.

### A. `lru_cache` on `build_county_value_panel`

Rejected. The result is a mutable DataFrame, the year-range argument is not consistently
hashable, and calls generally occur in separate CLI processes where memoization would not
avoid work. See item 5 above.

### B. Panel rendering function consolidation

Deferred. The reliability and precision-recall diagrams share subplot setup but differ in axis
scales, reference lines, curve multiplicity, and legends. A callback-driven generic renderer
would replace modest repetition with a more indirect interface. Reconsider only if a third
materially similar panel figure is added.

### C. Reverse-fold loop orchestration into a shared runner

Completed in part by `density_ratio/pipeline.py` `run_grid` for the mixture-adjusted families.
The raw-logistic calibration diagnostic (`calibration._run_reverse_diagnostics`) retains its
own loop because it uses the raw logistic model path and checkpoint schema rather than the
density-ratio family protocol. This is an intentional boundary.

### D. Inline checkpoint-completion wrappers

Rejected. The remaining private wrappers give workflow-specific names to completion criteria
and often appear at several call sites. Some also check several artifact tables, required
non-null outputs, complete estimator sets, or floating-point candidate identity. Inlining the
simple-looking cases would make call sites longer and less descriptive without removing a
second implementation of the underlying lookup; `checkpoints.rows_present` already owns that
operation.

### E. Rename `clean.build_county_value_panel`

Deferred. The `clean` function is a loading-and-construction façade over the lower-level
`county_values.build_county_value_panel`, and fully qualified call sites distinguish them. A
rename would create churn without reducing code or clarifying estimator behavior. If the name
causes demonstrated confusion later, prefer `build_scaled_county_value_panel` or
`load_and_build_county_value_panel` over the ambiguous `build_full_county_value_panel`.

### F. `model_selection._cell_complete` compound condition

Retained. This function checks `specification`, `regularization_c` (with `np.isclose`),
`train_start`, and `validation_year` simultaneously. Collapsing it into `checkpoints.rows_present`
would lose the floating-point-safe C comparison.

---

## Recommended sequence for remaining work

1. **Completed: Item 10** (remove migration re-export) — tests now use the family owner and the
   alias has been removed.
2. **Completed: Item 11** (audit FHFA load) — the audit now uses the canonical loader.
3. **Completed: Item 12** (probability and numerical primitives) — shared helpers and direct
   tests are in place, and all callers have been migrated.
4. **Completed: Item 13** (pairwise comparison helper) — the three canonical comparisons use
   the shared pure-DataFrame primitive with schema tests.
5. **Completed: Item 14** (two-stage horizon aggregation) — the narrow shared primitive is in
   place with completeness, numerical-parity, and schema tests.

After each step, run `pytest tests/` to confirm the synthetic suite continues to pass.
Do not combine any of these with changes to specifications, folds, weighting, thresholds,
mixture estimation, or model-selection objectives.
