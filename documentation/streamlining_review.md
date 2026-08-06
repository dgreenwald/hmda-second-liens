# Codebase streamlining review

## Scope

This review examines the current Python codebase for dead code, duplicated helpers, overlapping
responsibilities, and compatibility layers that may now be removable. The initial findings are
retained below, with implementation status updated as the streamlining steps are completed.

The review was comprehensively refreshed in August 2026 after all nine original refactoring
steps were completed. Items 10–16 reflect a fresh audit of the codebase as it currently stands.

The repository currently contains approximately:

- 8,512 lines under `src/hmda_seconds/`;
- 762 lines under `scripts/`; and
- 2,738 lines under `tests/`.

Large modules are not automatically redundant. In particular, splitting a large module can
improve navigation without reducing total code. The recommendations below prioritize actual
duplication and unnecessary dependency layers.

## Baseline status

The Step 8 add/add conflicts identified during the initial audit have been resolved. Ruff,
Python compilation, and all 129 remaining synthetic tests pass. Streamlining work should preserve this
baseline and continue to use the bounded real-data family-parity checks where estimator code is
moved.

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

## Round 2 assessment

The suggestions in `streamlining_suggestions_gemini_v2.md` were checked against the post-Step-8
tree. Some file references describe code that had already changed, so the following decisions
are based on the live implementations rather than the suggested line numbers.

### 10. Centralize prediction and numerical primitives

**Status: recommended.**

Second-lien probability extraction still repeats the scikit-learn class lookup in selected
logistic, density-ratio logistic, boosting, Random Forest, plausibility, and threshold paths.
Use one small class-aware probability helper, while retaining model methods as the public
prediction interface. Also consolidate the duplicate finite-vector and stable log-mean-exp
implementations. These are genuine identical numerical operations and should have direct unit
tests for class order, shape, empty inputs, and non-finite values.

Place the numerical primitives in a neutral density-ratio utility rather than expanding
`evaluation.py` into a miscellaneous helper module.

### 11. Reuse the canonical FHFA loader in the sample audit

**Status: recommended.**

`audit.run_sample_audit` still repeats `clean.load_fhfa_county_hpi`, including year and FIPS
normalization. It should call the canonical loader. The proposed historical-parquet
consolidation, however, is already complete: plausibility delegates to
`clean.load_and_clean_year` with narrow columns and an explicit pre-2004 label policy.

### 12. Centralize the two-stage reverse-horizon aggregation protocol

**Status: recommended, with a narrow interface.**

The selection, mixture, boosting, and threshold modules independently encode equal weighting
within horizon followed by equal weighting across horizons. A shared primitive should own the
two grouping stages and completeness/count semantics, while callers continue to name and sort
their family-specific outputs. This is more than cosmetic deduplication: it prevents the
scientific weighting protocol from drifting between estimators.

### 13. Consolidate simple pairwise cell comparisons where schemas align

**Status: recommended in part.**

Boosting-versus-logistic and the straightforward challenger tables repeat one-to-one merges on
the canonical reverse-cell keys and metric subtraction. Add a small comparison primitive and
adopt it where both sides have one metric per canonical cell. Do not force
`geographic_incremental_brier` or multi-estimator comparison tables through it: their candidate
selection, join keys, and output schemas are materially different.

### 14. Keep semantic checkpoint completion checks local

**Status: not recommended.**

`checkpoints.rows_present` already centralizes the generic lookup. The remaining wrappers are
not six identical four-line functions: some require non-null estimator outputs, some require
several artifact tables, and others verify complete estimator sets. Their names document the
workflow's definition of a complete cell. Adding a second generic `cell_present` wrapper would
save little code and obscure these distinctions.

### 15. Keep the two panel renderers separate

**Status: not recommended.**

The reliability and precision-recall figures share subplot setup but differ in limits, log
scales, reference lines, curve multiplicity, and legends. A callback-driven generic renderer
would replace modest plotting repetition with a more indirect interface. Reconsider only if a
third materially similar panel figure is added.

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
8. **Completed:** remove adapters and the superseded legacy full-release workflows.
9. Centralize class-aware probability extraction and the duplicate numerical primitives.
10. Route the audit through the canonical FHFA loader.
11. Centralize the two-stage reverse-horizon aggregation contract and migrate each caller with
    parity tests.
12. Consolidate only the pairwise comparison joins that share the canonical cell schema.

After each phase, run the full synthetic suite and the existing bounded family-parity checks.
Do not combine estimator relocation with changes to specifications, folds, weighting,
thresholds, mixture estimation, or model-selection objectives.
