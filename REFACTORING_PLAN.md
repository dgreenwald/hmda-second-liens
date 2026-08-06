# Density-Ratio Model Refactoring Plan

## Purpose

The logistic, gradient-boosting, and Random Forest estimators now implement the same
statistical workflow:

1. fit a density-ratio score on a labeled source window;
2. evaluate that score on common temporal folds;
3. estimate a target-year second-lien share from the unlabeled covariate distribution;
4. convert density ratios and the estimated share into adjusted probabilities; and
5. calculate a common set of validation and release outputs.

The current modules repeat parts of this workflow. This refactor will put the shared fold,
mixture-adjustment, evaluation, and artifact logic in one place while leaving feature
construction and estimator fitting with the model family that owns them. The immediate goal
is behavioral parity. It is not an opportunity to change the selected specification,
validation design, mixture-share estimator, or model-selection objective.

The refactor should also make the model search straightforward to distribute across a compute
cluster. A local run and a collection of cluster jobs must produce inputs to the same
deterministic aggregation code.

## Design principles

- Use composition and small interfaces, not a deep estimator inheritance hierarchy.
- Treat `log_ratio(frame)` as the common output of every fitted model.
- Keep family-specific feature transformations, fitting, and hyperparameter grids separate.
- Define temporal folds once and reuse the identical fold objects across model families.
- Centralize target-share estimation, probability adjustment, and metric calculation.
- Save every fitted model, including models fitted only for tuning or diagnostics.
- Write one immutable result shard per job; never have parallel jobs append to a shared file.
- Persist aggregate diagnostics and release outputs, not loan-level diagnostic probabilities.
- Keep command-line scripts thin and place all substantive logic under `src/hmda_seconds/`.

## Proposed package structure

```text
src/hmda_seconds/density_ratio/
    __init__.py
    protocols.py       # structural interfaces and common result types
    folds.py           # temporal-fold definitions and validation
    evaluation.py      # mixture adjustment and common metrics
    artifacts.py       # model metadata, naming, save/load helpers
    shards.py          # immutable job-result schema and aggregation
    runner.py          # fit/evaluate orchestration for one or many jobs
    families/
        __init__.py
        logistic.py
        gradient_boosting.py
        random_forest.py
```

Existing public entry points can initially delegate to this package. The old duplicated code
should be removed only after parity tests pass and all Makefile targets use the shared path.

## Interfaces

### Fitted model

The shared evaluator needs only a fitted object that can return a log density ratio. Define a
structural protocol:

```python
class FittedDensityRatioModel(Protocol):
    model_id: str
    train_years: tuple[int, ...]

    def log_ratio(self, frame: pd.DataFrame) -> np.ndarray:
        ...
```

Concrete logistic, boosting, and Random Forest fitted-model classes do not inherit from this
protocol. A class satisfies it by exposing the named attributes and method. This keeps model
artifacts independent of a shared superclass and allows lightweight adapters around existing
saved objects. Protocol checking is primarily static; deterministic contract tests will
provide runtime enforcement because the repository does not currently require mypy or
pyright.

The protocol must not prescribe estimator internals, feature matrices, coefficient access, or
serialization format. Those details differ legitimately across families.

### Model family

A separate family interface owns fitting:

```python
class DensityRatioFamily(Protocol):
    family_name: str

    def fit_many(
        self,
        training: pd.DataFrame,
        configurations: Sequence[ModelConfiguration],
        *,
        train_years: tuple[int, ...],
    ) -> Mapping[str, FittedDensityRatioModel]:
        ...
```

`fit_many` is preferable to a single-model-only interface because logistic candidates can
reuse a transformed design matrix and warm starts. Tree-family implementations may simply
loop over configurations. The orchestration layer should make no assumptions about how that
reuse occurs.

An abstract base class is not needed for either interface at the outset. Introduce a shared
base or mixin only if the concrete implementations later acquire substantial, genuinely
identical behavior or state that cannot be expressed cleanly as free functions.

## Shared data objects

Use frozen dataclasses or similarly explicit records for the boundaries between components:

- `TemporalFold`: fold ID, source years, target years, direction, and horizons.
- `ModelConfiguration`: family, specification ID, hyperparameters, and random seed.
- `ModelArtifactMetadata`: model ID, configuration, training years, code/schema version,
  training counts, feature names, and artifact path.
- `EvaluationResult`: model ID, fold/cell identifiers, target year, horizon, observation count,
  estimated mixture share, Brier score, and the agreed secondary diagnostics.
- `JobSpecification`: stage, family, specification, source window, candidate set, input paths,
  and output paths.

These records should have stable serialization and explicit schema versions. Avoid passing
loosely structured dictionaries across module boundaries.

## Common evaluation path

For each fitted model and target-year frame, the shared evaluator will:

1. call `log_ratio(frame)` exactly once;
2. estimate the target mixture share using the frozen mixture-share routine;
3. construct adjusted probabilities from the log ratio and target share;
4. compute raw Brier score and the established diagnostic metrics;
5. compute the hard classification using the agreed threshold when requested;
6. compute the count share as the unweighted mean probability; and
7. return aggregate results without saving loan-level probabilities.

The evaluator should accept arrays directly in its low-level functions so its numerical logic
can be tested without fitting a model. It must validate finite log ratios, probability bounds,
row counts, class coding, and the target-share solution. Numerical clipping must be defined in
one place and applied identically to every model family.

The final release path should derive all three agreed outputs from the same probability vector:
probabilities, thresholded classifications, and annual count shares. Validation code may keep
the probability vector in memory but must not persist it at loan level.

## Fold ownership

Move reverse and forward temporal-fold construction into `folds.py`. A fold definition must be
data-independent and identified by a stable ID. All families receive the same fold sequence;
no estimator module may reconstruct folds privately.

The reverse design remains the 45-cell design with four-year source windows and backward
horizons one through nine. Aggregation continues to average within horizon and then equally
across horizons. Forward diagnostics use their separately defined common folds and must not be
mixed into the selection objective.

Tests should assert the exact source years, target years, horizons, number of cells, ordering,
and absence of overlap. This turns fold comparability from a convention into an enforced
property.

## Artifact policy

Every successful fit must be saved before evaluation begins. Each artifact should include or
sit beside enough metadata to reconstruct:

- model family and configuration;
- feature specification and fitted feature schema;
- source years and training sample counts;
- weighting and source-prior convention;
- random seed and relevant software versions; and
- the artifact/result schema version.

Use deterministic paths derived from model ID and fold ID. Never overwrite one candidate with
another, and write through a temporary file followed by an atomic rename. Aggregators should
reject duplicate logical keys with conflicting contents rather than silently choosing one.

Existing fitted artifacts should remain readable during the migration. Where practical, wrap
them in adapters instead of refitting solely to change class layout. New fits use the new
metadata contract.

## Cluster execution and result shards

The natural cluster job is one `(model specification, source window)` pair. The job should fit
all penalty or hyperparameter values that benefit from shared preprocessing, save every fitted
model, evaluate all target years belonging to that source window, and write one immutable
result shard.

The worker must accept ordinary command-line arguments, with environment variables as a thin
cluster convenience. Proposed variables are:

```text
HMDA_DENSITY_RATIO_STAGE
HMDA_DENSITY_RATIO_FAMILY
HMDA_DENSITY_RATIO_SPEC
HMDA_DENSITY_RATIO_TRAIN_START
HMDA_DENSITY_RATIO_OUTPUT_ROOT
```

Environment parsing belongs in the thin script, not in the fitting or evaluation library.
Explicit command-line arguments should take precedence where both are provided. The worker
must be idempotent: it may recognize an already complete matching shard, but it must never
partially append to one.

For the mixture-native logistic reselection, cluster availability removes the need for the
latest-window four-survivor shortcut. Subject to a small resource pilot, run all 13 feature
specifications over all nine source windows and the complete coarse penalty grid. Refinement
jobs can then be generated from the aggregate coarse results. This computational change should
be recorded in an updated selection protocol before results are inspected.

The aggregation command reads completed shards, verifies that the planned job matrix is
complete and unique, computes the frozen equal-horizon objective, and writes the consolidated
tables. It performs no fitting and obtains the same answer whether shards were generated
locally, sequentially, or through Slurm.

## Migration agenda

### 1. Freeze current behavior

- Record golden synthetic outputs for the logistic, gradient-boosting, and Random Forest
  mixture paths.
- Cover target-share estimation, adjusted probabilities, cell-level Brier scores, and
  equal-horizon aggregation.
- Inventory current artifact names and document which existing files must remain readable.
- Do not resume the large logistic selection run during this step.

Status: completed before structural changes. The numerical characterization is in
`tests/test_density_ratio_characterization.py`, and the compatibility boundary is recorded in
`MODEL_ARTIFACT_INVENTORY.md`.

### 2. Extract protocols and common value types

- Add `protocols.py` and the frozen records described above.
- Write contract tests using a minimal fake fitted model.
- Add adapters for existing fitted logistic, boosting, and Random Forest objects without
  changing their numerical behavior.

Status: completed. The structural interfaces and frozen, schema-versioned records are in
`src/hmda_seconds/density_ratio/protocols.py`. Compatibility adapters delegate to the existing
fitted classes without moving or modifying their pickle-visible definitions.

### 3. Centralize fold construction

- Add `folds.py` with the reverse and forward designs.
- Replace private fold loops with the common definitions.
- Verify exactly 45 reverse cells and parity with existing result identifiers.

Status: completed. `src/hmda_seconds/density_ratio/folds.py` now owns reverse and forward fold
construction. Estimator and diagnostic modules consume these shared objects, while the former
`model_selection.ReverseFold` API remains as a compatibility shim.

### 4. Centralize mixture evaluation

- Move target-share adjustment and common metrics behind one evaluator.
- Make each existing model path call the shared evaluator.
- Compare new and old aggregate outputs within tight numerical tolerances before deleting any
  old implementation.

### 5. Standardize model artifacts

- Add deterministic model IDs, metadata, atomic writes, and load-time validation.
- Ensure every tuning, validation, diagnostic, and final refit is persisted.
- Retain backward-compatible loaders or adapters for current artifacts.

### 6. Introduce family adapters

- Implement logistic, gradient-boosting, and Random Forest family objects.
- Preserve logistic design-matrix reuse and warm-start opportunities through `fit_many`.
- Keep feature construction and estimator-specific hyperparameters inside each family.
- Run family contract and numerical parity tests.

### 7. Add the common runner and immutable shards

- Implement a single-fold worker and a local multi-job runner.
- Define the shard schema, completeness checks, and deterministic aggregator.
- Test interrupted jobs, duplicate shards, missing cells, and aggregation independent of shard
  order.

### 8. Add cluster-facing entry points

- Add a thin CLI that maps arguments or environment variables into `JobSpecification`.
- Add a Slurm script generator modeled on the user's existing transition-script pattern.
- Pilot one simple and one spline-heavy logistic specification to measure memory, runtime, and
  shared-input contention before submitting the complete grid.

### 9. Migrate pipeline commands and remove duplication

- Point Makefile targets and diagnostic scripts at the common runner/evaluator.
- Confirm that final release outputs and challenger comparisons are unchanged.
- Remove superseded loops and helpers only after all parity checks pass.
- Update `AGENTS.md`, the relevant protocols, and the implementation agenda to describe the
  final architecture.

### 10. Resume model selection

- Amend and freeze the mixture-logistic selection protocol for the complete cluster grid.
- Run the coarse grid, aggregate and inspect completeness, define refinement jobs, and run the
  refinements.
- Select using raw mixture-adjusted Brier with the already agreed horizon weighting.
- Save the selected full-sample refit and all candidate/fold artifacts.

## Testing and acceptance criteria

The refactor is complete when:

- all current unit tests pass without real HMDA data;
- new tests prove that each fitted-model adapter satisfies the runtime contract;
- all families consume identical fold definitions;
- the shared evaluator reproduces existing mixture shares, adjusted probabilities, Brier
  scores, and horizon aggregates within documented tolerances;
- every fit produces a durable model artifact and metadata record;
- the aggregator detects missing, duplicate, malformed, and incompatible shards;
- sequential local execution and shuffled shard aggregation produce identical results;
- no loan-level diagnostic probabilities are written;
- scripts remain thin and contain no estimator or metric logic; and
- the old duplicate implementations are removed only after parity is demonstrated.

## Decisions deferred until implementation evidence exists

- Whether static type checking should become a required development command. Protocols remain
  useful documentation without it, but mypy or pyright would enforce them earlier.
- Whether multiple fitted families should share a serialization container. Start with common
  metadata and family-specific payloads to minimize pickle compatibility risk.
- Whether a base class or mixin is justified by common implementation discovered during the
  migration. Do not introduce one merely for nominal uniformity.
- Final cluster memory and concurrency limits. Set these from the two-job resource pilot rather
  than guessing from local runs.
