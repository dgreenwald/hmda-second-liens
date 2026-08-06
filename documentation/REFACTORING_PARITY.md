# Density-ratio refactoring parity

## Scope

The Step 9 migration routes the logistic mixture-selection grid, histogram-gradient-boosting
grid, and reverse Random Forest mixture evaluation through the common density-ratio families,
runner, immutable shards, and deterministic aggregator. Existing CSV outputs remain derived
compatibility views so downstream decision, comparison, and plotting code retains its schema.

Reliability-bin diagnostics still evaluate probabilities in memory because shards deliberately
exclude loan-level predictions. Those diagnostics use the same common evaluator as the shard
worker and persist only aggregate bins and metrics.

## Automated parity

Synthetic tests fit each family through both its established family-specific primitive and the
new shared grid path. They compare target mixture shares, adjusted Brier scores, and adjusted
log loss. The full synthetic suite also covers fold identity, artifact reuse, shard
completeness, interrupted jobs, idempotent retries, duplicate/conflicting shards, shuffled
aggregation, and CLI/manifest behavior.

## Bounded real-data check

The migration was checked without restarting the large selection exercise. For each family,
one existing model trained on 2013--2016 was loaded from its validated artifact and evaluated
on target year 2012 through the new runner. The resulting aggregate cell was compared with the
corresponding pre-refactor checkpoint.

| Family | Maximum absolute difference across mixture share, Brier, and log loss |
|---|---:|
| Logistic | `6.8e-17` |
| Histogram gradient boosting | `6.8e-17` |
| Random Forest | `1.4e-13` |

The checks wrote immutable shards only under `/tmp`. They did not alter released tables, retain
loan-level probabilities, refit the saved models, or resume mixture-logistic selection.

Fit and prediction timing columns remain in compatibility tables but are operational
diagnostics rather than parity targets: cached fits have no new fit time, and shard evaluation
does not make nondeterministic wall-clock measurements part of the immutable statistical
result.
