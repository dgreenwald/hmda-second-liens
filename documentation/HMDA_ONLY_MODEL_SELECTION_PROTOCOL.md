# HMDA-only model-selection protocol

## Purpose

This protocol selects restricted logistic and histogram-gradient-boosting challengers whose
predictors are available in HMDA LAR data alone. The restricted models omit
`log_county_value_to_loan`; they are robustness and portability specifications and do not
replace the frozen primary logistic estimator or the existing four-feature boosting finalist.

The searches reuse the existing narrow selection Parquets and read only `year`, lien status,
`log_lti`, purchaser type, loan type, and state. This holds the estimation sample fixed relative
to the full-feature search and avoids creating a redundant annual cache. The existing cache
was originally constructed with the county-value merge, so the exact model-selection sample
still reflects county-value availability even though the restricted estimator does not consume
that variable. A fresh production fit can construct its three predictors from HMDA alone, but
would need an explicit sample-comparability check if it does not reproduce that merge filter.

## Restricted logistic grid

The primitive inputs are `log_lti`, purchaser type, and loan type. The grid crosses:

- linear or restricted-cubic-spline `log_lti`;
- no interactions, loan-type interactions, purchaser-type interactions, or both;
- ridge `C = 0.0001, 0.01, 1, 100` in the coarse stage;
- the two adjacent decades around each coarse winner in the refinement stage.

This gives eight specifications and 72 array jobs per stage: one reusable ridge path for each
specification and each of the nine reverse folds. Selection uses raw Brier score averaged
within horizon and then equally across the nine horizons, matching the primary raw-logistic
protocol. Restricted names begin with `hmda_only__`, and all data, artifacts, shards, and
tables have separate paths from the primary search.

## Restricted gradient-boosting grid

Boosting uses `log_lti` as continuous and purchaser type and loan type as native categorical
features. It preserves the existing staged, mixture-adjusted protocol:

1. Screen the six combinations of 7, 15, or 31 leaves and learning rate 0.05 or 0.1 on the
   2013--2016 source fold against all nine earlier labeled years.
2. Retain the two lowest-Brier structures and evaluate them across all nine reverse folds.
3. Around the better survivor, evaluate the four frozen one-dimensional iteration and L2
   refinements across all nine folds.
4. Select from the two survivors and four refinements using mixture-adjusted Brier score,
   averaged within horizon and then equally across horizons.

The three stages contain 1, 9, and 9 array jobs. A fold's candidate set stays in one job so the
immutable density-ratio shard has one unambiguous logical identity. Every fitted candidate is
saved with artifact metadata before its result shard is published.

## Execution

The paths below show the repository-relative defaults. When
`HMDA_SECONDS_OUTPUT_DIR` is set, the generators print the corresponding configured Slurm
script paths; submit those printed paths instead.

For logistic, generate, inspect, and submit the coarse array; aggregate it before generating
the refinement array:

```bash
make generate-hmda-only-logistic-coarse
sbatch output/slurm/hmda_only_logistic_selection/coarse/logistic_selection_jobs.slurm
make aggregate-hmda-only-logistic-coarse
make generate-hmda-only-logistic-refinement
sbatch output/slurm/hmda_only_logistic_selection/refinement/logistic_selection_jobs.slurm
make aggregate-hmda-only-logistic-refinement
```

After reviewing the decision table, generate the single-job 2004--2007 refit, adding the
generator's `--submit` flag when ready:

```bash
make generate-finalize-hmda-only-logistic-slurm
python scripts/generate_finalize_logistic_slurm.py --feature-set hmda_only --submit
```

Equivalently, pass the flag through the Make target with
`make generate-finalize-hmda-only-logistic-slurm FINALIZE_LOGISTIC_FLAGS=--submit`. The fitted
artifact is written to `$HMDA_SECONDS_OUTPUT_DIR/model/logistic_hmda_only_selected.pkl` with
its required metadata sidecar.

For boosting, complete and aggregate each stage before generating the next:

```bash
make generate-hmda-only-boosting-screen
sbatch output/slurm/hmda_only_boosting/screen/density_ratio_jobs.slurm
make aggregate-hmda-only-boosting-screen

make generate-hmda-only-boosting-survivors
sbatch output/slurm/hmda_only_boosting/survivors/density_ratio_jobs.slurm
make aggregate-hmda-only-boosting-survivors

make generate-hmda-only-boosting-refinement
sbatch output/slurm/hmda_only_boosting/refinement/density_ratio_jobs.slurm
make aggregate-hmda-only-boosting-refinement
make finalize-hmda-only-boosting-selection
```

After reviewing the decision table, generate the single-job 2004--2007 refit and submit it
when ready:

```bash
make generate-finalize-hmda-only-boosting-slurm
make submit-finalize-hmda-only-boosting
```

Equivalently, pass `--submit` through the generator target with
`make generate-finalize-hmda-only-boosting-slurm FINALIZE_HMDA_ONLY_BOOSTING_FLAGS=--submit`.
The fitted artifact and required metadata sidecar are written under
`$HMDA_SECONDS_OUTPUT_DIR/model/boosting_hmda_only_challenger.pkl`. Stage generation never
submits boosting arrays automatically. Aggregation refuses to produce a decision from missing
or duplicate candidates, folds, horizons, or shards.

Before submitting an array, its first manifest entry can be exercised directly on a compute
node:

```bash
python scripts/run_logistic_selection_job.py \
    --manifest output/slurm/hmda_only_logistic_selection/coarse/logistic_selection_jobs.json \
    --job-index 0

python scripts/run_density_ratio_job.py \
    --manifest output/slurm/hmda_only_boosting/screen/density_ratio_jobs.json \
    --job-index 0
```

These are real single jobs, not reduced synthetic fits; their artifacts and immutable shards
are reused when the corresponding arrays are later submitted.
