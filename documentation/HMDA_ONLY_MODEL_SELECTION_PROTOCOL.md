# HMDA-only model-selection protocol

## Purpose

This protocol selects restricted logistic and histogram-gradient-boosting challengers that can
be estimated from HMDA LAR data alone. The restricted models omit
`log_county_value_to_loan`, so they do not require FHFA or Zillow inputs. They are robustness
and portability specifications; they do not replace the frozen primary logistic estimator or
the existing four-feature boosting finalist.

The HMDA-only sample retains the project's loan-purpose, occupancy, action, lien-label,
income, loan-amount, loan-type, state, and county validity restrictions. It does not inner-join
the county-value panel. Consequently, results can differ because both the predictor set and
the eligible sample differ from the full model. Comparisons must report this distinction.

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

Prepare the independent HMDA-only annual files once:

```bash
make hmda-only-selection-data
```

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

Generation never submits boosting jobs automatically. Aggregation refuses to produce a
decision from missing or duplicate candidates, folds, horizons, or shards.

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
