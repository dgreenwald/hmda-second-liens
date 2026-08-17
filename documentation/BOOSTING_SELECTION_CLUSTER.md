# Unrestricted boosting selection on Slurm

## Scope

This workflow reproduces the frozen four-feature histogram-gradient-boosting selection with
immutable density-ratio shards. It is the distributed counterpart to the selection and final
refit inside `make evaluate-gradient-boosting`; it does not change the candidate grid,
reverse-time folds, mixture adjustment, or selection objective.

The unrestricted model uses `log_lti`, `log_county_value_to_loan`, purchaser type, and loan
type. The first two variables are continuous and the categories use native categorical
handling. The expected frozen winner has 7 leaves, learning rate 0.05, 200 iterations, L2
regularization 10, and 1,000 observations per leaf. Finalization refuses to publish a decision
if the distributed result selects a different configuration.

All commands below run from the repository root on the cluster. Paths are repository-relative
defaults; when `HMDA_SECONDS_OUTPUT_DIR` is set, submit the path printed by each generator.

## Screen

Generate and inspect the one-job, six-candidate structure screen before submitting it:

```bash
make generate-boosting-screen
sbatch output/slurm/boosting/screen/density_ratio_jobs.slurm
```

After the job completes, validate its immutable shard and aggregate the nine target years:

```bash
make aggregate-boosting-screen
```

## Survivors

The survivor generator reads the complete screen summary, retains its two lowest-Brier
structures, and creates one job for each of the nine reverse folds:

```bash
make generate-boosting-survivors
sbatch output/slurm/boosting/survivors/density_ratio_jobs.slurm
```

After all nine jobs complete:

```bash
make aggregate-boosting-survivors
```

## Refinement

The refinement generator reads the survivor summary and creates the frozen four-candidate
iteration/L2 neighborhood in each reverse fold:

```bash
make generate-boosting-refinement
sbatch output/slurm/boosting/refinement/density_ratio_jobs.slurm
```

After all nine jobs complete, aggregate and verify the winner:

```bash
make aggregate-boosting-refinement
make finalize-boosting-selection
```

Finalization writes the existing compatibility filenames
`output/tables/boosting_challenger_{cells,horizons,summary,decision}.csv`. The compatibility
tables are derived from the validated shards; they are not independent checkpoints.

## Final 2004--2007 refit

Generate and inspect the single-job final-refit script:

```bash
make generate-finalize-boosting-slurm
```

Submit it when ready:

```bash
make submit-finalize-boosting
```

Equivalently, generate and submit in one command with
`make generate-finalize-boosting-slurm FINALIZE_BOOSTING_FLAGS=--submit`. The job writes
`output/model/boosting_challenger.pkl` and its required `.metadata.json` sidecar. It refuses to
overwrite an existing artifact.

## Retry behavior

Stage generators never submit arrays. Completed matching shards and fitted fold artifacts are
reused on resubmission. Missing, duplicate, conflicting, cross-specification, or incomplete
results are rejected during aggregation. The final-refit generator submits only when passed
`--submit` explicitly (including through the `submit-finalize-boosting` Make target).
