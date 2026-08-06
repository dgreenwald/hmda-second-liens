# hmda-second-liens

A validated classifier for imputing lien status (first vs. second lien) in HMDA mortgage
records, 1990-2016, for the years HMDA does not reliably report it (pre-2004). A fair
full-sample benchmark makes logistic regression the provisional primary estimator and retains
the Random Forest as a substantive robustness model; see
[Benchmark findings](documentation/BENCHMARK_FINDINGS.md).

This repo is being ported and extended from an exploratory script into a standalone,
citable letter. See the [migration plan](documentation/MIGRATION_PLAN.md) for the full scope,
rationale, and porting plan, and [AGENTS.md](AGENTS.md) for repository conventions. Methodology
plans, frozen protocols, implementation findings, and refactoring documentation live under
[`documentation/`](documentation/).

Cluster execution uses immutable result shards and a generated Slurm array; see the
[density-ratio cluster workflow](documentation/DENSITY_RATIO_CLUSTER.md). Generation never
submits jobs automatically.

Status: scaffolding in progress. Usage instructions will be filled in once the pipeline runs
end to end (see the migration plan's "Execution order").

The estimator comparison is reproduced with `make benchmark-estimators`. This first rebuilds
the separate common benchmark extract, then fits both models and writes the `benchmark_*`
model and metric artifacts without replacing the historical RF outputs.
