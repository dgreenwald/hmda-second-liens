# hmda-second-liens

A validated classifier for imputing lien status (first vs. second lien) in HMDA mortgage
records, 1990-2016, for the years HMDA does not reliably report it (pre-2004). A fair
full-sample benchmark makes logistic regression the provisional primary estimator and retains
the Random Forest as a substantive robustness model; see `BENCHMARK_FINDINGS.md`.

This repo is being ported and extended from an exploratory script into a standalone,
citable letter. See `MIGRATION_PLAN.md` for the full scope, rationale, and porting plan, and
`AGENTS.md` for repository conventions.

Status: scaffolding in progress. Usage instructions will be filled in once the pipeline runs
end to end (see `MIGRATION_PLAN.md`, "Execution order").

The estimator comparison is reproduced with `make benchmark-estimators`. This first rebuilds
the separate common benchmark extract, then fits both models and writes the `benchmark_*`
model and metric artifacts without replacing the historical RF outputs.
