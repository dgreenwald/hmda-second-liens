# hmda-second-liens

A validated classifier for imputing lien status (first vs. second lien) in HMDA mortgage
records, 1990-2016, for the years HMDA does not reliably report it (pre-2004). The frozen ridge
logistic model is the primary loan-level estimator; known-source-prior mixture adjustment is
used for annual shares, with boosting and Random Forest density-ratio robustness models.

This repo is being ported and extended from an exploratory script into a standalone,
citable letter. See the [migration plan](documentation/MIGRATION_PLAN.md) for the full scope,
rationale, and porting plan, and [AGENTS.md](AGENTS.md) for repository conventions. Methodology
plans, frozen protocols, implementation findings, and refactoring documentation live under
[`documentation/`](documentation/).

Cluster execution uses immutable result shards and a generated Slurm array; see the
[density-ratio cluster workflow](documentation/DENSITY_RATIO_CLUSTER.md). Generation never
submits jobs automatically.

The original full-release Random Forest train/classify/validate workflow has been removed after
the replacement estimators were validated through the common temporal-fold and metric
protocols. Current commands are documented in [AGENTS.md](AGENTS.md).
