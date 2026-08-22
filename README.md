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
submits jobs automatically. Portable model variants using HMDA-only predictors follow the
[HMDA-only model-selection protocol](documentation/HMDA_ONLY_MODEL_SELECTION_PROTOCOL.md).

## Sync results from the cluster

Run the sync from the repository root on the local machine. Configure the cluster account,
repository path, and local destination in the local `.env` file (the host defaults to NYU's
Torch Data Transfer Node):

```bash
HMDA_SECONDS_CLUSTER_USER=dlg340
HMDA_SECONDS_CLUSTER_REPO=/home/dlg340/research/hmda-second-liens
HMDA_SECONDS_OUTPUT_DIR=/absolute/path/to/local/hmda-second-liens-output
```

`HMDA_SECONDS_OUTPUT_DIR` is the local output root into which the synchronized `model/`,
selection-result, `slurm/`, and `tables/` paths are placed. If it is unset, the destination is
the repository's `output/` directory. For a one-off destination, pass the equivalent CLI option
through the Make variable, for example
`make sync-selection-results SYNC_CLUSTER_FLAGS="--output-dir /absolute/path/to/output"`.

Preview the transfer without using the network, then apply it:

```bash
make sync-selection-results
make sync-selection-results SYNC_CLUSTER_FLAGS=--apply
```

The applied sync uses one authenticated `rsync` session to retrieve the unrestricted and
HMDA-only logistic and boosting selection results. It transfers fitted models, immutable
shards, Slurm manifests and logs, aggregate tables, and final models, but not raw HMDA data,
selection-data Parquets, or loan-level output. Before replacing local outputs, it validates
artifact digests, reaggregates the shards, and verifies the model-selection decisions. Existing
conflicting files are preserved under `output/sync_backups/<UTC timestamp>/`; a failed transfer
or validation leaves the current local results untouched.

For retrying a partial transfer with its retained staging directory and for NYU transfer
guidance, see [Sync model-selection results back to a local machine](documentation/BOOSTING_SELECTION_CLUSTER.md#sync-model-selection-results-back-to-a-local-machine).

The original full-release Random Forest train/classify/validate workflow has been removed after
the replacement estimators were validated through the common temporal-fold and metric
protocols. Current commands are documented in [AGENTS.md](AGENTS.md).
