# Raw-logistic model selection on Slurm

The raw-probability search in `MODEL_SELECTION_PROTOCOL.md` runs in two immutable Slurm
arrays. Each task owns one feature specification and one four-year reverse-validation
window. The four coarse ridge values (or two refinement values) share the fitted feature
transformation and warm-started ridge path. The generator never submits unless `--submit`
is passed.

## Cluster defaults in `.env`

Copy `.env.example` to `.env` and set the machine-specific roots and Slurm defaults once:

```dotenv
HMDA_SECONDS_HMDA_DATA_DIR=/absolute/path/to/py-tools/hmda
HMDA_SECONDS_EXTERNAL_DIR=/absolute/path/to/hmda-second-liens/data
HMDA_SECONDS_OUTPUT_DIR=/absolute/path/to/hmda-second-liens/output
HMDA_SECONDS_SLURM_ACTIVATE=/absolute/path/to/venv/bin/activate
HMDA_SECONDS_SLURM_ACCOUNT=torch_pr_609_general
HMDA_SECONDS_SLURM_TIME=8:00:00
HMDA_SECONDS_SLURM_MEMORY=32G
HMDA_SECONDS_SLURM_MAX_CONCURRENT=8
```

The selection input then resolves to
`$HMDA_SECONDS_EXTERNAL_DIR/intermediate/logistic_selection`. CLI arguments override every
corresponding `.env` default.

## 1. Generate and run the coarse grid

With `.env` configured, generate the coarse manifest from the repository root:

```bash
make generate-logistic-selection-coarse
```

Inspect the 108-job manifest and Slurm file under
`output/slurm/logistic_selection/coarse/`, then submit manually or repeat the command with
`--submit`. Resubmitting the same array is safe: a task validates and reuses its complete
matching shard.

After all tasks finish, aggregate them:

```bash
python scripts/aggregate_logistic_selection_shards.py \
    --manifest output/slurm/logistic_selection/coarse/logistic_selection_jobs.json \
    --output-dir output/tables
```

Aggregation refuses missing, duplicate, malformed, or artifact-inconsistent results.

## 2. Generate and run the refinement grid

The next manifest is data-dependent but follows the frozen rule mechanically: each
specification receives the two decades adjacent to its own best coarse `C`.

```bash
make generate-logistic-selection-refinement
```

Inspect and submit `output/slurm/logistic_selection/refinement/logistic_selection_jobs.slurm`.
Then combine the coarse and refinement cells and write the final decision:

```bash
python scripts/aggregate_logistic_selection_shards.py \
    --manifest output/slurm/logistic_selection/refinement/logistic_selection_jobs.json \
    --coarse-cells output/tables/logistic_selection_core_coarse_cells.csv \
    --output-dir output/tables
```

The summary averages Brier scores equally within horizon and then equally across horizons.

## 3. Refit the selected model

Generate and submit the single-job batch refit with:

```bash
make submit-finalize-logistic
```

This writes the submitted script and its logs under
`output/slurm/logistic_selection/final/`. The account, time, memory, and activation script use
the same `.env` settings as the grid arrays. To inspect the script without submitting it, run
`make generate-finalize-logistic-slurm` instead.

For a local or already-allocated compute-node run, use:

```bash
make finalize-logistic-selection
```

This fits the declared winner on 2004--2007 and writes the model and metadata sidecar under
`output/model/`. Pass `--overwrite` only when intentionally replacing an existing selected
artifact.

Shards are immutable. To rerun estimates after a code or protocol change, choose a new
`--output-root` (and preferably a new `--destination`) rather than deleting or overwriting
the earlier run.
