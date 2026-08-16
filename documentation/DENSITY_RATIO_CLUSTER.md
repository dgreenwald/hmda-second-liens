# Density-ratio cluster jobs

The cluster workflow uses one immutable JSON manifest and one Slurm array rather than one
hand-maintained script per model/window pair. Array tasks share no writable result file: each
task saves its model artifacts and atomically publishes one result shard.

## Generate the resource pilot

Run this on the cluster from the repository root:

```bash
python scripts/generate_density_ratio_slurm.py
```

This writes `output/slurm/density_ratio_jobs.json` and
`output/slurm/density_ratio_jobs.slurm`. It does **not** submit anything. The default pilot
contains two tasks on the 2013--2016 source window:

- `linear__none`;
- `spline_lti__purchaser_type_spline_lti`.

Both tasks fit the complete coarse ridge grid, `C = 0.0001, 0.01, 1, 100`, and evaluate all
nine earlier labeled target years. The generated array uses the Slurm resources configured in
`.env`. Generator options can override repository, data, output, activation, time, memory, and
source-window defaults. Inspect elapsed time and peak memory through Slurm accounting, for
example with `sacct --format=JobIDRaw,State,Elapsed,TotalCPU,MaxRSS`.

Inspect the generated files before manually submitting:

```bash
sbatch output/slurm/density_ratio_jobs.slurm
```

That command is intentionally not run by any repository script.

## Retry and aggregate

The array indices are stable manifest positions. A failed task can be resubmitted by index;
an already completed matching shard is recognized without refitting. Conflicting shard content
is rejected rather than overwritten.

After every planned shard exists, aggregate with:

```bash
python scripts/aggregate_density_ratio_shards.py \
    --manifest output/slurm/density_ratio_jobs.json \
    --output-dir output/tables/density_ratio_pilot
```

The command refuses incomplete, duplicate, malformed, or incompatible results and writes the
cell, horizon, and equal-horizon summary tables atomically.

## Direct worker invocation

`scripts/run_density_ratio_job.py` also accepts explicit job arguments. Explicit command-line
values take precedence over these optional cluster environment variables:

```text
HMDA_DENSITY_RATIO_STAGE
HMDA_DENSITY_RATIO_FAMILY
HMDA_DENSITY_RATIO_SPEC
HMDA_DENSITY_RATIO_TRAIN_START
HMDA_DENSITY_RATIO_OUTPUT_ROOT
```

Pass each candidate as a repeated JSON object, for example
`--configuration '{"C": 0.1}'`. Manifest mode is preferred for planned grids because it fixes
the complete job matrix before results are inspected.

## Pilot decision

Do not submit the first-order grid until the two pilot logs have been compared for peak memory,
wall time, and evidence of shared-input contention. Set final array concurrency and resource
requests from those measurements rather than extrapolating from the local machine.

## Generate the first-order logistic search

The frozen mixture-logistic protocol uses a coordinate-wise neighborhood rather than the full
global grid. After setting the shared paths and resources in `.env`, generate its 63-job
manifest with:

```bash
make generate-first-order-logistic-grid
```

This writes a separate manifest and Slurm script under `output/slurm/first_order/` and never
submits them. The default four-task concurrency cap is deliberately conservative; replace it
with the pilot-supported value before submission. Inspect both files, then submit manually:

```bash
sbatch output/slurm/first_order/density_ratio_jobs.slurm
```

When all 63 immutable shards exist, validate and aggregate the complete planned matrix:

```bash
python scripts/aggregate_density_ratio_shards.py \
    --manifest output/slurm/first_order/density_ratio_jobs.json \
    --output-dir output/tables/mixture_logistic_first_order
```

The summary table applies the frozen equal-within-horizon, then equal-across-horizon objective.
Apply the stopping rule in `MIXTURE_LOGISTIC_SELECTION_PROTOCOL.md` before proposing any further
jobs.
