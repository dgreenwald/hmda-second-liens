# Repository Guidelines

## Project Overview
This repo is a standalone letter spinning off the HMDA second-lien classifier from the `frm`
house-price paper. The primary loan-level estimator is now a formal ridge logistic model rather
than the original Random Forest. It trains on years where `lien_status` is reliably reported,
uses reverse-time validation to study backward extrapolation, and ultimately imputes lien
status for 1990-2003. A known-source-prior mixture estimator is the leading method for annual
count shares and mixture-adjusted probabilities. Read `documentation/MIGRATION_PLAN.md`,
`documentation/recommendations_revised.md`, `documentation/MODEL_SELECTION_PROTOCOL.md`, and
`documentation/mixture_estimation.md` before changing estimator behavior or validation design.

## Project Structure & Module Organization
- `src/hmda_seconds/` — the package: `config.py` (paths, features, hyperparameters), `clean.py`
  and `county_values.py` (sample construction and Zillow-scaled FHFA county values),
  `logistic_features.py` and `model_selection.py` (formal logistic estimator and
  frozen reverse-time selection), `calibration.py` (raw-probability diagnostics), `mixture.py`
  (known-source-prior density-ratio shares), `mixture_calibration.py` (adjusted-probability
  diagnostics). `density_ratio/` owns the cross-family protocols, temporal folds, mixture
  evaluation, atomic artifact/metadata contract, immutable shards, local/cluster runners, and
  deterministic aggregation. Model-family modules retain feature construction, fitting, and
  compatibility-table/diagnostic translation; they must not rebuild orchestration loops.
- `scripts/` — thin argparse CLIs, one per pipeline stage, each wrapping one `src/hmda_seconds`
  entry point. These are the `Makefile` targets' bodies; keep logic in `src/`, not here.
- `documentation/` — methodology plans, frozen protocols, implementation findings, and the
  refactoring agenda. Keep `README.md` and `AGENTS.md` at the repository root; place other
  project-level Markdown documentation here unless it belongs to a specific subdirectory.
- `data/raw/` — gitignored; user-populated HMDA LAR downloads (see `data/README.md` for exact
  source URLs). `data/public/` — small vendored public inputs (FHFA county HPI) that *are*
  committed.
- `output/` — regenerable: fitted model, figures, tables. Do not hand-edit; regenerate via
  `make`. Fitted model parameters are required artifacts, not disposable temporaries: every
  estimation path must save them under `output/model/` with enough transformer metadata to
  reproduce predictions exactly.
- `paper/` — the letter itself (`letter.tex` + `refs.bib`).
- `tests/` — pytest, synthetic fixtures only (no real HMDA data required to run the suite).

## Build, Test, and Development Commands
- `pip install -e ".[dev]"` — editable install with test tooling.
- Depends on `dgreenwald-py-tools[datasets]` for the HMDA/FHFA/Zillow loaders.
- Core logistic workflow: `make selection-data`, `make select-logistic`,
  `make diagnose-logistic-calibration`.
- Cluster raw-logistic selection uses `make generate-logistic-selection-coarse`, aggregation,
  then `make generate-logistic-selection-refinement`; see
  `documentation/LOGISTIC_SELECTION_CLUSTER.md`. The generators never submit unless their CLI
  is passed `--submit`, and immutable completed shards are reused on resubmission.
- Mixture workflow: `make estimate-mixture-shares`, then
  `make diagnose-mixture-calibration`, then `make diagnose-threshold-subgroups`.
  Reverse/forward fold parameters are cached under `output/model/mixture_folds/` and must be
  reused rather than silently refitted.
- Every newly fitted production, tuning, validation, or diagnostic model must be saved through
  `density_ratio.artifacts`. Each pickle has a sibling `.metadata.json` sidecar containing its
  schema version, SHA-256 payload digest, model/configuration identity, source years, training
  counts, feature schema, weighting/prior convention, and software versions. Loaders accept
  metadata-free pickles are not supported; regenerate artifacts after incompatible refactors.
- Density-ratio grids run through `density_ratio.pipeline.run_grid`, which delegates each
  `(specification, source window)` job to the shared family/runner/shard path. Existing pipeline
  CSVs are compatibility views derived from shard results rather than independent checkpoints.
  Use `make generate-density-ratio-pilot` to generate—but never submit—the two-job Slurm pilot;
  use `make generate-first-order-logistic-grid` to generate—but never submit—the frozen
  63-job coordinate-search array after reviewing pilot resources. See
  `documentation/DENSITY_RATIO_CLUSTER.md`.
- Historical plausibility workflow: `make plausibility-checks` applies the frozen final raw and
  known-source-prior models to 1990--2016, writes annual aggregate shares and the 2003--2004
  continuity table, and renders the predicted/actual series. It checkpoints annual aggregates
  and never retains loan-level historical probabilities.
- Gradient-boosting challenger: `make evaluate-gradient-boosting` runs the frozen staged
  histogram-boosting density-ratio grid, reuses per-fold artifacts in
  `output/model/boosting_folds/`, and produces reverse and forward calibration diagnostics.
- Random Forest mixture robustness: `make evaluate-rf-mixture` applies equal source-year class
  priors and annual target mixture shares to the fixed 50-tree, depth-10 full-sample forest.
  Reuse saved fits in `output/model/rf_mixture_folds/`; do not reopen RF tuning automatically.
- `pytest tests/` — must pass without real data.
- Set `PY_TOOLS_DATA_DIR` (or copy `.env.example` to `.env`) to point `py_tools` dataset loaders
  at `data/raw/`.

## Coding Style & Naming Conventions
- PEP 8, 4-space indentation, `snake_case` for functions/variables, `CapWords` for classes.
- One pipeline stage per module in `src/hmda_seconds/`; `scripts/*.py` stay thin (arg parsing +
  one call into `src/`), matching the pattern in
  `~/research/frm/replication_package_proposal/code/diagnostics/hmda_lien_diagnostics.py`.
- No hardcoded absolute paths (`/data/hmda`, `/home/dan/Dropbox/...`) — all paths flow through
  `src/hmda_seconds/config.py` and `PY_TOOLS_DATA_DIR`/repo-relative `data/`, `output/`.

## Testing Guidelines
- `tests/test_<module>.py`, deterministic assertions on small fabricated DataFrames — exercise
  the sample-filter edge cases (e.g. `lien_status` restricted to `{1, 2}`, see
  `documentation/MIGRATION_PLAN.md`) and the validation/metric logic, not visual inspection of
  plots.
- Real-data estimator runs are manual verification steps, not part of the automated test suite.
- The frozen logistic specification is `spline_lti__purchaser_type` with ridge `C=0.1`:
  restricted-cubic-spline `log_lti`, linear `log_county_value_to_loan`, purchaser- and loan-type
  reference indicators, and linear interactions of both continuous variables with purchaser
  type. The richer full `log_lti` spline-basis-by-purchaser challenger was rejected. Do not
  retune or expand this specification during diagnostic stages.
- Reverse validation uses four-year later training windows against every earlier labeled year,
  yielding 45 cells and horizons 1-9. Preserve equal-within-horizon, then equal-across-horizon
  aggregation where declared by the protocol.
- Step 7 keeps the canonical 0.5 threshold frozen. Precision-recall curves are diagnostic only;
  do not select year- or subgroup-specific thresholds. Subgroup outputs use loan type,
  purchaser type, Census region, and target-year deciles of both continuous variables.
- Step 8 treats the 2003--2004 continuity check and broader public HELOC/second-mortgage series
  as plausibility evidence, not identification. None of the reviewed public external series
  directly measures subordinate-lien originations in the owner-occupied home-purchase sample.
- Step 9 boosting uses only the four primitive core predictors with native categorical handling
  and the same annual mixture-share adjustment as logistic. The selected 7-leaf model improves
  Brier in all 45 reverse cells but loses to logistic in all nine forward years; retain it as a
  finalist rather than silently replacing the logistic primary estimator.
- The mixture-adjusted fixed Random Forest improves on logistic in 35 of 45 reverse cells but
  loses to boosting in 42 of 45 and is worst in every forward year. It is the canonical RF
  robustness model; these results do not support another RF hyperparameter search.

## Commit & Pull Request Guidelines
- Short, imperative commit subjects, first line <= 72 chars.
- One logical change per commit.
- Note any data-path or hyperparameter assumptions in the PR description.

## Security & Configuration Tips
- Never commit files under `data/raw/` or anything derived from raw HMDA microdata beyond the
  aggregated/binned outputs this letter is designed to release publicly.
- Do not persist loan-level diagnostic probabilities or target characteristics. Persist fitted
  model objects, aggregated metrics, reliability bins, and approved binned outputs.
- Keep environment-specific settings (`PY_TOOLS_DATA_DIR`) in `.env`/shell env, not source files.
