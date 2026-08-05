# Repository Guidelines

## Project Overview
This repo is a standalone letter spinning off the HMDA second-lien classifier from the `frm`
house-price paper. The primary loan-level estimator is now a formal ridge logistic model rather
than the original Random Forest. It trains on years where `lien_status` is reliably reported,
uses reverse-time validation to study backward extrapolation, and ultimately imputes lien
status for 1990-2003. A known-source-prior mixture estimator is the leading method for annual
count shares and mixture-adjusted probabilities. Read `MIGRATION_PLAN.md`,
`recommendations_revised.md`, `MODEL_SELECTION_PROTOCOL.md`, and `mixture_estimation.md` before
changing estimator behavior or validation design.

## Project Structure & Module Organization
- `src/hmda_seconds/` — the package: `config.py` (paths, features, hyperparameters), `clean.py`
  and `county_values.py` (sample construction and Zillow-scaled FHFA county values),
  `logistic.py`, `logistic_features.py`, and `model_selection.py` (formal logistic estimator and
  frozen reverse-time selection), `calibration.py` (raw-probability diagnostics), `mixture.py`
  (known-source-prior density-ratio shares), `mixture_calibration.py` (adjusted-probability
  diagnostics), plus the RF compatibility, classification, validation, and figure modules.
- `scripts/` — thin argparse CLIs, one per pipeline stage, each wrapping one `src/hmda_seconds`
  entry point. These are the `Makefile` targets' bodies; keep logic in `src/`, not here.
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
- Depends on `dgreenwald-py-tools[ml,datasets]` for the RF wrapper and HMDA/FHFA loaders — reuse
  those rather than reimplementing (see `MIGRATION_PLAN.md`, "What already exists").
- Core logistic workflow: `make selection-data`, `make select-logistic`,
  `make diagnose-logistic-calibration`.
- Mixture workflow: `make estimate-mixture-shares`, then
  `make diagnose-mixture-calibration`. Reverse/forward fold parameters are cached under
  `output/model/mixture_folds/` and must be reused rather than silently refitted.
- Legacy/full release workflow: `make data && make train && make validate && make classify &&
  make figures`; each target maps to one `scripts/*.py` CLI.
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
  the sample-filter edge cases (e.g. `lien_status` restricted to `{1, 2}`, see MIGRATION_PLAN.md)
  and the validation/metric logic, not visual inspection of plots.
- Real-data end-to-end runs (`make validate`, etc.) are a manual verification step, not part of
  the automated test suite.
- The frozen logistic specification is `spline_lti__purchaser_type` with ridge `C=0.1`:
  restricted-cubic-spline `log_lti`, linear `log_county_value_to_loan`, purchaser- and loan-type
  reference indicators, and linear interactions of both continuous variables with purchaser
  type. The richer full `log_lti` spline-basis-by-purchaser challenger was rejected. Do not
  retune or expand this specification during diagnostic stages.
- Reverse validation uses four-year later training windows against every earlier labeled year,
  yielding 45 cells and horizons 1-9. Preserve equal-within-horizon, then equal-across-horizon
  aggregation where declared by the protocol.

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
