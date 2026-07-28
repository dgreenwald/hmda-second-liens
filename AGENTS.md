# Repository Guidelines

## Project Overview
This repo is a standalone letter spinning off the HMDA second-lien Random Forest classifier
from the `frm` house-price paper. It trains a classifier on 2004-2007 HMDA records (the years
`lien_status` is reliably reported), validates it out-of-time against 2008-2016, and applies it
to impute lien status for 1990-2003. Scope, rationale, and the full porting/validation plan are
in `MIGRATION_PLAN.md` — read that first for context on *why* a module exists before changing
its behavior.

## Project Structure & Module Organization
- `src/hmda_seconds/` — the package: `config.py` (paths, feature lists, hyperparameters),
  `clean.py` (HMDA sample restrictions + FHFA county-HPI feature construction), `train.py`,
  `validate.py` (out-of-time validation, baselines, metrics), `classify.py` (apply the fitted
  model to all years), `diagnostics.py` (binned LTI-by-year/lien-status cells for figures).
- `scripts/` — thin argparse CLIs, one per pipeline stage, each wrapping one `src/hmda_seconds`
  entry point. These are the `Makefile` targets' bodies; keep logic in `src/`, not here.
- `data/raw/` — gitignored; user-populated HMDA LAR downloads (see `data/README.md` for exact
  source URLs). `data/public/` — small vendored public inputs (FHFA county HPI) that *are*
  committed.
- `output/` — regenerable: fitted model, figures, tables. Do not hand-edit; regenerate via
  `make`.
- `paper/` — the letter itself (`letter.tex` + `refs.bib`).
- `tests/` — pytest, synthetic fixtures only (no real HMDA data required to run the suite).

## Build, Test, and Development Commands
- `pip install -e ".[dev]"` — editable install with test tooling.
- Depends on `dgreenwald-py-tools[ml,datasets]` for the RF wrapper and HMDA/FHFA loaders — reuse
  those rather than reimplementing (see `MIGRATION_PLAN.md`, "What already exists").
- `make data && make train && make validate && make classify && make figures` — full pipeline;
  each target maps to one `scripts/*.py` CLI.
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

## Commit & Pull Request Guidelines
- Short, imperative commit subjects, first line <= 72 chars.
- One logical change per commit.
- Note any data-path or hyperparameter assumptions in the PR description.

## Security & Configuration Tips
- Never commit files under `data/raw/` or anything derived from raw HMDA microdata beyond the
  aggregated/binned outputs this letter is designed to release publicly.
- Keep environment-specific settings (`PY_TOOLS_DATA_DIR`) in `.env`/shell env, not source files.
