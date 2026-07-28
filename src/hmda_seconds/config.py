"""Paths, feature lists, and hyperparameters for the HMDA second-lien classifier.

Centralizes what was scattered as module-level globals across
``classify_seconds.py`` in the original research script, and adds the
train/validate/apply year partition used for out-of-time validation (see
MIGRATION_PLAN.md, "Methodological improvements", item 1).
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("HMDA_SECONDS_DATA_DIR", REPO_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PUBLIC_DIR = DATA_DIR / "public"

OUTPUT_DIR = Path(os.environ.get("HMDA_SECONDS_OUTPUT_DIR", REPO_ROOT / "output"))
MODEL_DIR = OUTPUT_DIR / "model"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

# Large, regenerable, non-public intermediate files (the concatenated
# training extract, per-year classified outputs) live outside the repo
# entirely rather than under output/, following the same convention the
# original project used for /data/hmda/. Defaults to a repo-relative
# fallback so the pipeline still runs for someone without that local
# convention; override via HMDA_SECONDS_EXTERNAL_DIR (see .env).
EXTERNAL_DATA_DIR = Path(os.environ.get("HMDA_SECONDS_EXTERNAL_DIR", DATA_DIR))
INTERMEDIATE_DIR = EXTERNAL_DATA_DIR / "intermediate"

# Directory holding one cleaned HMDA extract per year, named hmda<year>.parquet,
# in the same format produced by py_tools.datasets.hmda.load_hmda. See
# data/README.md for how these are built from raw LAR files; override locally
# to point at an already-materialized cache (e.g. an existing
# /data/hmda/save/ directory) via the HMDA_SECONDS_YEARLY_DIR env var.
HMDA_YEARLY_DIR = Path(
    os.environ.get("HMDA_SECONDS_YEARLY_DIR", RAW_DIR / "hmda" / "save")
)

# Vendored public input: FHFA's HPI_AT_BDL_county.xlsx lives directly under
# this directory; py_tools.datasets.fhfa.load('county', data_dir=...) reads
# it from there and caches a derived parquet alongside it.
FHFA_DATA_DIR = PUBLIC_DIR

MODEL_FILE = MODEL_DIR / "rf_fit.pkl"

# Lien status is reliably reported in HMDA starting in 2004. Train on the
# same 2004-2007 window as the original script; unlike the original script,
# separately hold out every other labeled year (2008-2016) for out-of-time
# validation rather than only spot-checking a single year.
TRAIN_YEARS = range(2004, 2008)
VALIDATE_YEARS = range(2008, 2017)
APPLY_YEARS = range(1990, 2017)

assert set(TRAIN_YEARS).isdisjoint(VALIDATE_YEARS)
assert set(TRAIN_YEARS) | set(VALIDATE_YEARS) <= set(APPLY_YEARS)

TRAIN_PARQUET = (
    INTERMEDIATE_DIR / f"hmda_train_{min(TRAIN_YEARS)}_{max(TRAIN_YEARS)}.parquet"
)

LABEL_VAR = "lien_status"
CONTINUOUS_VARS = ["log_lti", "log_ltv"]
CATEGORY_VARS = ["purchaser_type", "loan_type", "has_edit_status", "loan_below_10k"]

# Not used as model features, but retained (unlike the original script, which
# dropped them) so a released predicted-lien-status crosswalk can be joined
# back to a same-vintage raw LAR file by anyone who has one.
ID_VARS = ["resp_id", "seq_num"]

RF_KWARGS = {
    "n_estimators": 50,
    "max_depth": 10,
    "random_state": 17,
}

TRAIN_SIZE = 0.25
TEST_SIZE = 0.75
